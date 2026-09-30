defmodule ArgusNxTensorAnalyses do
  @moduledoc """
  [Argus](https://github.com/QuinnWilton/argus) analyses of code that uses
  [Nx](https://github.com/elixir-nx/nx), run over a project's compiled
  modules.

  What they find is reported in categories, each an analysis to Argus
  (`error[argus.nx_shapes]`) that a run selects by name (`analyses/0`):
  the Nx categories (`nx_shapes`, `nx_names`, `nx_math`, `nx_types`,
  `nx_options`, `nx_indices`, `nx_traced`, `nx_gradients`,
  `nx_containers`, `nx_random`, `nx_freed` and `nx_serving`), and `emlx`,
  for a project whose tensors live on EMLX. `docs/checks.md` lists each
  category's findings. Two engines find them:

    * `ArgusNxTensorAnalyses.TensorShapes` — the Nx categories: Nx calls
      whose shapes, types or options Nx rejects, calls Nx accepts where
      the code does not line its axes up, math that can give an infinity
      or a NaN, and misuse of traced code, gradients, random keys,
      containers and servings.
    * `ArgusNxTensorAnalyses.EMLX` — `emlx`: Nx calls EMLX computes
      differently from BinaryBackend and EXLA, and calls that get tensors
      of two backends that cannot meet.

  `run/3` runs the engines a set of categories needs, and solves each
  engine's program once, whichever of its categories are asked for.

  Argus runs only the analyses it ships, so `mix argus_nx_tensor_analyses`
  runs these beside Argus's own and reports both as one; aliased as
  `argus`, it takes `mix argus`'s place. A run that names no category runs
  `default_analyses/0`, or the ones the project's `mix.exs` chooses.
  """

  alias ArgusNxTensorAnalyses.Categories
  alias ArgusNxTensorAnalyses.EMLX
  alias ArgusNxTensorAnalyses.TensorShapes

  @doc "The categories the findings are reported under, in the order a run lists them."
  @spec analyses() :: [atom()]
  def analyses, do: Categories.names()

  @doc """
  The categories a run with none named runs: every one but `emlx`, about a
  single backend's differences, which a project on another backend has no
  use for.
  """
  @spec default_analyses() :: [atom()]
  def default_analyses, do: Categories.default()

  @doc """
  The categories `selection` names: category names (`:nx_shapes`) and the
  sets `:default` (`default_analyses/0`) and `:all` (`analyses/0`), in
  `analyses/0`'s order.
  """
  @spec select([atom()]) :: {:ok, [atom()]} | {:error, String.t()}
  def select(selection) do
    names = analyses()

    case Enum.reject(selection, &(&1 in [:default, :all] or &1 in names)) do
      [] ->
        {:ok, Enum.filter(names, &selected?(&1, selection))}

      unknown ->
        {:error,
         "unknown analyses #{Enum.map_join(unknown, ", ", &inspect/1)}; " <>
           "choose from #{Enum.map_join([:default, :all | names], ", ", &inspect/1)}"}
    end
  end

  defp selected?(category, selection) do
    :all in selection or category in selection or
      (:default in selection and category in default_analyses())
  end

  @doc """
  Solves the modules (atoms or `.beam` paths) for the categories
  `selection` names, as `select/1` takes them, and returns each selected
  category's findings placed at their calls in the modules' source, an
  empty list for a category that finds nothing.

  Takes `ArgusNxTensorAnalyses.TensorShapes.solve/3`'s options, but for
  `:cache`: a directory under which each engine's program keeps its work
  in a directory of its own, so that runs asking for other categories of
  one program share what it keeps. Default: nil, a run that keeps nothing.
  """
  @spec run([module() | Path.t()], [atom()], keyword()) ::
          {:ok, %{atom() => [Argus.Located.t()]}} | {:error, term()}
  def run(modules, selection, options \\ []) do
    with {:ok, categories} <- select(selection),
         {:ok, placed} <- placed(modules, categories, options) do
      found = Enum.group_by(placed, & &1.finding.analysis)
      {:ok, Map.new(categories, &{&1, Map.get(found, &1, [])})}
    end
  end

  # The findings of each engine the categories need, placed: the Nx
  # engine's for the Nx categories, EMLX's for `emlx`, each only of those
  # categories.
  defp placed(modules, categories, options) do
    engines =
      for {engine, needed?} <- [
            {TensorShapes, Enum.any?(categories, &(&1 != :emlx))},
            {EMLX, :emlx in categories}
          ],
          needed?,
          do: engine

    Enum.reduce_while(engines, {:ok, []}, fn engine, {:ok, placed} ->
      case engine.run(modules, program_options(options, engine)) do
        {:ok, found} ->
          {:cont, {:ok, placed ++ Enum.filter(found, &(&1.finding.analysis in categories))}}

        {:error, _reason} = error ->
          {:halt, error}
      end
    end)
  end

  # The options an engine runs with: `:cache` its program's own directory
  # under the one given, named for the program.
  defp program_options(options, engine) do
    case Keyword.get(options, :cache) do
      nil -> options
      cache -> Keyword.put(options, :cache, Path.join(cache, program_name(engine)))
    end
  end

  defp program_name(engine), do: Path.basename(engine.rules_file(), ".dl")
end
