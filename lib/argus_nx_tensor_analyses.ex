defmodule ArgusNxTensorAnalyses do
  @moduledoc """
  [Argus](https://github.com/QuinnWilton/argus) analyses of code that uses
  [Nx](https://github.com/elixir-nx/nx), run over a project's compiled
  modules.

    * `ArgusNxTensorAnalyses.TensorShapes` — Nx calls whose shapes, types
      or options Nx rejects, calls Nx accepts where the code does not line
      its axes up, math that can give an infinity or a NaN, and misuse of
      traced code, gradients, random keys, containers and servings.
    * `ArgusNxTensorAnalyses.EMLX` — Nx calls EMLX computes differently
      from BinaryBackend and EXLA, and calls that get tensors of two
      backends that cannot meet: for a project whose tensors live on EMLX.

  Argus runs only the analyses it ships, so `mix argus_nx_tensor_analyses`
  runs these beside Argus's own and reports both as one; aliased as
  `argus`, it takes `mix argus`'s place. A run that names no analysis runs
  `default_analyses/0`, or the ones the project's `mix.exs` chooses.
  """

  @doc "The analyses this package defines, each an `Argus.Analysis`."
  @spec analyses() :: [module()]
  def analyses, do: [ArgusNxTensorAnalyses.TensorShapes, ArgusNxTensorAnalyses.EMLX]

  @doc """
  The analyses a run with none named runs: every one but those about a
  single backend's differences, which a project on another backend has no
  use for.
  """
  @spec default_analyses() :: [module()]
  def default_analyses, do: [ArgusNxTensorAnalyses.TensorShapes]

  @doc """
  The analyses `selection` names: analysis names (`:tensor_emlx`) and the
  sets `:default` (`default_analyses/0`) and `:all` (`analyses/0`), in
  `analyses/0`'s order.
  """
  @spec select([atom()]) :: {:ok, [module()]} | {:error, String.t()}
  def select(selection) do
    names = Enum.map(analyses(), & &1.name())

    case Enum.reject(selection, &(&1 in [:default, :all] or &1 in names)) do
      [] ->
        {:ok, Enum.filter(analyses(), &selected?(&1, selection))}

      unknown ->
        {:error,
         "unknown analyses #{Enum.map_join(unknown, ", ", &inspect/1)}; " <>
           "choose from #{Enum.map_join([:default, :all | names], ", ", &inspect/1)}"}
    end
  end

  defp selected?(analysis, selection) do
    :all in selection or analysis.name() in selection or
      (:default in selection and analysis in default_analyses())
  end
end
