defmodule Mix.Tasks.ArgusNxTensorAnalyses do
  @shortdoc "Runs argus's analyses and the Nx tensor analyses"

  @moduledoc """
  `mix argus`, with this package's analyses, the categories of
  `ArgusNxTensorAnalyses.analyses/0`, beside argus's own. Argus runs only
  the analyses it ships, so this task runs argus as `mix argus` does, then
  these over the project, and reports both as one. Aliased as `argus` in
  the project's `mix.exs` (`aliases: [argus: "argus_nx_tensor_analyses"]`),
  it takes `mix argus`'s place:

      mix argus                       # argus's configured analyses and the default ones here
      mix argus --all                 # every analysis, argus's and these
      mix argus emlx                  # the EMLX category alone
      mix argus nx_shapes ets         # the Nx shape category and argus's `ets`
      mix argus --list                # what's available

  It takes `mix argus`'s command line. `--format`, `--fail-above` and
  `--color` apply to every finding. These analyses read the project's own
  beams, not its dependencies', and keep what they find under
  `_build/<env>/argus_nx_tensor_analyses`, a directory for each program
  they solve, until the code they read changes; souffle must be on
  `PATH`. A run solves each program once, whichever of its categories it
  asks for (`ArgusNxTensorAnalyses.run/3`).

  The project configures these analyses in its `mix.exs`, under
  `argus_nx_tensor_analyses:` in `project/0`. `analyses` chooses the
  categories a run that names none runs, by name or as the sets
  `:default` (`ArgusNxTensorAnalyses.default_analyses/0`, the default) and
  `:all`.
  `unsupported_types` lists the tensor types its backend lacks, and a call
  that makes one is reported. `float_types` lists the float types the code
  may run at, and a type the code reads from configuration is checked as
  each of them:

      argus_nx_tensor_analyses: [
        analyses: [:default, :emlx],
        unsupported_types: [:f64],
        float_types: [:f16, :bf16, :f32]
      ]
  """

  use Mix.Task

  alias Argus.CLI.Options
  alias ArgusNxTensorAnalyses.Categories

  @impl Mix.Task
  def run(arguments) do
    case Options.parse(arguments, :mix) do
      {:ok, %Options{command: :analyze} = options} ->
        analyze(options)

      {:ok, %Options{command: :list}} ->
        Mix.Task.run("argus", arguments)
        IO.puts("\nThe Nx tensor analyses (✓ runs by default, mix argus --all runs all):\n")

        for category <- ArgusNxTensorAnalyses.analyses() do
          mark = if category in ArgusNxTensorAnalyses.default_analyses(), do: "✓", else: " "
          IO.puts("  #{mark} #{category} — #{Categories.description(category)}")
        end

      {:ok, _help} ->
        Mix.Task.run("argus", arguments)

      {:error, message} ->
        Mix.raise("argus: " <> message)
    end
  end

  defp analyze(options) do
    {:ok, _apps} = Application.ensure_all_started(:telemetry)

    categories = ArgusNxTensorAnalyses.analyses()
    selected = selected(options, categories)

    options = %{
      options
      | analyses: options.analyses && Enum.reject(options.analyses, &(&1 in categories))
    }

    compile!()

    config = Argus.CLI.override(Argus.Config.load(), options)
    result = Argus.Driver.run(config, force: options.force)

    if Argus.Driver.Result.souffle_missing?(result) do
      Mix.raise(
        "argus: souffle binary not found on PATH — install souffle " <>
          "(https://souffle-lang.github.io) to run the analyses"
      )
    end

    beams = Path.wildcard(Path.join(Mix.Project.compile_path(), "*.beam"))
    cache = Path.join(Mix.Project.build_path(), "argus_nx_tensor_analyses")

    # Each category's findings, or why the solve it needs failed.
    located =
      case ArgusNxTensorAnalyses.run(beams, selected, [cache: cache] ++ project_options()) do
        {:ok, placed} -> Map.new(placed, fn {category, found} -> {category, {:ok, found}} end)
        {:error, _reason} = error -> Map.new(selected, &{&1, error})
      end

    result = %{result | located: Map.merge(result.located, located)}
    cwd = File.cwd!()
    notices = Argus.Report.Notice.from_result(result, config, cwd)
    %{entries: entries} = Argus.CLI.report(result, notices, config, options, cwd, cwd)

    if options.fail_above && length(entries) > options.fail_above do
      Mix.raise("argus: #{length(entries)} findings exceed --fail-above #{options.fail_above}")
    end
  end

  # The categories this run asks for: all of them with `--all`, the ones
  # the command line names when it names any (argus's analyses among
  # them), and otherwise the ones the project's `mix.exs` chooses
  # (`:default` unless it says).
  defp selected(%Options{all: true}, categories), do: categories

  defp selected(%Options{analyses: named}, categories) when is_list(named),
    do: Enum.filter(categories, &(&1 in named))

  defp selected(_options, _categories) do
    configured =
      Mix.Project.config()
      |> Keyword.get(:argus_nx_tensor_analyses, [])
      |> Keyword.get(:analyses, [:default])

    case ArgusNxTensorAnalyses.select(configured) do
      {:ok, selected} -> selected
      {:error, message} -> Mix.raise("argus_nx_tensor_analyses: " <> message)
    end
  end

  # The options the project gives these analyses in its `mix.exs`, under
  # `argus_nx_tensor_analyses:` (`unsupported_types: [:f64]`,
  # `float_types: [:f16, :bf16, :f32]`).
  defp project_options do
    Mix.Project.config()
    |> Keyword.get(:argus_nx_tensor_analyses, [])
    |> Keyword.take(ArgusNxTensorAnalyses.Solve.config_options())
  end

  # As `mix argus` compiles: the argus compiler fails `mix compile` on its
  # own findings, which are what this task reports; any other compiler's
  # error means the beams are not the source's.
  defp compile! do
    case Mix.Task.run("compile", ["--no-prune-code-paths", "--return-errors"]) do
      {:error, diagnostics} ->
        if Enum.any?(diagnostics, &(&1.severity == :error and &1.compiler_name != "argus")) do
          Mix.raise("argus: the project does not compile; fix the errors above first")
        end

      _compiled ->
        :ok
    end
  end
end
