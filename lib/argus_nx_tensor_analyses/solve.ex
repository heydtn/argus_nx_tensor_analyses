defmodule ArgusNxTensorAnalyses.Solve do
  @moduledoc false
  # How both analyses extract, solve and place what they find: through a
  # query graph (`ArgusNxTensorAnalyses.Graph`), an analysis's extractors
  # give each module's facts, its program is solved over the relations it
  # reads after Argus's rules, each kept by what it read, and each finding
  # is placed at its call's line.

  alias Argus.Findings
  alias ArgusNxTensorAnalyses.Graph

  # The options that configure the rules, each by the input relation it
  # fills with the types it lists.
  @config_facts [unsupported_types: "unsupported_type", float_types: "float_type"]

  # The options a project gives the analyses (`unsupported_types: [:f64]`).
  @spec config_options() :: [atom()]
  def config_options, do: Keyword.keys(@config_facts)

  # Extracts the modules (atoms or `.beam` paths) with the analysis's
  # extractors and solves `program` over the facts, returning its output
  # relations' rows by name (`ArgusNxTensorAnalyses.TensorShapes.solve/3`
  # says what the options are).
  @spec solve(module(), [module() | Path.t()], Path.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def solve(analysis, modules, program, options),
    do: graph(analysis, modules, program, options, &Graph.rows/1)

  # Solves the analysis's program over the modules and returns each of its
  # findings placed at its call in the module's source.
  @spec run(module(), [module() | Path.t()], keyword()) ::
          {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(analysis, modules, options) do
    solved =
      graph(analysis, modules, analysis.rules_file(), options, fn db ->
        with {:ok, rows} <- Graph.rows(db),
             {:ok, lines} <- Graph.lines(db),
             do: {:ok, rows, lines}
      end)

    with {:ok, rows, lines} <- solved do
      beams = beams(modules)
      findings = Findings.build(analysis, rows)
      {:ok, Enum.map(findings, &place(&1, beams, lines))}
    end
  end

  # `demand` over the graph, set for the analysis over the modules and for
  # `program` after the Argus files it builds on.
  defp graph(analysis, modules, program, options, demand) do
    roots = argus_includes() ++ [Path.expand(program)]
    cache = Keyword.get(options, :cache)
    Graph.run(analysis, modules, roots, config_facts(options), cache, demand)
  end

  # Each relation the options fill, by its file's text.
  defp config_facts(options) do
    Map.new(@config_facts, fn {option, relation} ->
      {relation, options |> Keyword.get(option, []) |> Enum.map_join(&(type_name(&1) <> "\n"))}
    end)
  end

  # A type as the rules name it: `f64` for `:f64` or `{:f, 64}`.
  defp type_name({family, size}) when is_atom(family) and is_integer(size), do: "#{family}#{size}"
  defp type_name(type) when is_atom(type), do: Atom.to_string(type)
  defp type_name(type) when is_binary(type), do: type

  @doc false
  # The Argus files a program is solved after: its facts and its reach
  # components, not all of `clientlib/imports.dl`, which declares hundreds
  # of relations these rules never read and Soufflé's front end walks the
  # program once for each. The few other words the rules read are copied
  # in `priv/argus.dl`. Souffle resolves an `.include` against the file it
  # is in, and Argus's are wherever Mix put the dependency.
  def argus_includes do
    Enum.map(~w(base.dl clientlib/reach.dl), &Application.app_dir(:argus_beam, "priv/dl/#{&1}"))
  end

  # Each module's beam and the source file it was compiled from.
  defp beams(modules) do
    for module <- modules,
        path <- [beam_path(module)],
        {:ok, {name, [compile_info: info]}} <- [:beam_lib.chunks(path, [:compile_info])],
        into: %{},
        do:
          {name,
           %{path: List.to_string(path), source: info |> Keyword.get(:source) |> source_path()}}
  end

  defp beam_path(module) when is_atom(module), do: :code.which(module)
  defp beam_path(path), do: String.to_charlist(path)

  defp source_path(nil), do: nil
  defp source_path(source), do: List.to_string(source)

  defp place(finding, beams, lines) do
    %Argus.Located{
      finding: finding,
      file: source_file(beams, finding.module),
      line: line_at(finding, beams, lines),
      end_line: nil,
      related:
        Enum.map(finding.related, fn frame ->
          %{
            file: source_file(beams, frame.module),
            line: line_at(frame, beams, lines),
            end_line: nil
          }
        end)
    }
  end

  # Where a finding or frame is, as Argus places its own (`Argus.Located`):
  # its instruction's line (`Argus.Lines`), else its function's first,
  # else the line its module is declared on, else 1.
  defp line_at(anchored, beams, lines) do
    anchor_line(anchored, lines) || declaration_line(anchored, beams) || 1
  end

  defp anchor_line(%{instr: %Argus.InstrId{} = instruction}, lines),
    do: Argus.Lines.resolve(lines, instruction)

  defp anchor_line(%{mfa: {_module, _name, _arity} = mfa}, lines),
    do: Argus.Lines.resolve(lines, mfa)

  defp anchor_line(_anchored, _lines), do: nil

  defp declaration_line(%{module: module}, beams) do
    case Map.get(beams, module) do
      %{path: path} -> Argus.Lines.declaration_line(path)
      nil -> nil
    end
  end

  defp source_file(beams, module) do
    case Map.get(beams, module) do
      %{source: source} -> source
      nil -> nil
    end
  end
end
