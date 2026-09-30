defmodule ArgusNxTensorAnalyses.Solve do
  @moduledoc false
  # How both analyses extract, solve and place what they find: an
  # analysis's extractors write the modules' facts, its program is solved
  # over them after Argus's rules, the rows are kept under a digest of what
  # the solve reads, and each finding is placed at its call's line.

  alias Argus.Findings

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
    do: with_facts(analysis, modules, &solve_facts(&1, program, options))

  # Solves the analysis's program over the modules and returns each of its
  # findings placed at its call in the module's source.
  @spec run(module(), [module() | Path.t()], keyword()) ::
          {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(analysis, modules, options) do
    solved =
      with_facts(analysis, modules, fn directory ->
        with {:ok, rows} <- solve_facts(directory, analysis.rules_file(), options),
             do: {:ok, rows, Argus.Lines.from_facts_dir(directory)}
      end)

    with {:ok, rows, lines} <- solved do
      beams = beams(modules)
      findings = Findings.build(analysis, rows)
      {:ok, Enum.map(findings, &place(&1, beams, lines))}
    end
  end

  # Extracts the modules into a directory of facts named for the analysis,
  # hands it to `solve`, and removes it.
  defp with_facts(analysis, modules, solve) do
    directory =
      Path.join(
        System.tmp_dir!(),
        "#{analysis.name()}_#{Base.url_encode64(:crypto.strong_rand_bytes(9), padding: false)}"
      )

    try do
      with {:ok, directory} <- extract(analysis, modules, directory), do: solve.(directory)
    after
      File.rm_rf!(directory)
    end
  end

  defp solve_facts(directory, program, options) do
    for {option, relation} <- @config_facts do
      File.write!(
        Path.join(directory, "#{relation}.facts"),
        options
        |> Keyword.get(option, [])
        |> Enum.map(&[type_name(&1), "\n"])
      )
    end

    case Keyword.get(options, :cache) do
      nil -> solve_rules(directory, program)
      cache -> solve_cached(directory, program, cache)
    end
  end

  # A type as the rules name it: `f64` for `:f64` or `{:f, 64}`.
  defp type_name({family, size}) when is_atom(family) and is_integer(size), do: "#{family}#{size}"
  defp type_name(type) when is_atom(type), do: Atom.to_string(type)
  defp type_name(type) when is_binary(type), do: type

  defp extract(analysis, modules, directory) do
    extractors = analysis.extractors()

    with {:ok, directory} <- Argus.Pipeline.run(modules, directory, extractors: extractors) do
      # Souffle fails on a missing input file, and an extractor writes a
      # relation's file only when it has rows.
      for extractor <- extractors,
          relation <- extractor.relations(),
          path = Path.join(directory, "#{relation}.facts"),
          not File.exists?(path),
          do: File.write!(path, "")

      {:ok, directory}
    end
  end

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

  # The program after the Argus rules it builds on.
  #
  # The call graph (Argus's stage 0) is derived here, and the solve told
  # it is provided: left to itself, `run_rules/3` would also learn
  # whether the program reads Argus's process points-to, which it never
  # does, by compiling the program, which takes most of a solve's time.
  defp solve_rules(directory, program) do
    wrapper = directory <> ".dl"
    includes = argus_includes() ++ [Path.expand(program)]
    File.write!(wrapper, Enum.map(includes, &~s(.include "#{&1}"\n)))

    try do
      with :ok <- Argus.Analysis.derive_stage0(directory),
           do: Argus.Analysis.run_rules(directory, {:custom, wrapper}, stage0: :provided)
    after
      File.rm(wrapper)
    end
  end

  defp solve_cached(directory, program, cache) do
    entry = Path.join(cache, digest(directory, program) <> ".json")

    with {:ok, text} <- File.read(entry),
         {:ok, rows} <- decode(text) do
      {:ok, rows}
    else
      _missing ->
        with {:ok, rows} <- solve_rules(directory, program) do
          File.rm_rf!(cache)
          File.mkdir_p!(cache)
          File.write!(entry, Jason.encode!(rows))
          {:ok, rows}
        end
    end
  end

  # The rows as `solve_rules/2` gave them: relation names to rows of
  # strings. Anything else in the file is ignored, and solved afresh.
  defp decode(text) do
    case Jason.decode(text) do
      {:ok, %{} = rows} -> if Enum.all?(rows, &rows?/1), do: {:ok, rows}, else: :error
      _other -> :error
    end
  end

  defp rows?({relation, rows}) when is_binary(relation) and is_list(rows),
    do: Enum.all?(rows, &(is_list(&1) and Enum.all?(&1, fn column -> is_binary(column) end)))

  defp rows?(_entry), do: false

  # Every relation's facts but the lines, which no rule reads and which a
  # comment moves; the program and every file it includes, without their
  # comments, as Argus keys a program; the Argus whose declarations they
  # build on; and the solver.
  defp digest(directory, program) do
    directory
    |> File.ls!()
    |> Enum.reject(&(&1 == "line_info.facts"))
    |> Enum.sort()
    |> Enum.reduce(:crypto.hash_init(:sha256), fn file, hash ->
      hash
      |> :crypto.hash_update(file <> "\n")
      |> :crypto.hash_update(File.read!(Path.join(directory, file)))
    end)
    |> hash_program(program)
    |> :crypto.hash_update(to_string(Application.spec(:argus_beam, :vsn)))
    |> hash_solver(Argus.Souffle.executable())
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  # The solver by its version, as Argus names it in its own keys, and by
  # the file it runs: a build can print an empty version (`Version: `),
  # and its file still tells it from another. Without one on PATH there
  # is no solve to cache.
  defp hash_solver(state, nil), do: :crypto.hash_update(state, "no solver")

  defp hash_solver(state, executable) do
    state
    |> :crypto.hash_update(Argus.Souffle.version(executable))
    |> :crypto.hash_update(File.read!(executable))
  end

  # The program's files, found as Souffle finds an include, each by the
  # name the program gives it and its text as a solve reads it.
  defp hash_program(state, program) do
    program
    |> Argus.Souffle.Program.program_files()
    |> Enum.reduce(state, fn {spelled, path}, hash ->
      hash
      |> :crypto.hash_update(spelled <> "\n")
      |> :crypto.hash_update(path |> File.read!() |> Argus.Souffle.Program.uncommented())
    end)
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
