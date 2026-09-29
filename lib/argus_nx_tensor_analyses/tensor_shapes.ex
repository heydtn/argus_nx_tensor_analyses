defmodule ArgusNxTensorAnalyses.TensorShapes do
  @moduledoc """
  Tensor shapes that do not fit where they meet, found in compiled code by
  the Datalog program in `priv/tensor_shapes.dl`: Nx calls whose operands
  Nx rejects, and calls Nx accepts where the code does not line its axes
  up.

  The program reads what `ArgusNxTensorAnalyses.TensorShapes.ShapeFlow`
  extracts besides Argus's own facts, and Argus's analyses run only the
  extractors it ships with, so this module extracts and solves itself
  (`solve/3`),
  turns the program's rows into findings (`c:Argus.Analysis.finding/2`),
  and places them at their calls' lines (`run/2`) for `Argus.Report` to
  render.

  A finding says what the call does wrong in its title, labels the call
  with the shapes it gets, and notes why Nx rejects them (or why the code
  should not rely on Nx accepting them) with what Nx raises. Its related
  frames point at the calls that make each operand's shape, then at the
  calls that bring the operands into the function.

  Which way it errs: a shape the rules cannot follow is not known, and an
  operation over one gives no shape, so what the analysis cannot see
  yields no finding (it errs quiet). Branches it cannot tell apart are
  the exception: a value that reaches a call along several paths has
  every shape it can arrive with, and a finding between two of them has
  certainty `on_some_path`, a warning, though the program's branches may
  never combine them (it errs loud there, on purpose).
  """

  @behaviour Argus.Analysis

  alias Argus.Findings
  alias ArgusNxTensorAnalyses.Finding
  alias ArgusNxTensorAnalyses.TensorShapes.ShapeFlow
  alias ArgusNxTensorAnalyses.TensorShapes.Wording
  alias ArgusNxTensorAnalyses.TensorShapes.Wording.Causes

  import ArgusNxTensorAnalyses.Text

  @external_resource Path.expand("../../priv/tensor_shapes.dl", __DIR__)

  @impl true
  def name, do: :tensor_shapes

  @impl true
  def description, do: "Tensor shapes that do not fit where they meet"

  @impl true
  def rules_file, do: Application.app_dir(:argus_nx_tensor_analyses, "priv/tensor_shapes.dl")

  @impl true
  def extractors, do: [ShapeFlow]

  @impl true
  def output_relations do
    [
      %{
        name: :tensor_shape_mismatch,
        fields: finding_fields("the error Nx raises"),
        key: [:id, :kind],
        doc: "An Nx call whose operand shapes Nx rejects."
      },
      %{
        name: :tensor_shape_mismatch_via,
        fields: frame_fields(),
        evidence: %{of: :tensor_shape_mismatch, on: [:id, :kind], limit: 8},
        doc:
          "Where an operand of a tensor shape mismatch gets its shape, or a call on the way to it."
      },
      %{
        name: :tensor_axis_misalignment,
        fields: finding_fields("what the code does not line up"),
        key: [:id, :kind],
        doc: "An Nx call Nx accepts where the code does not line its axes up."
      },
      %{
        name: :tensor_axis_misalignment_via,
        fields: frame_fields(),
        evidence: %{of: :tensor_axis_misalignment, on: [:id, :kind], limit: 8},
        doc:
          "Where an operand of a tensor axis misalignment gets its shape, or a call on the way to it."
      },
      %{
        name: :tensor_nonfinite_result,
        fields: [
          {:id, :instr_id, "the Nx call"},
          {:func, :func_id, "the function making it"},
          {:operation, :symbol, "the Nx function, as Nx.divide/2"},
          {:kind, :symbol, "what goes wrong: divide_by_zero, ..."},
          {:cause, :symbol, "how the operand gets there: square, comparison, index, ..."},
          {:origin, :symbol, "the Nx call whose math lets it, or empty for a written value"},
          {:origin_operation, :symbol, "that call's Nx function, as Nx.multiply/2"}
        ],
        key: [:id, :kind],
        doc: "An Nx call whose result the code's own math can make infinite or NaN."
      },
      %{
        name: :tensor_type_error,
        fields: [
          {:id, :instr_id, "the Nx call"},
          {:func, :func_id, "the function making it"},
          {:operation, :symbol, "the Nx function, as Nx.take/2"},
          {:kind, :symbol, "what goes wrong: non_integer_operand or unsupported_type"},
          {:subject, :symbol, "the operand's class (float, complex), or the type made (f64)"},
          {:position, :number, "the operand's argument position, or -1 for a type made"},
          {:certain, :number, "1 where the operand is never an integer in some context"},
          {:origin, :symbol, "the Nx call that makes the operand what it is, or empty"},
          {:origin_operation, :symbol, "that call's Nx function, as Nx.divide/2"}
        ],
        key: [:id, :kind, :position],
        doc:
          "An Nx call Nx rejects for its operand's type, or that makes a tensor of a type the backend does not support."
      },
      %{
        name: :tensor_call_error,
        fields: [
          {:id, :instr_id, "the call"},
          {:func, :func_id, "the function making it"},
          {:operation, :symbol,
           "the function it calls, as Nx.sum/2, or empty for an instruction"},
          {:kind, :symbol, "the rule it breaks: unknown_option, ..."},
          {:detail, :symbol, "what breaks it: the option, the value, ..."},
          {:origin, :symbol, "the call it comes from, or empty"},
          {:origin_operation, :symbol, "that call's function, as Nx.divide/2"}
        ],
        key: [:id, :kind, :detail],
        doc:
          "A call Nx rejects, or that computes something other than what the code means, for a reason other than its operands' shapes, math or types."
      }
    ]
  end

  defp finding_fields(detail) do
    [
      {:id, :instr_id, "the Nx call"},
      {:func, :func_id, "the function making it"},
      {:operation, :symbol, "the Nx function, as Nx.add/2"},
      {:kind, :symbol, "the rule it breaks: broadcast, names, size_variables, ..."},
      {:detail, :symbol, detail},
      {:certainty, :symbol, "always, or on_some_path when the operands also arrive otherwise"},
      {:operands, :symbol, "the shapes the call gets, `;` between them, where each holds one"}
    ]
  end

  defp frame_fields do
    [
      {:id, :instr_id, "the Nx call"},
      {:kind, :symbol, "the rule it breaks"},
      {:order, :symbol,
       "1 then the operand's position for where it gets its shape, 2 for a call"},
      {:at, :instr_id, "the call the frame is at"},
      {:func, :func_id, "the function making that call"},
      {:how, :symbol,
       "made (an Nx call makes the operand), returned (a call returns it), or call"},
      {:subject, :symbol,
       "the Nx function the operand is an argument of, or the function the call runs"},
      {:position, :number, "the operand's argument position, or -1 for a call"},
      {:shown, :symbol, "the operand's shape, where it holds one"}
    ]
  end

  @impl true
  def finding(
        :tensor_shape_mismatch = relation,
        [id, _func, operation, kind, detail, certainty, operands] = row
      ) do
    %{title: title, why: why, help: help} = wording = violation(kind)

    raises =
      case certainty do
        "always" -> "Nx raises"
        _some_path -> "On some path to this call, Nx raises"
      end

    Findings.new(
      Finding.severity(relation, row, wording),
      "#{operation} #{title}",
      "#{why} #{raises}: #{detail}.",
      at: Findings.at_instr(id),
      at_label: gets(operands) || "Nx raises here",
      help: [help]
    )
  end

  def finding(
        :tensor_axis_misalignment = relation,
        [id, _func, operation, kind, detail, certainty, operands] = row
      ) do
    %{title: title, why: why, help: help} = wording = violation(kind)

    where =
      case certainty do
        "always" -> ""
        _some_path -> "on some path to this call "
      end

    Findings.new(
      Finding.severity(relation, row, wording),
      "#{operation} #{title}",
      "Nx accepts this, but #{where}#{detail}. #{why}",
      at: Findings.at_instr(id),
      at_label: gets(operands) || "the axes meet here",
      help: [help]
    )
  end

  def finding(
        :tensor_nonfinite_result = relation,
        [_id, _func, operation, kind, cause | _rest] = row
      ) do
    wording = Wording.hazard(kind, cause, operation) || unknown_hazard(cause)
    Finding.build(relation, row, wording)
  end

  def finding(
        :tensor_type_error = relation,
        [_id, _func, operation, kind, subject, position, certain | _rest] = row
      ) do
    wording =
      Wording.type_error(kind, subject, operation, position, certain) ||
        unknown_wording(kind, subject)

    Finding.build(relation, row, wording)
  end

  def finding(:tensor_call_error = relation, [_id, _func, operation, kind, detail | _rest] = row) do
    wording = Wording.call_error(kind, detail, operation) || unknown_wording(kind, detail)
    Finding.build(relation, row, wording)
  end

  # A kind the program has and no wording module describes still reads.
  defp unknown_wording(kind, detail) do
    %{
      title: "breaks a rule of Nx's (#{kind})",
      detail: "The analysis reports #{kind}: #{detail}.",
      label: "here",
      help: "see what Nx expects of this call",
      frame: "because of this"
    }
  end

  defp unknown_hazard(cause) do
    %{
      title: "can give an infinity or a NaN",
      detail: "Its operand can reach a value the call is not defined at: #{Causes.zero(cause)}.",
      label: "here",
      help: "keep the operand where the call is defined",
      frame: "because of this"
    }
  end

  defp violation(kind) do
    Wording.violation(kind) ||
      %{
        title: "rejects these tensor shapes",
        why: "The operands' shapes do not fit the operation.",
        help: "make the operands' shapes fit the operation"
      }
  end

  @impl true
  def evidence(relation, [_id, _kind, _order, at, _func, how, subject, position, shown])
      when relation in [:tensor_shape_mismatch_via, :tensor_axis_misalignment_via] do
    Findings.related(frame_label(how, subject, position, shown), Findings.at_instr(at))
  end

  # The label under the call: the shapes it gets, or nil where some
  # operand holds several.
  defp gets(""), do: nil

  defp gets(operands), do: "gets #{operands |> String.split(";") |> join("and")}"

  defp frame_label("call", callee, _position, _shown),
    do: "calls #{function_name(callee)} with these shapes"

  defp frame_label(how, operation, position, shown) do
    verb = if how == "made", do: "makes", else: "returns"
    argument = "the #{ordinal(position, 4)} argument of #{operation}"

    case shown do
      "" -> "#{verb} #{argument}"
      shape -> "#{verb} #{shape}, #{argument}"
    end
  end

  # A function as its reader writes it: a `defn`'s body, which the compiler
  # names `__defn:name__`, by its name in the source.
  defp function_name(function) do
    function
    |> String.replace(~r/:__defn:(.+)__\//, ":\\1/")
    |> Findings.call_name()
  end

  @doc """
  Extracts the modules (atoms or `.beam` paths) with the flow extractor
  and solves `program` over the facts, returning its output relations'
  rows by name. `program` defaults to this analysis's; a program that
  includes it can read its relations too.

  ## Options

    * `:cache` — a directory to keep the rows in, under a digest of what
      the solve reads (the facts extracted from the modules, the program
      and rules without their comments, Argus's version, the solver's):
      a solve that would read the same reads them back instead, so an
      edit that leaves the compiled code as it was (a comment, a doc, a
      comment in the rules) solves nothing. Only the latest rows are
      kept. Default: nil, no cache.
    * `:unsupported_types` — the tensor types the backend the code runs
      on does not support, as Nx names them (`:f64`, `{:f, 64}`): a call
      that makes a tensor of one is reported. EMLX, for one, computes f64
      as f32. Default: none.
    * `:float_types` — the float types the code may run at, as Nx names
      them (`[:f16, :bf16, :f32]`): a tensor whose type the code reads from
      configuration (a `type:` or a type argument it does not write) is
      checked as each of them, so an f16 run's overflows and a bf16 run's
      lost precision are reported. Default: none, and such a type is not
      known.
  """
  @spec solve([module() | Path.t()], Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def solve(modules, program \\ rules_file(), options \\ []),
    do: with_facts(modules, &solve_facts(&1, program, options))

  @doc """
  Solves the program over the modules (atoms or `.beam` paths) and returns
  each finding placed at its call in the module's source. Takes `solve/3`'s
  options, and:

    * `:analysis` — the `Argus.Analysis` whose program (its
      `c:Argus.Analysis.rules_file/0`) is solved and whose output
      relations the findings are built from, for an analysis whose program
      includes this one's. Default: this analysis.
  """
  @spec run([module() | Path.t()], keyword()) :: {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(modules, options \\ []) do
    analysis = Keyword.get(options, :analysis, __MODULE__)

    solved =
      with_facts(modules, fn directory ->
        with {:ok, rows} <- solve_facts(directory, analysis.rules_file(), options),
             do: {:ok, rows, Argus.Lines.from_facts_dir(directory)}
      end)

    with {:ok, rows, lines} <- solved do
      beams = beams(modules)
      findings = Findings.build(analysis, rows)
      {:ok, Enum.map(findings, &place(&1, beams, lines))}
    end
  end

  # Extracts the modules into a directory of facts, hands it to `solve`,
  # and removes it.
  defp with_facts(modules, solve) do
    directory =
      Path.join(
        System.tmp_dir!(),
        "tensor_shapes_#{Base.url_encode64(:crypto.strong_rand_bytes(9), padding: false)}"
      )

    try do
      with {:ok, directory} <- extract(modules, directory), do: solve.(directory)
    after
      File.rm_rf!(directory)
    end
  end

  defp solve_facts(directory, program, options) do
    File.write!(
      Path.join(directory, "unsupported_type.facts"),
      options
      |> Keyword.get(:unsupported_types, [])
      |> Enum.map(&[type_name(&1), "\n"])
    )

    File.write!(
      Path.join(directory, "float_type.facts"),
      options
      |> Keyword.get(:float_types, [])
      |> Enum.map(&[type_name(&1), "\n"])
    )

    case Keyword.get(options, :cache) do
      nil -> solve_rules(directory, program)
      cache -> solve_cached(directory, program, cache)
    end
  end

  # The rules files `rules_file/0` includes, as they are read.
  defp hash_rules_files(hash) do
    rules_file()
    |> Path.dirname()
    |> Path.join("tensor_shapes/*.dl")
    |> Path.wildcard()
    |> Enum.sort()
    |> Enum.reduce(hash, fn file, acc ->
      :crypto.hash_update(acc, file |> File.read!() |> Argus.Souffle.Program.uncommented())
    end)
  end

  # A type as the rules name it: `f64` for `:f64` or `{:f, 64}`.
  defp type_name({family, size}) when is_atom(family) and is_integer(size), do: "#{family}#{size}"
  defp type_name(type) when is_atom(type), do: Atom.to_string(type)
  defp type_name(type) when is_binary(type), do: type

  defp extract(modules, directory) do
    with {:ok, directory} <- Argus.Pipeline.run(modules, directory, extractors: [ShapeFlow]) do
      # Souffle fails on a missing input file, and the extractor writes a
      # relation's file only when it has rows.
      for relation <- ShapeFlow.relations(),
          path = Path.join(directory, "#{relation}.facts"),
          not File.exists?(path),
          do: File.write!(path, "")

      {:ok, directory}
    end
  end

  # The program after the Argus rules it builds on: its facts, its call
  # graph and its shared words (`clientlib/imports.dl`). Souffle resolves
  # an `.include` against the file it is in, and Argus's are wherever Mix
  # put the dependency.
  #
  # The call graph (Argus's stage 0) is derived here, and the solve told
  # it is provided: left to itself, `run_rules/3` would also learn
  # whether the program reads Argus's process points-to, which it never
  # does, by compiling the program, which takes most of a solve's time.
  defp solve_rules(directory, program) do
    wrapper = directory <> ".dl"

    File.write!(wrapper, """
    .include "#{Application.app_dir(:argus_beam, "priv/dl/clientlib/imports.dl")}"
    .include "#{Path.expand(program)}"
    """)

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
  # comment moves; the program and this analysis's rules (which a probe
  # program includes) without their comments, as Argus keys a program; the
  # Argus whose declarations they build on; and the solver, named by its
  # version as Argus names it in its own keys.
  defp digest(directory, program) do
    directory
    |> File.ls!()
    |> Enum.reject(&(&1 == "line_info.facts"))
    |> Enum.sort()
    |> Enum.reduce(:crypto.hash_init(:sha256), fn file, acc ->
      acc
      |> :crypto.hash_update(file <> "\n")
      |> :crypto.hash_update(File.read!(Path.join(directory, file)))
    end)
    |> :crypto.hash_update(program |> File.read!() |> Argus.Souffle.Program.uncommented())
    |> :crypto.hash_update(rules_file() |> File.read!() |> Argus.Souffle.Program.uncommented())
    |> hash_rules_files()
    |> :crypto.hash_update(to_string(Application.spec(:argus_beam, :vsn)))
    |> :crypto.hash_update(Argus.Souffle.version(Argus.Souffle.executable() || "souffle"))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
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
