defmodule ArgusNxTensorAnalyses.TensorShapes do
  @moduledoc """
  Shapes, values, types and uses of Nx that are wrong before the code
  runs, found in compiled code by the Datalog program in
  `priv/tensor_shapes.dl` and the files it includes: Nx calls whose
  shapes, types or options Nx rejects, calls Nx accepts where the code
  does not line its axes up, math that can give an infinity or a NaN, and
  misuse of traced code, gradients, random keys, containers and servings.
  `docs/checks.md` lists what it reports.

  The program reads what `ArgusNxTensorAnalyses.TensorShapes.ShapeFlow`
  extracts besides Argus's own facts, and Argus's analyses run only the
  extractors it ships with, so this module extracts and solves itself
  (`solve/3`), turns the program's rows into findings
  (`c:Argus.Analysis.finding/2`), and places them at their calls' lines
  (`run/2`) for `Argus.Report` to render.

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
  alias ArgusNxTensorAnalyses.Solve
  alias ArgusNxTensorAnalyses.TensorShapes.ShapeFlow
  alias ArgusNxTensorAnalyses.TensorShapes.Wording
  alias ArgusNxTensorAnalyses.TensorShapes.Wording.Causes

  import ArgusNxTensorAnalyses.Text

  @external_resource Path.expand("../../priv/tensor_shapes.dl", __DIR__)

  @impl true
  def name, do: :tensor_shapes

  @impl true
  def description,
    do: "Shapes, values, types and uses of Nx that are wrong before the code runs"

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
  Extracts the modules (atoms or `.beam` paths) with this analysis's
  extractors and solves `program` over the facts, returning its output
  relations' rows by name. `program` defaults to this analysis's; a
  program that includes it can read its relations too.

  ## Options

    * `:cache` — a directory to keep the rows in, under a digest of what
      the solve reads (the facts extracted from the modules, the program
      and every file it includes without their comments, Argus's version,
      the solver's): a solve that would read the same reads them back
      instead, so an edit that leaves the compiled code as it was (a
      comment, a doc, a comment in the rules) solves nothing. Only the
      latest rows are kept. Default: nil, no cache.
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
    do: Solve.solve(__MODULE__, modules, program, options)

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
  def run(modules, options \\ []),
    do: Solve.run(Keyword.get(options, :analysis, __MODULE__), modules, options)
end
