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
  """

  @behaviour Argus.Analysis

  alias Argus.Findings
  alias ArgusNxTensorAnalyses.TensorShapes.ShapeFlow

  @external_resource Path.expand("../../priv/tensor_shapes.dl", __DIR__)

  # Each kind of finding: what the title says the call does, why it fails
  # (or why the code should not rely on it), and what to change.
  @kinds %{
    "axis" => %{
      title: "gets an axis its operand does not have",
      why:
        "An axis is an index below the operand's rank, counted from the end when negative, or one of its names, and a list of axes names each once.",
      help: "name an axis the operand has, or check the operand's rank where it is made"
    },
    "broadcast" => %{
      title: "gets shapes that do not broadcast",
      why:
        "Nx broadcasts by lining axes up from the last one: each pair of sizes must be equal, or one of them 1.",
      help:
        "make the sizes that meet equal, or 1 where one side should repeat; Nx.new_axis/3 or Nx.reshape/2 moves an axis to where it belongs"
    },
    "concatenate" => %{
      title: "joins tensors whose other axes differ",
      why: "Tensors joined along an axis must agree on every other axis.",
      help:
        "make the tensors' other axes the same size, or join them along the axis where they differ"
    },
    "conv" => %{
      title: "gets an input and kernel that do not fit",
      why:
        "A convolution needs an input and kernel of one rank, and windows that fit inside the padded input.",
      help:
        "check the input's and kernel's ranks, and the kernel's size against the input's spatial axes"
    },
    "diagonal" => %{
      title: "gets a diagonal of the wrong length",
      why:
        "A diagonal written into a tensor must be exactly as long as the diagonal its offset picks.",
      help: "make the diagonal as long as the one the offset picks"
    },
    "diff" => %{
      title: "takes more differences than the axis holds",
      why:
        "Each order of a difference shortens the axis by one, so the order must be below the axis's size.",
      help: "lower the order, or take the differences along a longer axis"
    },
    "dot" => %{
      title: "contracts axes that do not match",
      why:
        "Nx.dot contracts axes in pairs, one from each side, and each pair must have one size; batch axes pair up the same way.",
      help:
        "contract axes of one size: transpose or reshape an operand, or give the contraction and batch axes explicitly"
    },
    "flatten" => %{
      title: "flattens axes that are not consecutive",
      why: "Flattening merges a run of neighboring axes into one.",
      help: "list axes that sit next to each other, or transpose them together first"
    },
    "gather" => %{
      title: "gets indices that do not fit the tensor",
      why:
        "Each entry along the indices' last axis addresses one axis of the tensor, so there can be no more of them than the tensor has axes.",
      help:
        "make the indices' last axis as long as the axes it addresses, and list those axes in order"
    },
    "indexed" => %{
      title: "gets indices or updates that do not fit the tensor",
      why:
        "An indexed update takes one update per index, and each index addresses the tensor's axes in order.",
      help:
        "make the updates' leading axis match the number of indices, and the indices' last axis match the axes they address"
    },
    "least_squares" => %{
      title: "gets a system and right-hand side that do not fit",
      why: "Least squares needs a right-hand side with as many rows as the system.",
      help: "make the right-hand side's rows match the system's"
    },
    "linspace" => %{
      title: "gets a start and stop that do not fit",
      why:
        "Nx.linspace/3 interpolates element by element, so start and stop need one shape, and n must be given.",
      help: "give start and stop the same shape, and give n"
    },
    "names" => %{
      title: "merges axes of different names",
      why:
        "Where axes meet, each keeps one name: a name merges with nil or with itself, and a tensor's names are its own.",
      help:
        "rename one side with Nx.rename/2 so the names agree, or drop the names that should not meet"
    },
    "pad" => %{
      title: "gets a padding configuration that does not fit",
      why:
        "Padding takes one {low, high, interior} per axis, with interior padding never negative.",
      help: "give one padding entry for each axis of the tensor"
    },
    "put_slice" => %{
      title: "puts a slice that does not fit the tensor",
      why:
        "A slice put into a tensor needs the tensor's rank, no axis longer than the tensor's, and one start index per axis.",
      help: "shrink the slice to fit the tensor, or give one start index per axis"
    },
    "rank" => %{
      title: "gets a tensor of the wrong rank",
      why:
        "The operation works on a certain number of axes: a vector, a matrix, or a batch of them.",
      help:
        "reshape the operand, or add an axis with Nx.new_axis/3, to the rank the operation takes"
    },
    "reshape" => %{
      title: "gets a shape it cannot reshape to",
      why:
        "A reshape keeps every element: the new sizes must multiply to as many elements as the old, with at most one :auto to infer.",
      help:
        "make the new sizes multiply to the old number of elements, or leave one of them :auto"
    },
    "scalar" => %{
      title: "gets a tensor where it takes a scalar",
      why: "This argument must be a single number: a tensor of rank 0 with no vectorized axes.",
      help: "pass a scalar, or reduce the tensor to one first"
    },
    "slice" => %{
      title: "slices outside the tensor",
      why:
        "A slice takes one start index, length and stride per axis, and each length must fit in its axis.",
      help: "give one start, length and stride for each axis, each within the axis"
    },
    "solve" => %{
      title: "gets a right-hand side that does not fit the system",
      why:
        "A system of n equations solves right-hand sides of n rows (n columns when solving from the right), batched as the system is.",
      help: "make the right-hand side's rows match the system's size"
    },
    "split" => %{
      title: "splits an axis at a point outside it",
      why: "A split cuts an axis in two, so the point must fall strictly inside it.",
      help: "split at a point between 0 and the axis's size"
    },
    "square" => %{
      title: "gets a matrix that is not square",
      why: "The operation takes square matrices: the last two axes must have one size.",
      help: "pass a square matrix, or a batch of them"
    },
    "squeeze" => %{
      title: "squeezes an axis whose size is not 1",
      why: "A squeeze only removes axes of size 1.",
      help: "squeeze only axes of size 1, or reshape to drop the axis"
    },
    "stack" => %{
      title: "stacks tensors of different shapes",
      why:
        "Stacking puts tensors side by side along a new axis, so they must all have one shape.",
      help: "give every tensor the same shape, or concatenate them along an axis they share"
    },
    "take_along_axis" => %{
      title: "gets indices that do not fit the tensor",
      why: "The indices must match the tensor on every axis but the one taken along.",
      help: "match the indices' shape to the tensor's on the other axes"
    },
    "top_k" => %{
      title: "takes more elements than the axis holds",
      why: "Nx.top_k/2 takes k elements from the last axis, which must hold at least k.",
      help: "lower k to at most the last axis's size"
    },
    "transpose" => %{
      title: "gets a permutation of the wrong length",
      why: "A transpose's permutation names every axis once.",
      help: "list every axis once in the permutation"
    },
    "vectorize" => %{
      title: "cannot vectorize these axes",
      why:
        "Vectorizing turns leading axes into vectorized ones, which need names of their own and the sizes they are given.",
      help: "vectorize axes the tensor has, under names it does not already use"
    },
    "vectorized_axes" => %{
      title: "gets vectorized axes of different sizes",
      why:
        "Vectorized axes of one name meet as broadcast axes do: their sizes must be equal, or one of them 1.",
      help: "make vectorized axes of one name the same size, or 1 on one side"
    },
    "weighted_mean" => %{
      title: "gets weights that do not fit the input",
      why:
        "A weighted mean needs weights of the input's shape, or axes that say where lower-rank weights go.",
      help: "give the weights the input's shape, or pass :axes for the axes they cover"
    },
    "window" => %{
      title: "gets a window that does not fit the tensor",
      why:
        "A window operation takes one window size and stride per axis, and windows that leave something of each axis.",
      help: "give one window size and stride per axis, each within the padded axis"
    },
    "size_variables" => %{
      title: "lines up sizes the code names differently",
      why:
        "That holds only while the two happen to be equal, or one of them is 1 and broadcasts, and nothing in the code makes them so.",
      help: "derive both sizes from one variable, or check they agree where the variables are set"
    },
    "unnamed_axis" => %{
      title: "lines up an unnamed axis with a named one",
      why:
        "Nx merges an unnamed axis with any name, so it cannot check that these are the axes meant to meet.",
      help: "name the axis (Nx.rename/2, or names: where it is made) so the axes line up by name"
    },
    "contracted_names" => %{
      title: "contracts axes of different names",
      why:
        "Nx contracts axes by position and drops their names, so it cannot tell they were not meant to meet.",
      help: "rename one side so the contracted axes share a name"
    },
    "vectorize_name" => %{
      title: "vectorizes an axis under another name",
      why:
        "The axis keeps its size under the new name, and code that looks for it by its old name does not find it.",
      help: "vectorize the axis under its own name, or rename it first to say it is another"
    }
  }

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
  def finding(:tensor_shape_mismatch, [id, _func, operation, kind, detail, certainty, operands]) do
    %{title: title, why: why, help: help} = kind(kind)

    {severity, raises} =
      case certainty do
        "always" -> {:error, "Nx raises"}
        _some_path -> {:warning, "On some path to this call, Nx raises"}
      end

    Findings.new(severity, "#{operation} #{title}", "#{why} #{raises}: #{detail}.",
      at: Findings.at_instr(id),
      at_label: gets(operands) || "Nx raises here",
      help: [help]
    )
  end

  def finding(:tensor_axis_misalignment, [id, _func, operation, kind, detail, certainty, operands]) do
    %{title: title, why: why, help: help} = kind(kind)

    where =
      case certainty do
        "always" -> ""
        _some_path -> "on some path to this call "
      end

    Findings.new(
      :warning,
      "#{operation} #{title}",
      "Nx accepts this, but #{where}#{detail}. #{why}",
      at: Findings.at_instr(id),
      at_label: gets(operands) || "the axes meet here",
      help: [help]
    )
  end

  @impl true
  def evidence(relation, [_id, _kind, _order, at, _func, how, subject, position, shown])
      when relation in [:tensor_shape_mismatch_via, :tensor_axis_misalignment_via] do
    Findings.related(frame_label(how, subject, position, shown), Findings.at_instr(at))
  end

  # A kind the program has and this module does not describe still reads
  # as a finding.
  defp kind(kind) do
    Map.get(@kinds, kind, %{
      title: "rejects these tensor shapes",
      why: "The operands' shapes do not fit the operation.",
      help: "make the operands' shapes fit the operation"
    })
  end

  # The label under the call: the shapes it gets, or nil where some
  # operand holds several.
  defp gets(""), do: nil

  defp gets(operands) do
    case String.split(operands, ";") do
      [shape] ->
        "gets #{shape}"

      shapes ->
        leading = shapes |> Enum.drop(-1) |> Enum.join(", ")
        "gets #{leading} and #{List.last(shapes)}"
    end
  end

  defp frame_label("call", callee, _position, _shown),
    do: "calls #{function_name(callee)} with these shapes"

  defp frame_label(how, operation, position, shown) do
    verb = if how == "made", do: "makes", else: "returns"
    argument = "the #{ordinal(position)} argument of #{operation}"

    case shown do
      "" -> "#{verb} #{argument}"
      shape -> "#{verb} #{shape}, #{argument}"
    end
  end

  defp ordinal("0"), do: "first"
  defp ordinal("1"), do: "second"
  defp ordinal("2"), do: "third"
  defp ordinal("3"), do: "fourth"

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
  """
  @spec solve([module() | Path.t()], Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def solve(modules, program \\ rules_file(), options \\ []),
    do: with_facts(modules, &solve_facts(&1, program, options))

  @doc """
  Solves the program over the modules (atoms or `.beam` paths) and returns
  each finding placed at its call in the module's source. Takes `solve/3`'s
  options.
  """
  @spec run([module() | Path.t()], keyword()) :: {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(modules, options \\ []) do
    solved =
      with_facts(modules, fn directory ->
        with {:ok, rows} <- solve_facts(directory, rules_file(), options),
             do: {:ok, rows, Argus.Lines.from_facts_dir(directory)}
      end)

    with {:ok, rows, lines} <- solved do
      beams = beams(modules)
      findings = Findings.build(__MODULE__, rows)
      {:ok, Enum.map(findings, &place(&1, beams, lines))}
    end
  end

  # Extracts the modules into a directory of facts, hands it to `solve`,
  # and removes it.
  defp with_facts(modules, solve) do
    directory =
      Path.join(System.tmp_dir!(), "tensor_shapes_#{System.unique_integer([:positive])}")

    try do
      with {:ok, directory} <- extract(modules, directory), do: solve.(directory)
    after
      File.rm_rf!(directory)
    end
  end

  defp solve_facts(directory, program, options) do
    case Keyword.get(options, :cache) do
      nil -> solve_rules(directory, program)
      cache -> solve_cached(directory, program, cache)
    end
  end

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
