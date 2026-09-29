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
  alias ArgusNxTensorAnalyses.TensorShapes.ShapeFlow
  alias ArgusNxTensorAnalyses.TensorShapes.Wording

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

  def finding(:tensor_nonfinite_result, [id, _func, operation, kind, cause, origin, shown]) do
    %{title: title, detail: detail, label: label, help: help, frame: frame} =
      wording = hazard(kind, cause, operation)

    Findings.new(Map.get(wording, :severity, severity(kind)), "#{operation} #{title}", detail,
      at: Findings.at_instr(id),
      at_label: label,
      help: [help],
      related: origin_frame(frame, origin, shown)
    )
  end

  def finding(:tensor_type_error, [
        id,
        _func,
        operation,
        "non_integer_operand",
        class,
        position,
        certain,
        origin,
        shown
      ]) do
    {rejects, help} = integer_only(operation)

    Findings.new(
      if(to_integer(certain) == 1, do: :error, else: :warning),
      "#{operation} takes integers, and its #{ordinal(to_string(position))} argument can be a #{class}",
      "#{rejects} Nx raises for a #{class} tensor there.",
      at: Findings.at_instr(id),
      at_label: "gets a #{class} here",
      help: [help],
      related: origin_frame("makes it a #{class}:", origin, shown)
    )
  end

  def finding(:tensor_type_error, [id, _func, operation, "unsupported_type", name | _rest]) do
    Findings.new(
      :error,
      "#{operation} makes #{article(name)} #{name} tensor, which the backend does not support",
      "The analysis is run with #{name} among the types the backend lacks (the " <>
        ":unsupported_types option). A backend without #{name} raises making it, or at the " <>
        "first operation over it.",
      at: Findings.at_instr(id),
      at_label: "makes #{name} here",
      help: [
        "make it a type the backend supports, such as f32, or run this on a backend that has #{name}"
      ]
    )
  end

  def finding(:tensor_type_error, [id, _func, operation, kind, subject, _position, certain | rest]) do
    [origin, shown] = rest
    wording = Wording.type_error(kind, subject, operation) || unknown_wording(kind, subject)

    Findings.new(
      Map.get(wording, :severity, if(to_integer(certain) == 1, do: :error, else: :warning)),
      "#{operation} #{wording.title}",
      wording.detail,
      at: Findings.at_instr(id),
      at_label: wording.label,
      help: [wording.help],
      related: origin_frame(wording.frame, origin, shown)
    )
  end

  def finding(:tensor_call_error, [id, _func, operation, kind, detail, origin, shown]) do
    wording = Wording.call_error(kind, detail, operation) || unknown_wording(kind, detail)

    Findings.new(
      Map.get(wording, :severity, :error),
      String.trim("#{operation} #{wording.title}"),
      wording.detail,
      at: Findings.at_instr(id),
      at_label: wording.label,
      help: [wording.help],
      related: origin_frame(wording.frame, origin, shown)
    )
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

  # What a function that takes only integers raises for anything else, and
  # what to change.
  defp integer_only(operation) do
    name = String.replace(operation, ~r{/\d+$}, "")

    cond do
      name in ~w(Nx.take Nx.take_along_axis Nx.gather Nx.indexed_add Nx.indexed_put) ->
        {"Its indices must be an integer tensor.",
         "compute the indices as integers: round and then Nx.as_type(x, :s32), or use integer operations (Nx.quotient rather than Nx.divide)"}

      name in ~w(Nx.quotient Nx.Defn.Kernel.div) ->
        {"An integer quotient takes integer tensors only.",
         "make the operands integers, such as Nx.as_type(x, :s32), or divide and round (Nx.floor(Nx.divide(x, y))) to keep floats"}

      true ->
        {"Bitwise operations take integer tensors only.",
         "make the operand an integer tensor, such as Nx.as_type(x, :s32)"}
    end
  end

  # The article before a type's name as it is read: an f64, a u8.
  defp article(name), do: if(String.starts_with?(name, ["f", "s"]), do: "an", else: "a")

  defp to_integer(value) when is_integer(value), do: value
  defp to_integer(value) when is_binary(value), do: String.to_integer(value)

  # An operand nothing checks may still never reach where the call is not
  # defined, so those kinds are noisier than the rest and report as info.
  @unchecked ["unchecked_divisor", "unchecked_logarithm", "unchecked_root", "unchecked_domain"]

  defp severity(kind) when kind in @unchecked, do: :info
  defp severity(_kind), do: :warning

  @impl true
  def evidence(relation, [_id, _kind, _order, at, _func, how, subject, position, shown])
      when relation in [:tensor_shape_mismatch_via, :tensor_axis_misalignment_via] do
    Findings.related(frame_label(how, subject, position, shown), Findings.at_instr(at))
  end

  # How a divisor can be zero, by the cause the program names.
  @causes %{
    "square" => "it is made of a square, which is zero where its operand is",
    "absolute" => "it is made of an absolute value, which is zero where its operand is",
    "root" => "it is made of a square root, which is zero where its operand is",
    "norm" => "it is a norm, which is zero for a zero vector",
    "comparison" =>
      "it is made of a comparison, which is 0 where it does not hold, as nowhere on a row with nothing selected",
    "index" => "it is made of an index or an iota, which starts at zero",
    "identity" => "it is made of an identity matrix, which is zero off its diagonal",
    "spread" =>
      "it is a variance or standard deviation, which is zero where every value is the same",
    "clamp" => "it is clamped at zero, so it is zero wherever the value clamped is not above it",
    "remainder" => "it is a remainder, which is zero where the division comes out whole",
    "quotient" =>
      "it is an integer quotient, which is zero where the dividend is smaller than the divisor",
    "round" => "it is rounded, which takes a value between -1 and 1 to zero",
    "zero" => "it is a written zero",
    "input" => "it comes from an input, or from a value the analysis cannot follow",
    "cancel" => "it is a sum or difference whose terms can cancel",
    "negative" => "nothing in how it is computed keeps it from going below zero"
  }

  # A cause the program has and this module does not describe still reads.
  defp cause(cause), do: Map.get(@causes, cause, "its math lets it be zero")

  # Where each function defined on only part of the line is defined, by
  # the function's name.
  @domains %{
    "Nx.asin" => "is defined on [-1, 1]",
    "Nx.acos" => "is defined on [-1, 1]",
    "Nx.atanh" => "is defined between -1 and 1, and infinite at ±1",
    "Nx.erf_inv" => "is defined between -1 and 1, and infinite at ±1",
    "Nx.log1p" => "is defined above -1, and negative infinity at -1",
    "Nx.acosh" => "is defined from 1 up"
  }

  # How an operand comes to the edge of a domain or past it, by the cause
  # the program names.
  @domain_causes %{
    "rounding" =>
      "it is within ±1 only before rounding, as a cosine similarity or a vector over its norm is, and rounding can take it just past",
    "saturation" =>
      "it is made of a tanh, erf or sigmoid, which rounds to exactly ±1 for large inputs",
    "trigonometric" => "it is made of a sine or cosine, which reaches ±1",
    "clip" => "it is clipped to a bound at the edge",
    "written" => "a written number puts it there",
    "size" => "it is a size, which is at least 1",
    "index" => "it is made of an index or an iota, which counts up from zero",
    "comparison" => "it is made of a comparison, which is 0 or 1",
    "sign" => "it is a sign, which is -1, 0 or 1",
    "identity" => "it is made of an identity matrix, which is 0 or 1",
    "input" => "it comes from an input, or from a value the analysis cannot follow",
    "unbounded" => "its math does not keep it within the domain"
  }

  defp domain(operation) do
    name = String.replace(operation, ~r{/\d+$}, "")
    "#{name} #{Map.get(@domains, name, "is defined on only part of the line")}"
  end

  defp domain_cause(cause), do: Map.get(@domain_causes, cause, "its math takes it there")

  # What each kind of result that can be infinite or NaN says: its title,
  # why, the label at the call, what to change, and the frame at its origin.
  defp hazard("outside_domain", cause, operation) do
    %{
      title: "can take a value outside its domain",
      detail:
        "#{domain(operation)}, and its operand can go outside it: #{domain_cause(cause)}. " <>
          "The result there is NaN.",
      label: "takes it here",
      help:
        "clip the operand into the domain first, such as Nx.clip(x, -1.0, 1.0) before asin or acos",
      frame: "the operand leaves the domain because of this"
    }
  end

  defp hazard("infinite_at_edge", cause, operation) do
    %{
      title: "can reach the edge of its domain",
      detail:
        "#{domain(operation)}, and its operand can reach the edge: #{domain_cause(cause)}. " <>
          "The result there is infinite.",
      label: "takes it here",
      help:
        "keep the operand strictly inside: clip it a little short of the edge, such as Nx.clip(x, -1 + eps, 1 - eps)",
      frame: "the operand reaches the edge because of this"
    }
  end

  defp hazard("unchecked_domain", cause, operation) do
    %{
      title: "takes a value nothing keeps in its domain",
      detail:
        "#{domain(operation)}. Its operand can be outside it as far as the code shows: " <>
          "#{domain_cause(cause)}, and no clip and no test on the way to the call keeps it inside.",
      label: "takes it here",
      help: "clip the operand into the domain, or check it first",
      frame: "the operand can leave the domain because of this"
    }
  end

  defp hazard(kind, cause, operation),
    do: Wording.hazard(kind, cause, operation) || hazard(kind, cause)

  defp hazard("divide_by_zero", cause) do
    %{
      title: "can divide by zero",
      detail:
        "The divisor cannot be negative, but it can be zero: #{cause(cause)}. " <>
          "Nx gives an infinity or a NaN there, and an integer quotient or remainder raises.",
      label: "divides here",
      help:
        "keep the divisor away from zero: add a positive epsilon to it, or take Nx.max of it and one",
      frame: "the divisor can be zero because of this"
    }
  end

  defp hazard("infinite_gradient", cause) do
    %{
      title: "has an infinite gradient where its result is zero",
      detail:
        "A grad differentiates it, and its result cannot be negative but can be zero: " <>
          "#{cause(cause)}. The derivative of a square root or root there is infinite, and " <>
          "of a norm NaN (0/0), and the gradient carries it back into everything before it.",
      label: "differentiated here",
      help:
        "keep the operand away from zero where it is differentiated, such as Nx.sqrt(Nx.add(x, 1.0e-12))",
      frame: "the result can be zero because of this"
    }
  end

  defp hazard("log_of_zero", "underflow") do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand is a softmax written out: an exponential far below the largest underflows " <>
          "to zero, and the logarithm of zero is negative infinity.",
      label: "takes the logarithm here",
      help:
        "take the logarithm of a softmax directly: Nx.subtract(x, Nx.logsumexp(x, axes: [...], keep_axes: true))",
      frame: "the softmax is taken here"
    }
  end

  defp hazard("log_of_zero", cause) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand cannot be negative, but it can be zero: #{cause(cause)}. " <>
          "The logarithm of zero is negative infinity.",
      label: "takes the logarithm here",
      help:
        "keep the operand away from zero: add a positive epsilon to it, or take Nx.max of it and one",
      frame: "the operand can be zero because of this"
    }
  end

  defp hazard("root_of_negative", _cause) do
    %{
      title: "can take the square root of a negative value",
      detail:
        "Its operand is a difference of two values that cannot be negative, such as a variance " <>
          "written as E[x²] - E[x]². Where they are close, rounding takes the difference below " <>
          "zero, and the square root there is NaN.",
      label: "takes the root here",
      help:
        "clamp the difference at zero first (Nx.max of it and 0), or use a form that cannot go negative, such as Nx.variance/2",
      frame: "the difference is taken here"
    }
  end

  defp hazard("log_of_negative", _cause) do
    %{
      title: "can take the logarithm of a negative value",
      detail:
        "Its operand is a difference of two values that cannot be negative. Where they are " <>
          "close, rounding takes the difference below zero, and the logarithm there is NaN.",
      label: "takes the logarithm here",
      help: "keep the difference above zero first (Nx.max of it and a positive epsilon)",
      frame: "the difference is taken here"
    }
  end

  defp hazard("exp_overflow", _cause) do
    %{
      title: "can overflow its exponentials",
      detail:
        "It normalizes exponentials of values not shifted down by their maximum first: a large " <>
          "value overflows the exponential to infinity, and infinity over infinity, or its " <>
          "logarithm, is not finite.",
      label: "normalizes here",
      help:
        "subtract the maximum first (Nx.subtract(x, Nx.reduce_max(x, axes: [...], keep_axes: true))), which leaves the result as it is, or use Nx.logsumexp/2",
      frame: "the exponential that can overflow"
    }
  end

  defp hazard("unchecked_divisor", cause) do
    %{
      title: "divides by a value nothing checks is nonzero",
      detail:
        "The divisor can be zero as far as the code shows: #{cause(cause)}, and no test on " <>
          "the way to the call and no select keeps it from zero. Nx gives an infinity or a NaN " <>
          "there, and an integer quotient or remainder raises.",
      label: "divides here",
      help:
        "check the divisor first (a guard, or Nx.select on Nx.equal(divisor, 0)), or keep it from zero with a positive epsilon",
      frame: "the divisor can be zero because of this"
    }
  end

  defp hazard("unchecked_logarithm", cause) do
    %{
      title: "takes the logarithm of a value nothing checks is positive",
      detail:
        "The operand can be zero or negative as far as the code shows: #{cause(cause)}, and " <>
          "no test on the way to the call and no select keeps it positive. The logarithm of " <>
          "zero is negative infinity, and of a negative value NaN.",
      label: "takes the logarithm here",
      help:
        "check the operand first (a guard, or Nx.select on Nx.greater(operand, 0)), or take Nx.max of it and a positive epsilon",
      frame: "the operand can be zero because of this"
    }
  end

  defp hazard("unchecked_root", _cause) do
    %{
      title: "takes the square root of a value nothing checks is not negative",
      detail:
        "Nothing in how the operand is computed keeps it from going below zero, and no test " <>
          "on the way to the call and no select checks it. The square root of a negative " <>
          "value is NaN.",
      label: "takes the root here",
      help: "check the operand first, or clamp it at zero (Nx.max of it and 0)",
      frame: "the operand can be negative because of this"
    }
  end

  # A kind the program has and this module does not describe still reads.
  defp hazard(_kind, cause) do
    %{
      title: "can give an infinity or a NaN",
      detail: "Its operand can reach a value the call is not defined at: #{cause(cause)}.",
      label: "here",
      help: "keep the operand where the call is defined",
      frame: "because of this"
    }
  end

  # The frame at the call a finding comes from, none for a written value.
  defp origin_frame(_frame, "", _shown), do: []

  defp origin_frame(frame, origin, shown),
    do: [Findings.related("#{frame} #{shown}", Findings.at_instr(origin))]

  # A kind the program has and this module does not describe still reads
  # as a finding.
  defp kind(kind) do
    Map.get(@kinds, kind) || Wording.violation(kind) ||
      %{
        title: "rejects these tensor shapes",
        why: "The operands' shapes do not fit the operation.",
        help: "make the operands' shapes fit the operation"
      }
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
  defp ordinal(position), do: "#{String.to_integer(position) + 1}th"

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
