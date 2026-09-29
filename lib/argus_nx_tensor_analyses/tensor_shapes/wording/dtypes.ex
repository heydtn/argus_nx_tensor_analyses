defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Dtypes do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/dtypes.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  # The largest whole number up to which each type holds every whole
  # number.
  @limits %{
    "u2" => 3,
    "u4" => 15,
    "u8" => 255,
    "u16" => 65535,
    "s2" => 1,
    "s4" => 7,
    "s8" => 127,
    "s16" => 32767,
    "f8" => 8,
    "f8_e4m3fn" => 16,
    "bf16" => 256,
    "f16" => 2048,
    "f32" => 16_777_216,
    "c64" => 16_777_216
  }

  # Each low-range float type's largest finite value, as written.
  @largest %{"f16" => "65504", "f8" => "57344"}

  @impl true
  def call_error("unsigned_wraparound", type, operation) do
    %{
      title: "can go below zero in #{type}, which wraps around",
      detail: wraparound_detail(name(operation), type),
      label: "wraps around here",
      help: wraparound_help(name(operation)),
      frame: "makes it #{type}:",
      severity: :warning
    }
  end

  def call_error("count_wraparound", detail, _operation) do
    [type, length] = String.split(detail, " ", parts: 2)

    %{
      title: "counts past what #{type} holds",
      detail:
        "It adds up zeros and ones (a mask) in #{type}, which Nx keeps for this operation " <>
          "rather than widening it as Nx.sum/2 does, over #{elements(length)}: past " <>
          "#{limit(type)} the count wraps around (300 ones count to 44 in u8).",
      label: "counts here",
      help:
        "count in a wider type: make the mask s32 first, as in Nx.cumulative_sum(Nx.as_type(mask, :s32))",
      frame: "makes the mask:",
      severity: written_severity(length)
    }
  end

  def call_error("narrow_wraparound", type, operation) do
    %{
      title: "adds up or multiplies #{type} values in #{type}",
      detail:
        "#{name(operation)} keeps #{type} for its sums and products, where Nx.sum/2 would " <>
          "widen it: past #{limit(type)} they wrap around (the dot product of s8 [100, 100] " <>
          "and [2, 2] is -112).",
      label: "wraps around here",
      help: "widen the operands first, as in Nx.as_type(t, :s32), or make them f32",
      frame: "makes it #{type}:",
      severity: :warning
    }
  end

  def call_error("index_wraparound", detail, _operation) do
    [type, length] = String.split(detail, " ", parts: 2)

    %{
      title: "returns indices in a type too small for them",
      detail:
        "It is given type #{type}, whose largest value is #{limit(type)}, and searches " <>
          "#{elements(length)}: an index past #{limit(type)} wraps around (the argmax of 300 " <>
          "values in u8 is 43, not 299).",
      label: "returns #{type} indices here",
      help: "give a type that holds every index, such as the default :s32",
      frame: "",
      severity: written_severity(length)
    }
  end

  def call_error("sequence_precision", detail, _operation) do
    [type, length] = String.split(detail, " ", parts: 2)

    %{
      title: "counts past what #{type} holds exactly",
      detail:
        "A run of #{run(length)} whole numbers (an iota, a linspace's points) is made in, " <>
          "or brought into, #{type}, #{sequence_effect(type)}",
      label: "in #{type} here",
      help:
        "count in s32, or f32, and keep positions in it until they meet the values they scale",
      frame: "the count is made here:",
      severity: written_severity(length)
    }
  end

  def call_error("cast_wraparound", type, _operation) do
    %{
      title: "makes a negative or out-of-range number #{type}",
      detail:
        "A value that can be negative, or a written number outside #{type}'s range, is " <>
          "made #{type}: it wraps around rather than saturating (-1 becomes 255 in u8, and " <>
          "300 becomes 44).",
      label: "made #{type} here",
      help: "clip into the type's range first, as in Nx.clip(x, 0, 255), or use a signed type",
      frame: "",
      severity: :warning
    }
  end

  def call_error("unchecked_cast_wraparound", type, _operation) do
    %{
      title: "makes a value nothing keeps non-negative #{type}",
      detail:
        "The value can be negative only where an input is, which it may never be; where it " <>
          "is, making it #{type} wraps it around rather than saturating (-1 becomes 255 in u8).",
      label: "made #{type} here",
      help: "clip into the type's range first, as in Nx.clip(x, 0, 255), or check the input",
      frame: "",
      severity: :info
    }
  end

  def call_error("complex_to_real", detail, _operation) do
    [from, to] = String.split(detail, " ")

    %{
      title: "drops the imaginary part of a complex tensor",
      detail:
        "It makes a #{from} tensor #{to}: Nx keeps the real part and drops the imaginary one " <>
          "without a word (1+2i becomes 1.0).",
      label: "made #{to} here",
      help: "take Nx.real/1, Nx.abs/1 or Nx.phase/1, whichever is meant",
      frame: "makes it #{from}:",
      severity: :warning
    }
  end

  def call_error("float_truncation", detail, _operation) do
    [from, to] = String.split(detail, " ")

    %{
      title: "cuts the fraction off a float",
      detail:
        "It makes #{article(from)} #{from} tensor #{to}, which truncates toward zero (2.7 " <>
          "becomes 2, and -2.7 becomes -2).",
      label: "made #{to} here",
      help: "round first, with Nx.round/1 or Nx.floor/1, where rounding is meant",
      frame: "makes it #{from}:",
      severity: :info
    }
  end

  def call_error("literal_underflow", detail, _operation) do
    [number, type] = String.split(detail, " ")

    %{
      title: "adds a number #{type} rounds to zero",
      detail:
        "The number #{number} is below the smallest magnitude #{type} holds, and the tensor " <>
          "it meets is #{type}, which keeps its type: the number is 0 there, and an epsilon " <>
          "meant to keep a value from zero does not.",
      label: "#{number} is 0 in #{type} here",
      help: "use an epsilon the type holds (1.0e-4 or larger for f16), or compute in f32",
      frame: "",
      severity: :warning
    }
  end

  def call_error("pad_type_mismatch", detail, _operation) do
    [from, to] = String.split(detail, " ")

    %{
      title: "pads with a value of another type",
      detail:
        "It pads #{article(from)} #{from} tensor with a value whose type makes the result " <>
          "#{to}. Nx's binary backend splices the value's raw bits into the #{from} tensor " <>
          "(0.5 padded into s32 is 1056964608), and EMLX raises; only EXLA converts it.",
      label: "pads here",
      help:
        "give the pad value the tensor's type, as in Nx.pad(t, Nx.tensor(0, type: Nx.type(t)), config)",
      frame: "",
      severity: :warning
    }
  end

  def call_error("integer_determinant", type, _operation) do
    %{
      title: "takes the determinant of #{article(type)} #{type} matrix in #{type}",
      detail:
        "For 2×2 and 3×3 matrices Nx multiplies the entries out in the matrix's type: an " <>
          "unsigned type wraps where the determinant is negative (the determinant of u8 " <>
          "[[1, 2], [3, 4]] comes out 4.29e9), and a signed one overflows for large entries.",
      label: "multiplies in #{type} here",
      help: "make the matrix a float first, as in Nx.as_type(m, :f32)",
      frame: "makes it #{type}:",
      severity: if(String.starts_with?(type, "u"), do: :warning, else: :info)
    }
  end

  def call_error("f32_precision", type, _operation) do
    %{
      title: "computes #{article(type)} #{type} result with an f32 constant",
      detail:
        "It divides by the logarithm of its base computed as an f32 tensor, so its #{type} " <>
          "result is accurate only to about seven digits.",
      label: "divides by an f32 here",
      help:
        "divide Nx.log/1 of the tensor by the base's logarithm in #{type}, as in Nx.divide(Nx.log(x), Nx.log(Nx.tensor(2, type: :#{type})))",
      frame: "",
      severity: :info
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  @impl true
  def hazard("literal_overflow", type, _operation) do
    %{
      title: "makes a number infinite in #{type}",
      detail:
        "A number the code writes here is past #{type}'s largest finite value" <>
          "#{largest(type)}, and the tensor's type is #{type}, which it keeps: the number " <>
          "becomes infinite there, so a -1.0e9 mask gives -Inf, and a softmax over it NaN.",
      label: "infinite in #{type} here",
      help:
        "use a number the type holds, such as Nx.Constants.min_finite(Nx.type(x)), or compute in f32",
      frame: "",
      severity: :warning
    }
  end

  def hazard("cast_overflow", type, _operation) do
    %{
      title: "makes a value infinite in #{type}",
      detail:
        "The value made #{type} holds a number past #{type}'s largest finite value" <>
          "#{largest(type)}: it becomes infinite (-1.0e9, or Nx.Constants.min_finite(:f32), " <>
          "is -Inf in f16), and a fully masked softmax row over it NaN.",
      label: "infinite in #{type} here",
      help:
        "make the value in the target type, as with Nx.Constants.min_finite(:#{type}), or clip it into range first",
      frame: "the value is made here:",
      severity: :warning
    }
  end

  def hazard("float_sum_overflow", cause, _operation) do
    [type, length] = String.split(cause, " ", parts: 2)

    %{
      title: "sums more values than #{type} holds the total of",
      detail:
        "It sums #{elements(length)} in #{type}, which Nx keeps for the sums, means and " <>
          "variances of floats: a sum of values near 1 passes #{type}'s largest finite value" <>
          "#{largest(type)} and is infinite.",
      label: "sums in #{type} here",
      help: "sum in f32, as in Nx.sum(Nx.as_type(t, :f32)), and convert the result back",
      frame: "",
      severity: written_severity(length)
    }
  end

  def hazard("unsigned_logsumexp", type, _operation) do
    %{
      title: "subtracts the maximum of an unsigned tensor",
      detail:
        "Nx.logsumexp/2 subtracts the maximum in the tensor's type, #{type}: every smaller " <>
          "element wraps around to a large positive number, and its exponential is infinite.",
      label: "wraps around here",
      help: "make the tensor a float first, as in Nx.logsumexp(Nx.as_type(t, :f32))",
      frame: "makes it #{type}:",
      severity: :warning
    }
  end

  def hazard(_kind, _cause, _operation), do: nil

  @impl true
  def type_error("upcast", subject, operation) do
    [from, to] = String.split(subject, " ")

    %{
      title: "turns #{article(from)} #{from} operand into #{to}",
      detail: upcast_detail(name(operation), from, to),
      label: "#{from} becomes #{to} here",
      help: upcast_help(name(operation)),
      frame: "the #{to} operand:",
      severity: :warning
    }
  end

  def type_error("narrowing_merge", subject, _operation) do
    [from, to] = String.split(subject, " ")

    %{
      title: "merges #{article(from)} #{from} operand into #{to}, which holds less",
      detail:
        "Nx.Type.merge/2 takes #{to} for these operands, which has a smaller range or fewer " <>
          "significand bits than #{from}: a bf16 past 65504 becomes infinite in f16, and a u64 " <>
          "past 2^63 turns negative in s64.",
      label: "#{from} becomes #{to} here",
      help:
        "convert both operands to a type that holds either (f32, or s64 for integers) before they meet",
      frame: "makes it #{from}:",
      severity: :warning
    }
  end

  def type_error(_kind, _subject, _operation), do: nil

  defp wraparound_detail("Nx.diff", type),
    do:
      "It subtracts neighbors in #{type}, an unsigned type: wherever the tensor falls along " <>
        "the axis, the difference wraps around to a large positive number (a mask's 1 then 0 " <>
        "differ by 255 in u8)."

  defp wraparound_detail("Nx.all_close", type),
    do:
      "It subtracts its operands in #{type}, an unsigned type: where the second is the " <>
        "larger, the difference wraps around, and the tensors compare as far apart."

  defp wraparound_detail("Nx.linspace", type),
    do:
      "It computes its step as the stop minus the start in #{type}, an unsigned type, and " <>
        "the stop is below the start: the step wraps around."

  defp wraparound_detail(_operation, type),
    do:
      "Its result is #{type}, an unsigned type, and its operands let it go below zero: Nx " <>
        "does not make it signed, so a negative result wraps around to a large positive " <>
        "number (0 - 1 is 255 in u8, and 4294967295 in u32)."

  defp wraparound_help("Nx.diff"),
    do: "make the tensor signed first, as in Nx.diff(Nx.as_type(t, :s32))"

  defp wraparound_help("Nx.all_close"),
    do: "compare signed or float tensors, as in Nx.all_close(Nx.as_type(a, :s32), b)"

  defp wraparound_help("Nx.linspace"),
    do: "use a signed or float type, or go from the smaller end and reverse"

  defp wraparound_help(_operation),
    do:
      "make the operand signed first, as in Nx.as_type(mask, :s32), or add a negative number: Nx.add(x, -1) is s16 for a u8 x"

  defp upcast_detail(operation, from, _to) when operation in ["Nx.log2", "Nx.log10", "Nx.log"],
    do:
      "It divides by the logarithm of its base computed as an f32 tensor, so #{article(from)} " <>
        "#{from} operand comes out f32, not the type the code runs at."

  defp upcast_detail("Nx.LinAlg.invert", from, _to),
    do:
      "Its result merges an f32 tensor of its own, so #{article(from)} #{from} matrix comes " <>
        "out f32, not the type the code runs at."

  defp upcast_detail("Nx.clip", from, to),
    do:
      "Nx makes the bounds tensors (a float bound f32, a whole one s32) and merges their type " <>
        "with the operand's, so #{article(from)} #{from} tensor comes out #{to}."

  defp upcast_detail(_operation, from, to),
    do:
      "It meets a tensor fixed at #{to} (a constant, a number Nx makes a tensor, or a type the " <>
        "code gives), and the merged type is #{to}: the #{from} operand and the result are " <>
        "#{to}, not the type the code runs at."

  defp upcast_help(operation) when operation in ["Nx.log2", "Nx.log10", "Nx.log"],
    do:
      "divide Nx.log/1 of the tensor by a logarithm of its own type, as in Nx.divide(Nx.log(x), Nx.log(Nx.tensor(2, type: Nx.type(x))))"

  defp upcast_help("Nx.LinAlg.invert"),
    do: "convert the result back, as in Nx.as_type(Nx.LinAlg.invert(m), Nx.type(m))"

  defp upcast_help("Nx.clip"),
    do:
      "pass bounds of the tensor's type, as in Nx.clip(x, Nx.tensor(-1, type: Nx.type(x)), Nx.tensor(1, type: Nx.type(x)))"

  defp upcast_help(_operation),
    do:
      "give the other operand the tensor's type (Nx.as_type(c, Nx.type(x))), or pass a plain number, which keeps it"

  # What a run of whole numbers past a type's limit becomes in it.
  defp sequence_effect(type) do
    if String.starts_with?(type, ["u", "s"]),
      do:
        "whose largest value is #{limit(type)}: past it they wrap around (a u8 iota of 300 " <>
          "goes back to 0 at 256).",
      else:
        "which holds every whole number only up to #{limit(type)}: past it neighbors round " <>
          "to one value (a bf16 iota gives 256, 256, 258, 258, ...), and positions or " <>
          "timesteps built on them repeat."
  end

  # A count as a sentence writes it: written, read from the code, or not
  # shown.
  defp elements("?"), do: "an axis whose length the code does not show"

  defp elements(length) do
    case Integer.parse(length) do
      {_count, ""} -> "#{length} elements"
      _symbolic -> "#{length} elements, a number the code reads"
    end
  end

  # How long a run of whole numbers is, as a sentence writes it.
  defp run(length) do
    case Integer.parse(length) do
      {_count, ""} -> length
      _symbolic -> "#{length} (a number the code reads)"
    end
  end

  # Definite where the count is written, and only possible where the code
  # reads it or does not show it.
  defp written_severity(length) do
    case Integer.parse(length) do
      {_count, ""} -> :warning
      _symbolic -> :info
    end
  end

  defp limit(type), do: Map.get(@limits, type, "its largest value")

  defp largest(type) do
    case Map.fetch(@largest, type) do
      {:ok, value} -> " (#{value})"
      :error -> ""
    end
  end

  defp name(operation), do: String.replace(operation, ~r{/\d+$}, "")

  # The article before a type's name as it is read: an f16, a u8.
  defp article(name), do: if(String.starts_with?(name, ["f", "s"]), do: "an", else: "a")
end
