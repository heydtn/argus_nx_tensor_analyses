defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Dtypes do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/dtypes.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

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

  # The types that hold more than f32, whose tensors a float number meets
  # as the f32 Nx makes it outside traced code.
  @wider_than_f32 ~w(f64 c128)

  @impl true
  def call_error("unsigned_wraparound", type, operation) do
    %{
      title: "can go below zero in #{type}, which wraps around",
      detail: wraparound_detail(without_arity(operation), type),
      label: wraparound_label(operation, type),
      help: wraparound_help(without_arity(operation)),
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
      label: "#{counts(length)} in #{type}, which wraps past #{limit(type)}",
      help:
        "count in a wider type: make the mask s32 first, as in Nx.cumulative_sum(Nx.as_type(mask, :s32))",
      frame: "makes the mask:",
      severity: written_severity(length)
    }
  end

  def call_error("narrow_wraparound", type, operation) do
    name = without_arity(operation)

    %{
      title: "adds up or multiplies #{type} values in #{type}",
      detail:
        "#{name} keeps #{type} for its sums and products, where Nx.sum/2 would widen it: past " <>
          "#{limit(type)} they wrap around (#{narrow_example(name, type)}).",
      label: "#{narrow_verb(name)} in #{type}, where #{one_past(type)}",
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
          "#{elements(length)}: an index past #{limit(type)} wraps around " <>
          "(#{last_index(length, type)}).",
      label: "returns #{type} indices, where #{last_index(length, type)}",
      help: "give a type that holds every index, such as the default :s32",
      frame: "",
      severity: written_severity(length)
    }
  end

  def call_error("sequence_precision", detail, _operation) do
    [types, length] = String.split(detail, " ", parts: 2)
    names = String.split(types, "/")
    shown = join(names, "or")

    %{
      title: "counts past what #{shown} holds exactly",
      detail:
        "A run of #{run(length)} whole numbers (an iota, a linspace's points) is made in, " <>
          "or brought into, #{shown}, #{sequence_effect(names)}",
      label: "#{length} positions in #{shown}, #{exact_to(names)}",
      help:
        "count in s32, or f32, and keep positions in it until they meet the values they scale",
      frame: "makes the count:",
      severity: written_severity(length)
    }
  end

  # A written number outside the integer type it is made (`u8 -1`), or a
  # value the code's math can make negative made unsigned (`u8`).
  def call_error("cast_wraparound", detail, _operation) do
    case String.split(detail, " ") do
      [type, number] -> number_wraparound(type, String.to_integer(number))
      [type] -> negative_wraparound(type)
    end
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
    [from, types] = String.split(detail, " ")
    to = types |> String.split("/") |> join("or")

    %{
      title: "drops the imaginary part of a complex tensor",
      detail:
        "It makes a #{from} tensor #{to}: Nx keeps the real part and drops the imaginary one " <>
          "without a word (1+2i becomes 1.0).",
      label: "#{from} becomes #{to}, without its imaginary part",
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
      label: "#{from} becomes #{to}, truncated toward zero",
      help: "round first, with Nx.round/1 or Nx.floor/1, where rounding is meant",
      frame: "makes it #{from}:",
      severity: :info
    }
  end

  def call_error("literal_underflow", detail, _operation) do
    [number, types] = String.split(detail, " ")
    names = String.split(types, "/")
    type = join(names, "or")

    if Enum.all?(names, &(&1 in @wider_than_f32)),
      do: underflow_before_meeting(number, names, type),
      else: underflow_in_type(number, type)
  end

  def call_error("pad_type_mismatch", detail, _operation) do
    [from, to] = String.split(detail, " ")

    %{
      title: "pads with a value of another type",
      detail:
        "It pads #{article(from)} #{from} tensor with a value whose type makes the result " <>
          "#{to}. Nx's binary backend splices the value's raw bits, as #{article(to)} #{to}, " <>
          "into the #{from} tensor (0.5 padded into s32 is 1056964608, and -1 padded onto u8 " <>
          "[1] gives [255, 255]), and EMLX raises; only EXLA converts it.",
      label: "pads #{article(from)} #{from} tensor with a value that makes it #{to}",
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
      label: "multiplies #{type} entries in #{type}",
      help: "make the matrix a float first, as in Nx.as_type(m, :f32)",
      frame: "makes it #{type}:",
      severity: if(String.starts_with?(type, "u"), do: :warning, else: :info)
    }
  end

  def call_error("f32_precision", type, operation) do
    %{
      title: "computes #{article(type)} #{type} result with an f32 constant",
      detail:
        "It divides by the logarithm of its base computed as an f32 tensor, so its #{type} " <>
          "result is accurate only to about seven digits.",
      label: "divides the #{type} by #{base_logarithm(without_arity(operation))} in f32",
      help:
        "divide Nx.log/1 of the tensor by the base's logarithm in #{type}, as in Nx.divide(Nx.log(x), Nx.log(Nx.tensor(2, type: :#{type})))",
      frame: "",
      severity: :info
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  defp negative_wraparound(type) do
    %{
      title: "makes a value that can be negative #{type}",
      detail:
        "The code's math can make the value negative, and it is made #{type}, an unsigned " <>
          "type: it wraps around rather than saturating (-1 becomes " <>
          "#{wrapped_integer(-1, type)} in #{type}).",
      label: "made #{type}, where -1 becomes #{wrapped_integer(-1, type)}",
      help: "clip into the type's range first, as in Nx.clip(x, 0, 255), or use a signed type",
      frame: "",
      severity: :warning
    }
  end

  defp number_wraparound(type, number) do
    {low, high} = integer_range(type)
    wrapped = wrapped_integer(number, type)

    %{
      title: "makes #{number} #{article(type)} #{type}, which holds #{low} to #{high}",
      detail:
        "Nx writes the number into #{type} by its low bits, wrapping it around rather than " <>
          "saturating: the tensor holds #{wrapped} where the code writes #{number}.",
      label: "#{number} becomes #{wrapped} in #{type}",
      help:
        "write a number from #{low} to #{high}, or make the tensor a type that holds #{number}",
      frame: "",
      severity: :warning
    }
  end

  # A number the tensor's own type rounds to zero.
  defp underflow_in_type(number, type) do
    %{
      title: "gets #{number}, which #{type} rounds to zero",
      detail:
        "The number #{number} is below the smallest magnitude #{type} holds, and the tensor " <>
          "it meets is #{type}, which keeps its type: the number is 0 there, and an epsilon " <>
          "meant to keep a value from zero does not.",
      label: "#{number} is 0 in #{type}",
      help: "use an epsilon the type holds (1.0e-4 or larger for f16), or compute in f32",
      frame: "",
      severity: :warning
    }
  end

  # A number Nx makes an f32, which rounds it to zero, before it meets a
  # tensor of a type that would have held it.
  defp underflow_before_meeting(number, [name | _rest], type) do
    %{
      title: "rounds #{number} to zero in f32 before it meets #{type}",
      detail:
        "Outside traced code Nx makes a float number an f32 tensor before it meets another " <>
          "tensor, whatever that tensor's type: #{number} is below the smallest magnitude f32 " <>
          "holds, so it is 0 before it meets the #{type} tensor, which would have held it, and " <>
          "an epsilon meant to keep a value from zero does not.",
      label: "#{number} is 0 in f32, before it meets #{type}",
      help:
        "make the number a tensor of the other's type first, as in Nx.tensor(#{number}, type: :#{name}), or compute it in a defn, where it keeps the merged type",
      frame: "",
      severity: :warning
    }
  end

  # A number past the largest finite value of the tensor's own type.
  defp overflow_in_type(number, type) do
    %{
      title: "makes #{number} infinite in #{type}",
      detail:
        "The number #{number} is past #{type}'s largest finite value#{largest(type)}, and " <>
          "the tensor it meets is #{type}, which keeps its type: the number becomes infinite " <>
          "there, so a -1.0e9 mask gives -Inf, and a softmax over it NaN.",
      label: "#{number} becomes an infinity in #{type}#{largest(type, "largest ")}",
      help:
        "use a number the type holds, such as Nx.Constants.min_finite(Nx.type(x)), or compute in f32",
      frame: "",
      severity: :warning
    }
  end

  # A number Nx makes an f32, which makes it infinite, before it meets a
  # tensor of a type that would have held it.
  defp overflow_before_meeting(number, type) do
    %{
      title: "makes #{number} infinite in f32 before it meets #{type}",
      detail:
        "Outside traced code Nx makes a float number an f32 tensor before it meets another " <>
          "tensor, whatever that tensor's type: #{number} is past f32's largest finite value " <>
          "(3.4028235e38), so it is infinite before it meets the #{type} tensor, which would " <>
          "have held it.",
      label: "#{number} is infinite in f32, before it meets #{type}",
      help:
        "make the number a tensor of the other's type first, as in Nx.tensor(#{number}, type: :#{type}), or compute it in a defn, where it keeps the merged type",
      frame: "",
      severity: :warning
    }
  end

  @impl true
  # A written number past the float type it is made (`f16 -1.0e9`), or
  # past f32, which Nx makes it before it meets a wider type (`f64 1.0e39`).
  def hazard("literal_overflow", cause, _operation) do
    [type, number] = String.split(cause, " ")

    if type in @wider_than_f32,
      do: overflow_before_meeting(number, type),
      else: overflow_in_type(number, type)
  end

  def hazard("cast_overflow", type, _operation) do
    %{
      title: "makes a value infinite in #{type}",
      detail:
        "The value made #{type} holds a number past #{type}'s largest finite value" <>
          "#{largest(type)}: it becomes infinite (-1.0e9, or Nx.Constants.min_finite(:f32), " <>
          "is -Inf in f16), and a fully masked softmax row over it NaN.",
      label: "a number #{past_largest(type)} is infinite in #{type}",
      help:
        "make the value in the target type, as with Nx.Constants.min_finite(:#{type}), or clip it into range first",
      frame: "makes a number past #{type}'s range:",
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
      label: "#{sums(length)} in #{type}, where a sum of ones goes #{past_largest(type)}",
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
      label: "subtracts the maximum in #{type}, where #{zero_minus_one(type)}",
      help: "make the tensor a float first, as in Nx.logsumexp(Nx.as_type(t, :f32))",
      frame: "makes it #{type}:",
      severity: :warning
    }
  end

  def hazard(_kind, _cause, _operation), do: nil

  @impl true
  def type_error("upcast", subject, operation, _position, _certain) do
    [from, to] = String.split(subject, " ")

    %{
      title: "turns #{article(from)} #{from} operand into #{to}",
      detail: upcast_detail(without_arity(operation), from, to),
      label: "#{from} becomes #{to}",
      help: upcast_help(without_arity(operation)),
      frame: "the #{to} operand:",
      severity: :warning
    }
  end

  def type_error("narrowing_merge", subject, _operation, _position, _certain) do
    [from, to] = String.split(subject, " ")

    %{
      title: "merges #{article(from)} #{from} operand into #{to}, which holds less",
      detail:
        "Nx.Type.merge/2 takes #{to} for these operands, which has a smaller range or fewer " <>
          "significand bits than #{from}#{narrowing_example(from, to)}.",
      label: "#{from} becomes #{to}",
      help:
        "convert both operands to a type that holds either (f32, or s64 for integers) before they meet",
      frame: "makes it #{from}:",
      severity: :warning
    }
  end

  def type_error(_kind, _subject, _operation, _position, _certain), do: nil

  defp wraparound_label(operation, type) do
    case {without_arity(operation), String.ends_with?(operation, "/1")} do
      {"Nx.diff", _unary} -> "subtracts neighbors in #{type}, where #{zero_minus_one(type)}"
      {"Nx.linspace", _unary} -> "steps down in #{type}, where #{zero_minus_one(type)}"
      {"Nx.all_close", _unary} -> "subtracts in #{type}, where #{zero_minus_one(type)}"
      {_name, true} -> "negates in #{type}, where -1 is #{wrapped_integer(-1, type)}"
      {_name, false} -> "subtracts in #{type}, where #{zero_minus_one(type)}"
    end
  end

  defp zero_minus_one(type), do: "0 - 1 is #{wrapped_integer(-1, type)}"

  # The first sum past a narrow integer type's largest value, as the type
  # holds it.
  defp one_past(type) do
    {_low, high} = integer_range(type)
    "#{high} + 1 is #{wrapped_integer(high + 1, type)}"
  end

  defp narrow_verb("Nx.median"), do: "adds the two middle values"
  defp narrow_verb(_name), do: "sums and multiplies"

  defp narrow_example("Nx.median", "u8"),
    do: "the median of u8 [130, 140, 150, 160] is 17.0, not 145.0"

  defp narrow_example("Nx.dot", "s8"), do: "the dot product of s8 [100, 100] and [2, 2] is -112"
  defp narrow_example(_name, type), do: one_past(type)

  # What the last index of a written length comes out as in an index type,
  # or the first index past the type's range where the length is not
  # written.
  defp last_index(length, type) do
    index =
      case Integer.parse(length) do
        {count, ""} -> count - 1
        _symbolic -> elem(integer_range(type), 1) + 1
      end

    "index #{index} comes out #{wrapped_integer(index, type)}"
  end

  # How far a type holds a run of whole numbers.
  defp exact_to([type | _rest] = names) do
    if String.starts_with?(type, ["u", "s"]),
      do: "which holds up to #{limits(names)}",
      else: "exact only to #{limits(names)}"
  end

  # A count, a sum, as a label writes it: over a written length, one the
  # code reads, or an axis it does not show.
  defp counts("?"), do: "counts along an axis of unknown length"
  defp counts(length), do: "counts up to #{length}"

  defp sums("?"), do: "sums an axis of unknown length"
  defp sums(length), do: "sums #{length} elements"

  defp base_logarithm("Nx.log2"), do: "ln 2"
  defp base_logarithm("Nx.log10"), do: "ln 10"
  defp base_logarithm(_name), do: "the base's logarithm"

  defp narrowing_example("u64", _to), do: ": a u64 past 2^63 turns negative in s64"
  defp narrowing_example("bf16", "f16"), do: ": a bf16 past 65504 becomes infinite in f16"
  defp narrowing_example(_from, _to), do: ""

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

  # What a run of whole numbers past the types' limits becomes in them.
  defp sequence_effect([type | _rest] = names) do
    if String.starts_with?(type, ["u", "s"]),
      do:
        "whose largest value is #{limits(names)}: past it they wrap around (a u8 iota of 300 " <>
          "goes back to 0 at 256).",
      else:
        "which holds every whole number only up to #{limits(names)}: past it neighbors round " <>
          "to one value (a bf16 iota gives 256, 256, 258, 258, ...), and positions or " <>
          "timesteps built on them repeat."
  end

  # The types' limits, each named where there are several.
  defp limits([type]), do: limit(type)

  defp limits(names),
    do: names |> Enum.map(&"#{limit(&1)} in #{&1}") |> join("and")

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

  defp past_largest(type) do
    case Map.fetch(@largest, type) do
      {:ok, value} -> "past #{value}"
      :error -> "past #{type}'s largest value"
    end
  end

  # A low-range float type's largest finite value, in parentheses after
  # `prefix`, and nothing for another type.
  defp largest(type, prefix \\ "") do
    case Map.fetch(@largest, type) do
      {:ok, value} -> " (#{prefix}#{value})"
      :error -> ""
    end
  end
end
