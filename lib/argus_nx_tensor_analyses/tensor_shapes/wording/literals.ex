defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Literals do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/literals.dl` say. Each is a
  # `tensor_type_error` whose subject is the literal as written: `300 as
  # u8` for a literal and the type Nx makes it in, `:f32` for a type.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  @impl true
  def type_error("integer_past_s32", subject, operation, _position, _certain) do
    {spelled, _type} = literal(subject)
    value = String.to_integer(spelled)
    wrapped = wrap(value, "s", 32)
    saturated = saturate(value, "s", 32)

    how =
      if data?(operation),
        do: "Nx.tensor gives integer data the type s32 when it is given no :type",
        else:
          "Nx gives an integer literal the type s32 (Nx.Type.infer/1) before it merges it " <>
            "with the other operands' types, so a wider operand does not widen it"

    %{
      title: "makes #{value} an s32, which cannot hold it",
      detail:
        "#{how}: #{value} becomes #{wrapped} (EMLX saturates a scalar to #{saturated} " <>
          "instead).",
      label: "#{value} becomes #{wrapped} here",
      help: past_s32_help(value, data?(operation)),
      frame: "",
      severity: :warning
    }
  end

  def type_error("atom_type_rejected", subject, operation, _position, _certain) do
    tuple = tuple_form(subject)

    detail =
      if String.starts_with?(operation, "Nx.Random.gumbel"),
        do:
          "Nx.Random.gumbel_split/2 hands its :type option to Nx.Type.float?/1 without " <>
            "normalizing it, and Nx.Type.float?/1 has no clause for the short atom " <>
            "#{subject}: it raises FunctionClauseError.",
        else:
          "#{without_arity(operation)} has clauses for a type only in its tuple form, " <>
            "#{tuple}, and raises for the short atom #{subject} (FunctionClauseError, or an " <>
            "ArgumentError from the function it hands the atom to). Most of Nx's functions take " <>
            "either form because they normalize a type with Nx.Type.normalize!/1 first."

    %{
      title: "gets the type #{subject}, which it takes only as a tuple",
      detail: detail,
      label: "raises for #{subject} here",
      help: atom_type_help(tuple),
      frame: "",
      severity: :error
    }
  end

  def type_error("atom_type_misread", subject, operation, _position, _certain) do
    tuple = tuple_form(subject)
    {given, expected} = misread(without_arity(operation), subject, tuple)

    %{
      title: "gets the type #{subject}, and answers as if for another type",
      detail:
        "#{without_arity(operation)} matches a type only in its tuple form, and falls through " <>
          "to its catch-all clause for the short atom #{subject}: it returns #{given}, where " <>
          "for #{tuple} it returns #{expected}.",
      label: "returns #{given} here",
      help: atom_type_help(tuple),
      frame: "",
      severity: :warning
    }
  end

  def type_error("invalid_type", subject, operation, _position, _certain) do
    raised =
      if String.starts_with?(operation, "Nx.Random.gumbel"),
        do:
          "Nx.Random.gumbel hands its :type to Nx.Type.float?/1, which raises FunctionClauseError",
        else: "Nx raises ArgumentError, \"invalid numerical type: #{subject}\""

    %{
      title: "gets the type #{subject}, which Nx does not have",
      detail:
        "#{raised}. Nx's types are s8, s16, s32, s64, u8, u16, u32, u64 (and s2, s4, u2, u4), " <>
          "f8, f16, bf16, f32, f64, f8_e4m3fn, c64 and c128, written as an atom (:f32) or a " <>
          "tuple ({:f, 32}).",
      label: "raises for #{subject} here",
      help: invalid_type_help(subject),
      frame: "",
      severity: :error
    }
  end

  def type_error("literal_wraps", subject, _operation, _position, _certain) do
    {spelled, type} = literal(subject)
    value = String.to_integer(spelled)
    {family, size} = family_size(type)
    {low, high} = integer_range(family, size)
    wrapped = wrap(value, family, size)

    %{
      title: "makes #{value} #{article(type)} #{type}, which holds #{low} to #{high}",
      detail:
        "Nx writes an integer into #{type} modulo 2^#{size}, so the tensor holds #{wrapped} " <>
          "where the code writes #{value} (EMLX saturates a scalar that does not fit 32 bits " <>
          "instead).",
      label: "#{value} becomes #{wrapped} here",
      help: wraps_help(value, low, high),
      frame: "",
      severity: :warning
    }
  end

  def type_error("literal_overflows", subject, _operation, _position, _certain) do
    {spelled, type} = literal(subject)
    {largest, wider} = float_limits(type)

    %{
      title: "makes #{spelled} #{article(type)} #{type}, past its largest value",
      detail:
        "The largest finite #{type} is #{largest}, and #{spelled} is past it, so it becomes " <>
          "an infinity: math over the tensor gives infinities and NaNs.",
      label: "#{spelled} becomes an infinity here",
      help: "make the tensor #{wider}, or scale the value into #{type}'s range",
      frame: "",
      severity: :warning
    }
  end

  def type_error("literal_flushes", subject, _operation, _position, _certain) do
    {spelled, type} = literal(subject)
    smallest = smallest_subnormal(type)

    %{
      title: "makes #{spelled} #{article(type)} #{type}, which rounds it to zero",
      detail:
        "The smallest #{type} above zero is #{smallest}, and #{spelled} is too near zero " <>
          "to round to it, so it becomes 0.0: an epsilon written this way guards nothing, and " <>
          "a division by it divides by zero.",
      label: "#{spelled} becomes 0.0 here",
      help:
        "make the tensor a type whose range holds it (f32, or bf16, which has f32's range), or " <>
          "use a value #{type} holds, such as Nx.Constants.smallest_positive_normal(:#{type})",
      frame: "",
      severity: :warning
    }
  end

  def type_error("float_as_integer", subject, _operation, _position, _certain) do
    {spelled, type} = literal(subject)

    %{
      title: "makes #{spelled} #{article(type)} #{type}, an integer type",
      detail:
        "Nx cannot write a float, NaN or an infinity as an integer: it raises ArgumentError, " <>
          "\"construction of binary failed\", building the data (EMLX truncates a scalar float " <>
          "instead).",
      label: "raises for #{spelled} here",
      help:
        "write an integer, or make the tensor a float type and round it into an integer one: " <>
          "Nx.as_type(Nx.round(tensor), :#{type})",
      frame: "",
      severity: :error
    }
  end

  def type_error(_kind, _subject, _operation, _position, _certain), do: nil

  # `300 as u8` as `{"300", "u8"}`.
  defp literal(subject) do
    [spelled, type] = String.split(subject, " as ", parts: 2)
    {spelled, type}
  end

  defp data?(operation), do: String.starts_with?(operation, "Nx.tensor/")

  # A type's name as its family and size: `u8` as `{"u", 8}`.
  defp family_size("f8_e4m3fn"), do: {"f8_e4m3fn", 8}

  defp family_size(type) do
    [_whole, family, size] = Regex.run(~r/^([a-z]+)(\d+)$/, type)
    {family, String.to_integer(size)}
  end

  # A short atom type as a tuple: `:f32` as `{:f, 32}`.
  defp tuple_form(":" <> type) do
    {family, size} = family_size(type)
    "{:#{family}, #{size}}"
  end

  defp integer_range("s", size), do: {-Integer.pow(2, size - 1), Integer.pow(2, size - 1) - 1}
  defp integer_range("u", size), do: {0, Integer.pow(2, size) - 1}

  # What Nx writes for an integer in a type of the width: its low bits.
  defp wrap(value, "s", size) do
    <<wrapped::signed-size(^size)>> = <<value::size(size)>>
    wrapped
  end

  defp wrap(value, "u", size) do
    <<wrapped::unsigned-size(^size)>> = <<value::size(size)>>
    wrapped
  end

  defp saturate(value, family, size) do
    {low, high} = integer_range(family, size)
    value |> max(low) |> min(high)
  end

  defp past_s32_help(value, data?) do
    type =
      if value <= Integer.pow(2, 63) - 1 and value >= -Integer.pow(2, 63), do: :s64, else: :u64

    if data?,
      do: "give the data a type that holds it: Nx.tensor(data, type: #{inspect(type)})",
      else:
        "make the literal a tensor of a type that holds it: Nx.tensor(#{value}, type: " <>
          "#{inspect(type)}), which Nx merges as its own type"
  end

  defp wraps_help(value, low, high) do
    holding =
      Enum.find(~w(s16 u16 s32 u32 s64 u64), fn type ->
        {family, size} = family_size(type)
        {type_low, type_high} = integer_range(family, size)
        value >= type_low and value <= type_high
      end)

    wider = if holding, do: "#{holding}, say", else: "a float type, as no integer type holds it"

    "make the tensor a type that holds #{value} (#{wider}), or write a value from #{low} to #{high}"
  end

  # The largest finite value of a float type, and a type to make it
  # instead.
  defp float_limits("f16"), do: {"65504.0", "f32, or bf16, which has f32's range"}
  defp float_limits("f8"), do: {"57344.0", "f16, f32 or bf16"}
  defp float_limits("bf16"), do: {"3.3895314e38", "f64 (on a backend that has it)"}
  defp float_limits(type) when type in ["f32", "c64"], do: {"3.4028235e38", "f64 or c128"}

  defp smallest_subnormal("f16"), do: "5.960464477539063e-8"
  defp smallest_subnormal("f8"), do: "1.52587890625e-5"
  defp smallest_subnormal("bf16"), do: "9.183549615799121e-41"
  defp smallest_subnormal(type) when type in ["f32", "c64"], do: "1.401298464324817e-45"

  # What a catch-all clause returns for a short atom type, and what the
  # function returns for the type as a tuple.
  defp misread("Nx.Type.to_complex", _subject, _tuple), do: {"{:c, 64}", "{:c, 128}"}

  defp misread("Nx.Type.to_real", subject, tuple) do
    expected =
      case subject do
        ":c128" -> "{:f, 64}"
        _float -> tuple
      end

    {"{:f, 32}", expected}
  end

  defp misread("Nx.Type.to_aggregate", subject, _tuple) do
    expected =
      case subject do
        ":u64" -> "{:s, 64}"
        ":u" <> _size -> "{:u, 32}"
        ":s" <> _size -> "{:s, 32}"
      end

    {subject, expected}
  end

  defp misread("Nx.Type.infinite_float?", _subject, _tuple), do: {"false", "true"}

  defp misread("Nx.Type.merge_number", _subject, tuple),
    do: {"{:f, 32}", "#{tuple} or a wider type of its family"}

  defp atom_type_help(tuple),
    do: "write the type as a tuple, #{tuple}, or normalize it first: Nx.Type.normalize!(type)"

  # The type a mistaken spelling likely means.
  @meant %{
    ":float16" => ":f16",
    ":half" => ":f16",
    ":float32" => ":f32",
    ":float" => ":f32",
    ":single" => ":f32",
    ":float64" => ":f64",
    ":double" => ":f64",
    ":bfloat16" => ":bf16",
    ":int8" => ":s8",
    ":int16" => ":s16",
    ":int32" => ":s32",
    ":int" => ":s32",
    ":int64" => ":s64",
    ":long" => ":s64",
    ":uint8" => ":u8",
    ":byte" => ":u8",
    ":bool" => ":u8",
    ":uint16" => ":u16",
    ":uint32" => ":u32",
    ":uint64" => ":u64",
    ":complex64" => ":c64",
    ":complex128" => ":c128",
    "{:float, 16}" => "{:f, 16}",
    "{:float, 32}" => "{:f, 32}",
    "{:float, 64}" => "{:f, 64}",
    "{:int, 8}" => "{:s, 8}",
    "{:int, 16}" => "{:s, 16}",
    "{:int, 32}" => "{:s, 32}",
    "{:int, 64}" => "{:s, 64}"
  }

  defp invalid_type_help(subject) do
    case Map.fetch(@meant, subject) do
      {:ok, meant} -> "write #{meant}, Nx's name for the type"
      :error -> "write one of Nx's types, such as :f32 or {:f, 32}"
    end
  end
end
