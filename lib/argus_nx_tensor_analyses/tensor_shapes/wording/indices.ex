defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Indices do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/indices.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  alias ArgusNxTensorAnalyses.TensorShapes.Wording.Causes

  # The function Nx's message names for a complex operand of each call: its
  # own, with every argument it takes, or the one it reaches that checks.
  @rejected_in %{
    "Nx.reduce_max" => "Nx.reduce_max/2",
    "Nx.reduce_min" => "Nx.reduce_min/2",
    "Nx.argmax" => "Nx.argmax/2",
    "Nx.argmin" => "Nx.argmin/2",
    "Nx.window_max" => "Nx.window_max/3",
    "Nx.window_min" => "Nx.window_min/3",
    "Nx.window_scatter_max" => "Nx.window_scatter_max/5",
    "Nx.window_scatter_min" => "Nx.window_scatter_min/5",
    "Nx.cumulative_max" => "Nx.max/2",
    "Nx.cumulative_min" => "Nx.min/2",
    "Nx.sort" => "Nx.sort/2",
    "Nx.argsort" => "Nx.argsort/2",
    "Nx.top_k" => "Nx.argsort/2",
    "Nx.median" => "Nx.sort/2",
    "Nx.mode" => "Nx.sort/2",
    "Nx.logsumexp" => "Nx.reduce_max/2",
    "Nx.clip" => "Nx.clip/2",
    "Nx.Defn.Kernel.max" => "Nx.max/2",
    "Nx.Defn.Kernel.min" => "Nx.min/2",
    "Nx.Defn.Kernel.rem" => "Nx.remainder/2",
    "Nx.Defn.Kernel.__more_than__" => "Nx.greater/2",
    "Nx.Defn.Kernel.__less_than__" => "Nx.less/2",
    "Nx.Defn.Kernel.__more_than_equal_to__" => "Nx.greater_equal/2",
    "Nx.Defn.Kernel.__less_than_equal_to__" => "Nx.less_equal/2"
  }

  @impl true
  def call_error("negative_index", cause, operation) do
    %{
      title: "can get a negative index",
      detail:
        "Its indices can be negative: #{Causes.negative(cause)}. Nx reads no index from the end, " <>
          "and its backends disagree on a negative one, none as the code means: the binary " <>
          "backend raises that it is out of bounds, EXLA clamps a gathered index to 0 and drops " <>
          "an indexed update, and EMLX wraps -1 to the last element and drops an update out " <>
          "of range.",
      label: "gets indices that can be below 0: #{Causes.negative_value(cause)}",
      help:
        "keep the indices of #{without_arity(operation)} in range: replace the ones the code ignores " <>
          "with a valid index through Nx.select and mask their results, and take a modulo that " <>
          "stays positive as Nx.remainder(i + n, n)",
      frame: "can make an index negative:",
      severity: :warning
    }
  end

  def call_error("negative_slice_start", cause, operation) do
    %{
      title: "can start a slice below zero",
      detail:
        "Its start can be negative: #{Causes.negative(cause)}. Nx does not count a negative start " <>
          "from the end, as NumPy does, nor raise: it moves the start to 0, and the slice " <>
          "#{verb(operation)} the first elements (Nx.slice(Nx.iota({6}), [-1], [1]) is [0]).",
      label:
        "#{verb(operation)} from a start that can be below 0: #{Causes.negative_value(cause)}",
      help:
        "count a start from the end yourself, such as Nx.axis_size(t, axis) - length, and " <>
          "keep a window's start at least 0 where its first steps are shorter",
      frame: "can make the start negative:",
      severity: :warning
    }
  end

  def call_error("slice_past_end", detail, operation) do
    %{
      title: "slices past the end of an axis",
      detail:
        "Its start and length run past the end of the axis (#{detail}). Nx does not raise: it moves the " <>
          "start back until the slice fits, and the slice #{verb(operation)} other elements " <>
          "than the code names (Nx.slice(Nx.iota({6}), [5], [3]) is [3, 4, 5]).",
      label: "#{verb(operation)} #{detail}",
      help: "keep the start plus the length within the axis: start earlier, or slice less",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("negative_ddof", ddof, _operation) do
    %{
      title: "is given a negative ddof",
      detail:
        "Its ddof is #{ddof}. Nx divides by the count less ddof and does not check it, so a " <>
          "negative ddof divides by more than the count and gives a smaller spread than any " <>
          "estimator means.",
      label: "given ddof: #{ddof}, so it divides by more than the count",
      help: "use ddof: 0 for the population's spread, or ddof: 1 for the sample's",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("random_range_empty", range, operation) do
    %{
      title: "samples an empty range",
      detail:
        "Its range, #{range}, holds no integer: the maximum is exclusive. #{without_arity(operation)} " <>
          "takes the remainder of random bits by the maximum less the minimum, which is zero: " <>
          "the binary backend raises dividing by zero, and other backends give what the bits " <>
          "hold.",
      label: "samples #{range}, which holds no integer",
      help: "make the maximum at least one more than the minimum",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("random_range_reversed", range, operation) do
    %{
      title: "samples a reversed range",
      detail:
        "Its minimum is above its maximum (#{range}). #{without_arity(operation)} does not check the " <>
          "order: randint takes the remainder by the span as an unsigned integer, which wraps " <>
          "to a huge one and gives values outside the range, and uniform clamps at the " <>
          "minimum, which gives the minimum alone.",
      label: "samples #{range}, its minimum above its maximum",
      help: "pass the minimum first and the maximum second",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("random_range_outside_type", range, operation) do
    [_bounds, name] = String.split(range, " as ")

    %{
      title: "samples a range its type cannot hold",
      detail:
        "It samples #{range}, and the type does not hold every value of it. " <>
          "#{without_arity(operation)} takes the span as an unsigned integer of the type's width and " <>
          "the result in the type, unchecked: a span as wide as the type wraps to zero (the " <>
          "binary backend raises), a wider one wraps short (0 to 300 as u8 gives values " <>
          "below 44), and bounds past the type wrap around it.",
      label: "samples #{range}, which holds #{holds(name)}",
      help: "sample a range the type holds, or a wider type",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("random_type_not_integer", name, operation) do
    class = if String.starts_with?(name, "c"), do: "complex", else: "float"

    %{
      title: "samples integers of a #{class} type",
      detail:
        "#{without_arity(operation)} samples integers only, and the type it would make is " <>
          "#{name}: the type given, or with none given, the type of a bound written as a " <>
          "float. Nx raises: expected integer type, got type #{type_tuple(name)}.",
      label: "samples integers as #{name}",
      help: "give it an integer type (type: :s32), and integer bounds",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("random_bound_truncated", detail, operation) do
    [bound, name] = String.split(detail, " as ")

    %{
      title: "truncates a float bound",
      detail:
        "A bound is written as a float, #{bound}, and #{without_arity(operation)} makes #{name}, " <>
          "the integer type it is given: it truncates the bound to #{truncated(bound)}, and " <>
          "samples another range than the code writes.",
      label: "truncates the bound #{bound} to #{truncated(bound)} in #{name}",
      help: "round the bound to the integer meant, or sample floats with Nx.Random.uniform",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  @impl true
  def hazard("ddof_not_below_count", detail, operation) do
    ["ddof: " <> ddof, "count: " <> count] = String.split(detail, ", ")
    left = String.to_integer(count) - String.to_integer(ddof)

    %{
      title: "divides by a count its ddof leaves #{if(left == 0, do: "zero", else: "negative")}",
      detail:
        "Its written ddof, #{ddof}, is #{if(left == 0, do: "equal to", else: "more than")} the " <>
          "count of values it reduces, #{count}, and #{without_arity(operation)} divides by the " <>
          "count less ddof, unchecked. " <> ddof_consequence(left),
      label: "divides by its count #{count} less ddof: #{ddof}, which is #{left}",
      help:
        "reduce over more values than ddof, or use ddof: 0 where a count can be 1, as a batch of one",
      frame: "because of this"
    }
  end

  def hazard(_kind, _cause, _operation), do: nil

  @impl true
  def type_error("non_integer_start", class, operation, _position, _certain) do
    %{
      title: "starts a slice at a #{class}",
      detail:
        "A start can be a #{class}. #{without_arity(operation)} takes integer starts only, and " <>
          "Nx raises: index must be integer type.",
      label: "gets a #{class} start",
      help: "make the start an integer: round it and then Nx.as_type(start, :s32)",
      frame: "makes it a #{class}:"
    }
  end

  def type_error("complex_operand", _subject, operation, _position, _certain) do
    %{
      title: "gets a complex tensor, which it rejects",
      detail:
        "An operand can be complex, and #{complex_rejection(operation)}. Complex numbers have " <>
          "no order, so a maximum, a sort, a comparison or a rounding has no meaning for them.",
      label: "gets a complex tensor",
      help:
        "take a real value first: Nx.abs/1 for a spectrum's magnitude, or Nx.real/1 and Nx.imag/1",
      frame: "makes it complex:"
    }
  end

  def type_error("complex_spread", _subject, operation, _position, _certain) do
    %{
      title: "squares complex values rather than their magnitudes",
      detail:
        "Its operand can be complex, and #{without_arity(operation)} squares each deviation as it is " <>
          "(x ** 2, not |x| ** 2): the result is complex and not a spread " <>
          "(Nx.variance of [1+i, 0] is 0.5i). Nx does not raise.",
      label: "gets a complex tensor, whose squares are complex",
      help:
        "take the spread of the magnitudes, Nx.abs/1, or of the real and imaginary parts apart",
      frame: "makes it complex:",
      severity: :warning
    }
  end

  def type_error(_kind, _subject, _operation, _position, _certain), do: nil

  # A spread's result where its ddof leaves the count zero or negative.
  defp ddof_consequence(0),
    do:
      "The count less ddof is zero: the result is NaN, or infinite where the values differ, " <>
        "as a variance over axis 0 of a batch of one with ddof: 1 is."

  defp ddof_consequence(_negative),
    do:
      "The count less ddof is negative: a variance or covariance comes out negative, and a " <>
        "standard deviation, its square root, NaN."

  # The values an integer type holds, by its name: u8 holds 0 to 255.
  defp holds(name) do
    {signed, bits} = String.split_at(name, 1)
    width = String.to_integer(bits)

    case signed do
      "u" -> "0 to #{Integer.pow(2, width) - 1}"
      _signed -> "#{-Integer.pow(2, width - 1)} to #{Integer.pow(2, width - 1) - 1}"
    end
  end

  # A type's name as Nx's messages write it: f32 as {:f, 32}, and a name
  # of another form as its atom.
  defp type_tuple(name) do
    case Regex.run(~r/^([a-z]+)(\d+)$/, name, capture: :all_but_first) do
      [letters, bits] -> "{:#{letters}, #{bits}}"
      nil -> ":#{name}"
    end
  end

  # A bound written as a float, truncated to an integer as a cast does.
  defp truncated(bound) do
    {number, _rest} = Float.parse(bound)
    number |> trunc() |> Integer.to_string()
  end

  # A slice reads its elements, and `put_slice` writes them.
  defp verb(operation) do
    if String.starts_with?(operation, "Nx.put_slice/"), do: "writes", else: "reads"
  end

  # What Nx raises for a complex operand of the call, which names the
  # function it rejects it in.
  defp complex_rejection(operation) do
    case without_arity(operation) do
      "Nx.rfft" ->
        "Nx raises: Nx.rfft/2 expects a real tensor"

      name when name in ["Nx.LinAlg.svd", "Nx.LinAlg.pinv", "Nx.LinAlg.norm"] ->
        "Nx raises: Nx.LinAlg.svd/2 is not yet implemented for complex inputs"

      name when name in ["Nx.LinAlg.matrix_rank", "Nx.LinAlg.least_squares"] ->
        "Nx raises: #{name}/2 is not yet implemented for complex inputs"

      name ->
        "Nx raises: #{Map.get(@rejected_in, name, operation)} does not support complex inputs"
    end
  end
end
