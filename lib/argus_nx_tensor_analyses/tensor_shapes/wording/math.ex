defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Math do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/math.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  # How a value can be zero, for the causes these rules add.
  @zero_causes %{
    "sample" =>
      "it is a random sample, which is exactly its minimum now and then: a uniform sample " <>
        "from zero is zero once in 2^7 elements in bf16, 2^10 in f16 and 2^23 in f32, and an " <>
        "integer sample from zero once in as many draws as its range is wide",
    "product_underflow" =>
      "it is a product of fractions over many elements, which underflows to zero: 0.01 to " <>
        "the 23rd is below the smallest f32"
  }

  @impl true
  def hazard("exp_overflow", "softplus", _operation) do
    %{
      title: "can overflow its exponential",
      detail:
        "It is a softplus written out, the logarithm of one plus an exponential, and the " <>
          "exponent can be positive: above about 88 in f32 and bf16 (11 in f16) the " <>
          "exponential overflows to infinity, and so does the result. Its gradient there is " <>
          "infinity over infinity, NaN.",
      label: "takes the logarithm here",
      help:
        "write the softplus in its stable form, Nx.max(x, 0) + Nx.log1p(Nx.exp(-Nx.abs(x))), which is the same function (Nx has no softplus of its own)",
      frame: "the exponential that can overflow"
    }
  end

  def hazard("exp_overflow", "logistic", _operation) do
    %{
      title: "can overflow its exponentials",
      detail:
        "It is the logistic function written out, an exponential over one plus the same " <>
          "exponential, and the exponent can be positive: above about 88 in f32 and bf16 " <>
          "(11 in f16) both overflow to infinity, and infinity over infinity is NaN.",
      label: "divides here",
      help: "use Nx.sigmoid(x), which is the same function, computed without overflowing",
      frame: "the exponential that can overflow"
    }
  end

  def hazard("log_of_zero", "sigmoid", _operation) do
    %{
      title: "can take the logarithm of a sigmoid that underflows to zero",
      detail:
        "Its operand is a sigmoid of a value that can be negative. Below about -100 in f32 " <>
          "and bf16 (-17 in f16) the sigmoid rounds to zero, and its logarithm is negative " <>
          "infinity where the log-sigmoid is a finite negative number.",
      label: "takes the logarithm here",
      help:
        "write the log-sigmoid in its stable form, Nx.min(x, 0) - Nx.log1p(Nx.exp(-Nx.abs(x))), which is minus the softplus of -x (Nx has no log-sigmoid of its own)",
      frame: "the sigmoid that can underflow"
    }
  end

  def hazard("log_base_one", cause, _operation) do
    %{
      title: "can take a logarithm to base 1",
      detail:
        "Nx.log/2 divides the logarithm of its operand by the logarithm of its base, and the " <>
          "base can be exactly 1: #{base_cause(cause)}. The logarithm of 1 is zero, so the " <>
          "result there is an infinity or NaN, and Nx raises for a base that is the number 1.",
      label: "takes the logarithm here",
      help:
        "keep the base away from 1, or take the logarithm in a fixed base (Nx.log2/1, Nx.log10/1)",
      frame: "the base can be 1 because of this"
    }
  end

  def hazard(kind, cause, "Nx.logsumexp/" <> _arity)
      when kind in ["log_of_zero", "unchecked_logarithm"],
      do: scaling_hazard(kind, cause)

  def hazard(kind, cause, _operation) when is_map_key(@zero_causes, cause),
    do: zero_hazard(kind, Map.fetch!(@zero_causes, cause))

  def hazard(_kind, _cause, _operation), do: nil

  # A log-sum-exp whose scaling factor can be zero or negative.
  defp scaling_hazard("log_of_zero", _cause) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "It scales its exponentials by its :exp_scaling_factor before it sums them, and the " <>
          "factor cannot be negative but can be zero everywhere a sum takes, as a mask that " <>
          "selects nothing is. The sum is zero there, and its logarithm negative infinity.",
      label: "takes the logarithm here",
      help:
        "keep every reduced slice of the factor from being all zero, or handle the empty slice apart, such as with Nx.select on its sum",
      frame: "the factor can be zero because of this"
    }
  end

  defp scaling_hazard("unchecked_logarithm", "negative") do
    %{
      title: "scales its exponentials by a factor nothing keeps positive",
      detail:
        "It scales its exponentials by its :exp_scaling_factor before it sums them and takes " <>
          "the logarithm, and nothing in how the factor is computed keeps it from going below " <>
          "zero. The logarithm of a negative sum is NaN.",
      label: "takes the logarithm here",
      help: "keep the factor from going below zero, such as with Nx.abs/1 or Nx.max of it and 0",
      frame: "the factor can be negative because of this"
    }
  end

  defp scaling_hazard("unchecked_logarithm", _cause) do
    %{
      title: "scales its exponentials by a factor nothing checks is nonzero",
      detail:
        "It scales its exponentials by its :exp_scaling_factor before it sums them and takes " <>
          "the logarithm, and the factor can be zero as far as the code shows. Where it is " <>
          "zero everywhere a sum takes, the logarithm is negative infinity.",
      label: "takes the logarithm here",
      help: "check the factor first, or keep it from zero with a positive epsilon",
      frame: "the factor can be zero because of this"
    }
  end

  # A division, logarithm or root of a value these rules let be zero.
  defp zero_hazard("divide_by_zero", why) do
    %{
      title: "can divide by zero",
      detail:
        "The divisor cannot be negative, but it can be zero: #{why}. Nx gives an infinity or " <>
          "a NaN there, and an integer quotient or remainder raises.",
      label: "divides here",
      help: zero_help(),
      frame: "the divisor can be zero because of this"
    }
  end

  defp zero_hazard("log_of_zero", why) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand cannot be negative, but it can be zero: #{why}. The logarithm of zero " <>
          "is negative infinity.",
      label: "takes the logarithm here",
      help: zero_help(),
      frame: "the operand can be zero because of this"
    }
  end

  defp zero_hazard("infinite_gradient", why) do
    %{
      title: "has an infinite gradient where its result is zero",
      detail:
        "A grad differentiates it, and its result cannot be negative but can be zero: " <>
          "#{why}. The derivative of a square root or root there is infinite, and of a norm " <>
          "NaN, and the gradient carries it back into everything before it.",
      label: "differentiated here",
      help: zero_help(),
      frame: "the result can be zero because of this"
    }
  end

  defp zero_hazard("unchecked_divisor", why) do
    %{
      title: "divides by a value nothing checks is nonzero",
      detail:
        "The divisor can be zero as far as the code shows: #{why}, and no test on the way " <>
          "to the call keeps it from zero. Nx gives an infinity or a NaN there, and an " <>
          "integer quotient or remainder raises.",
      label: "divides here",
      help: zero_help(),
      frame: "the divisor can be zero because of this"
    }
  end

  defp zero_hazard("unchecked_logarithm", why) do
    %{
      title: "takes the logarithm of a value nothing checks is positive",
      detail:
        "The operand can be zero as far as the code shows: #{why}, and no test on the way " <>
          "to the call keeps it positive. The logarithm of zero is negative infinity.",
      label: "takes the logarithm here",
      help: zero_help(),
      frame: "the operand can be zero because of this"
    }
  end

  defp zero_hazard(_kind, _why), do: nil

  defp zero_help do
    "keep the value away from zero: sample from a positive minimum (such as the type's smallest positive normal number), add a positive epsilon, or sum logarithms rather than take the logarithm of a product"
  end

  # How a logarithm's base can be exactly 1, by the region cause.
  @base_causes %{
    "size" => "it is a size, which can be 1",
    "clip" => "it is clipped to a bound of 1",
    "written" => "it is written as 1",
    "index" => "it is made of an index or an iota, which counts through 1",
    "comparison" => "it is made of a comparison, which is 0 or 1",
    "identity" => "it is made of an identity matrix, which is 0 or 1",
    "sign" => "it is a sign, which is -1, 0 or 1",
    "saturation" => "it is made of a tanh, erf or sigmoid, which rounds to exactly 1",
    "trigonometric" => "it is made of a sine or cosine, which reaches 1",
    "cosh" => "it is a cosh, which is 1 at zero",
    "rounding" => "it is a ratio that is 1 where its operands meet"
  }

  defp base_cause(cause), do: Map.get(@base_causes, cause, "its math takes it there")

  @impl true
  def call_error("nan_comparison", relation, _operation) do
    value = if relation == "not_equal", do: "1", else: "0"

    %{
      title: "compares with NaN, so it is always #{value}",
      detail:
        "One of its operands is NaN, and NaN compares false with everything, itself " <>
          "included: the result is #{value} whatever the other operand holds.",
      label: "compares here",
      help: "test for NaN with Nx.is_nan/1",
      frame: "the NaN comes from",
      severity: :warning
    }
  end

  def call_error("integer_negative_power", _detail, _operation) do
    %{
      title: "raises an integer to a power that can be negative",
      detail:
        "Its base and exponent can both be integers, which makes it an integer power, and the " <>
          "exponent can be negative, where an integer power has no integer value. Nx's binary " <>
          "backend raises ArithmeticError there, EXLA and EMLX on the GPU give 0, and EMLX on " <>
          "the CPU hangs.",
      label: "raises to the power here",
      help:
        "make the base a float (Nx.as_type(base, :f32)) to get fractions, or keep the exponent from going below zero",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("constant_type", name, operation) do
    %{
      title: "is made in #{name}, which has no such value",
      detail: "#{constant_types(operation)} Nx raises for #{name}.",
      label: "makes the constant here",
      help: "make the constant in a type that has it, and convert the result if need be",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("rounded_logarithm", rounding, _operation) do
    {title, taken} = rounding(rounding)

    %{
      title: title,
      detail:
        "A logarithm computed in floating point misses the whole number at an exact power by " <>
          "a rounding error: Nx.log2 of 8192 is just under 13, and Nx.log10 of 10^29 is " <>
          "29.000002. The #{taken} is off by one there.",
      label: "rounds here",
      help:
        "count bits or digits with integer operations (for a power of two, 31 minus Nx.count_leading_zeros/1 of an s32), or correct the result where the power it gives misses the operand",
      frame: "the logarithm is taken here",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # What a call does to a logarithm, and the whole number it takes.
  defp rounding("floor"), do: {"takes the floor of a logarithm to base 2 or 10", "floor"}
  defp rounding("ceil"), do: {"takes the ceiling of a logarithm to base 2 or 10", "ceiling"}

  defp rounding(_truncation),
    do: {"converts a logarithm to base 2 or 10 to integers", "integer it truncates to"}

  # Which types have a constant, by the function that makes it.
  defp constant_types(operation) do
    name =
      operation
      |> String.replace(~r{/\d+$}, "")
      |> String.replace_prefix("Nx.Constants.", "")

    cond do
      name in ~w(nan infinity neg_infinity) ->
        "NaN and the infinities are values of floating-point types only."

      name in ~w(max_finite min_finite max min) ->
        "Complex numbers have no order, and so no largest or smallest value."

      name == "i" ->
        "The imaginary unit is a value of complex types only."

      true ->
        "Nx makes this constant in floating-point types only."
    end
  end
end
