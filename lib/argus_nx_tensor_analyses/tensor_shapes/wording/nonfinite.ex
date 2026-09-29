defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Nonfinite do
  @moduledoc false
  # What the results `priv/tensor_shapes.dl` finds can be infinite or NaN
  # say: a division by zero, a logarithm of zero or of a negative value, a
  # function taken outside its domain or at its edge, an exponential that
  # overflows, and a gradient that is infinite where its result is zero,
  # each also where nothing checks the operand.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  alias ArgusNxTensorAnalyses.TensorShapes.Wording.Causes

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

  @impl true
  def hazard("outside_domain", cause, operation) do
    %{
      title: "can take a value outside its domain",
      detail:
        "#{domain(operation)}, and its operand can go outside it: #{Causes.domain(cause)}. " <>
          "The result there is NaN.",
      label: "takes it here",
      help:
        "clip the operand into the domain first, such as Nx.clip(x, -1.0, 1.0) before asin or acos",
      frame: "the operand leaves the domain because of this"
    }
  end

  def hazard("infinite_at_edge", cause, operation) do
    %{
      title: "can reach the edge of its domain",
      detail:
        "#{domain(operation)}, and its operand can reach the edge: #{Causes.domain(cause)}. " <>
          "The result there is infinite.",
      label: "takes it here",
      help:
        "keep the operand strictly inside: clip it a little short of the edge, such as Nx.clip(x, -1 + eps, 1 - eps)",
      frame: "the operand reaches the edge because of this"
    }
  end

  def hazard("unchecked_domain", cause, operation) do
    %{
      title: "takes a value nothing keeps in its domain",
      detail:
        "#{domain(operation)}. Its operand can be outside it as far as the code shows: " <>
          "#{Causes.domain(cause)}, and no clip and no test on the way to the call keeps it inside.",
      label: "takes it here",
      help: "clip the operand into the domain, or check it first",
      frame: "the operand can leave the domain because of this"
    }
  end

  def hazard("log_of_zero", "underflow", _operation) do
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

  def hazard("root_of_negative", _cause, _operation) do
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

  def hazard("log_of_negative", _cause, _operation) do
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

  def hazard("exp_overflow", _cause, _operation) do
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

  def hazard("unchecked_root", _cause, _operation) do
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

  def hazard(kind, cause, _operation), do: zero_hazard(kind, Causes.zero(cause))

  # A division, a logarithm or a differentiated root or norm of a value
  # that can be zero, or a division or logarithm of one nothing checks, by
  # kind (nil for another kind); `why` says how the value comes to be
  # zero. Options: `:help`, what to change, for the kind's own; `:sampled`,
  # true for a value that is never negative and is zero only now and then
  # (a sample drawn at its minimum, a product that underflows), which the
  # texts describe more briefly.
  @spec zero_hazard(String.t(), String.t(), keyword()) :: map() | nil
  def zero_hazard(kind, why, options \\ []) do
    case zero_wording(kind, why, Keyword.get(options, :sampled, false)) do
      nil -> nil
      wording -> Map.update!(wording, :help, &Keyword.get(options, :help, &1))
    end
  end

  defp zero_wording("divide_by_zero", why, _sampled) do
    %{
      title: "can divide by zero",
      detail:
        "The divisor cannot be negative, but it can be zero: #{why}. " <>
          "Nx gives an infinity or a NaN there, and an integer quotient or remainder raises.",
      label: "divides here",
      help:
        "keep the divisor away from zero: add a positive epsilon to it, or take Nx.max of it and one",
      frame: "the divisor can be zero because of this"
    }
  end

  defp zero_wording("infinite_gradient", why, sampled) do
    norm = if sampled, do: "NaN", else: "NaN (0/0)"

    %{
      title: "has an infinite gradient where its result is zero",
      detail:
        "A grad differentiates it, and its result cannot be negative but can be zero: " <>
          "#{why}. The derivative of a square root or root there is infinite, and " <>
          "of a norm #{norm}, and the gradient carries it back into everything before it.",
      label: "differentiated here",
      help:
        "keep the operand away from zero where it is differentiated, such as Nx.sqrt(Nx.add(x, 1.0e-12))",
      frame: "the result can be zero because of this"
    }
  end

  defp zero_wording("log_of_zero", why, _sampled) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand cannot be negative, but it can be zero: #{why}. " <>
          "The logarithm of zero is negative infinity.",
      label: "takes the logarithm here",
      help:
        "keep the operand away from zero: add a positive epsilon to it, or take Nx.max of it and one",
      frame: "the operand can be zero because of this"
    }
  end

  defp zero_wording("unchecked_divisor", why, sampled) do
    %{
      title: "divides by a value nothing checks is nonzero",
      detail:
        "The divisor can be zero as far as the code shows: #{why}, and #{checks(sampled)} " <>
          "keeps it from zero. Nx gives an infinity or a NaN there, and an integer quotient or " <>
          "remainder raises.",
      label: "divides here",
      help:
        "check the divisor first (a guard, or Nx.select on Nx.equal(divisor, 0)), or keep it from zero with a positive epsilon",
      frame: "the divisor can be zero because of this"
    }
  end

  defp zero_wording("unchecked_logarithm", why, sampled) do
    {values, results} =
      if sampled,
        do: {"zero", "The logarithm of zero is negative infinity."},
        else:
          {"zero or negative",
           "The logarithm of zero is negative infinity, and of a negative value NaN."}

    %{
      title: "takes the logarithm of a value nothing checks is positive",
      detail:
        "The operand can be #{values} as far as the code shows: #{why}, and " <>
          "#{checks(sampled)} keeps it positive. #{results}",
      label: "takes the logarithm here",
      help:
        "check the operand first (a guard, or Nx.select on Nx.greater(operand, 0)), or take Nx.max of it and a positive epsilon",
      frame: "the operand can be zero because of this"
    }
  end

  defp zero_wording(_kind, _why, _sampled), do: nil

  # What could have kept the operand from zero on the way to the call.
  defp checks(true), do: "no test on the way to the call"
  defp checks(false), do: "no test on the way to the call and no select"

  defp domain(operation) do
    name = without_arity(operation)
    "#{name} #{Map.get(@domains, name, "is defined on only part of the line")}"
  end
end
