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

  # Each function defined on only part of the line, by its name: where it
  # is defined, the values outside that where it is NaN, the edge where it
  # is infinite (nil for none), all the values it is not finite at, and
  # how to keep an operand inside.
  @domains %{
    "Nx.asin" => %{
      defined: "is defined on [-1, 1]",
      outside: "outside [-1, 1]",
      edge: nil,
      excluded: "outside [-1, 1]",
      inside: "Nx.clip(x, -1.0, 1.0)"
    },
    "Nx.acos" => %{
      defined: "is defined on [-1, 1]",
      outside: "outside [-1, 1]",
      edge: nil,
      excluded: "outside [-1, 1]",
      inside: "Nx.clip(x, -1.0, 1.0)"
    },
    "Nx.atanh" => %{
      defined: "is defined between -1 and 1, and infinite at ±1",
      outside: "outside [-1, 1]",
      edge: "±1",
      excluded: "outside (-1, 1)",
      inside: "Nx.clip(x, -1 + 1.0e-6, 1 - 1.0e-6)"
    },
    "Nx.erf_inv" => %{
      defined: "is defined between -1 and 1, and infinite at ±1",
      outside: "outside [-1, 1]",
      edge: "±1",
      excluded: "outside (-1, 1)",
      inside: "Nx.clip(x, -1 + 1.0e-6, 1 - 1.0e-6)"
    },
    "Nx.log1p" => %{
      defined: "is defined above -1, and negative infinity at -1",
      outside: "below -1",
      edge: "-1",
      excluded: "at -1 or below",
      inside: "Nx.max(x, -1 + 1.0e-6)"
    },
    "Nx.acosh" => %{
      defined: "is defined from 1 up",
      outside: "below 1",
      edge: nil,
      excluded: "below 1",
      inside: "Nx.max(x, 1.0)"
    }
  }

  # A function this module has no domain for.
  @unknown_domain %{
    defined: "is defined on only part of the line",
    outside: "outside its domain",
    edge: nil,
    excluded: "outside its domain",
    inside: "a clip into the domain"
  }

  @impl true
  def hazard("outside_domain", cause, operation) do
    %{
      title: "can take a value outside its domain",
      detail:
        "#{defined(operation)}, and its operand can go outside it: #{Causes.domain(cause)}. " <>
          "The result there is NaN.",
      label: "takes a value #{domain(operation).outside}",
      help: "keep the operand in the domain first, such as #{domain(operation).inside}",
      frame: "takes it #{domain(operation).outside}:"
    }
  end

  def hazard("infinite_at_edge", cause, operation) do
    %{
      title: "can reach the edge of its domain",
      detail:
        "#{defined(operation)}, and its operand can reach the edge: #{Causes.domain(cause)}. " <>
          "The result there is infinite.",
      label: "takes a value that reaches #{edge(operation)}",
      help: "keep the operand strictly inside, such as #{domain(operation).inside}",
      frame: "takes it to #{edge(operation)}:"
    }
  end

  def hazard("unchecked_domain", cause, operation) do
    %{
      title: "takes a value nothing keeps in its domain",
      detail:
        "#{defined(operation)}. Its operand can be outside it as far as the code shows: " <>
          "#{Causes.domain(cause)}, and no clip and no test on the way to the call keeps it inside.",
      label: "takes a value that can be #{domain(operation).excluded}",
      help:
        "keep the operand in the domain, such as #{domain(operation).inside}, or check it first",
      frame: "can take it #{domain(operation).excluded}:"
    }
  end

  def hazard("log_of_zero", "underflow", _operation) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand is a softmax written out: an exponential far below the largest underflows " <>
          "to zero, and the logarithm of zero is negative infinity.",
      label: "takes the logarithm of a softmax, which underflows to 0",
      help:
        "take the logarithm of a softmax directly: Nx.subtract(x, Nx.logsumexp(x, axes: [...], keep_axes: true))",
      frame: "takes the softmax:"
    }
  end

  def hazard("root_of_negative", _cause, operation) do
    {title, _unchecked, verb} = root(operation)

    %{
      title: title,
      detail:
        "Its operand is a difference of two values that cannot be negative, such as a variance " <>
          "written as E[x²] - E[x]². Where they are close, rounding takes the difference below " <>
          "zero, and the result there is NaN.",
      label: "#{verb} a difference rounding can take below 0",
      help:
        "clamp the difference at zero first (Nx.max of it and 0), or use a form that cannot go negative, such as Nx.variance/2",
      frame: "takes the difference:"
    }
  end

  def hazard("log_of_negative", _cause, _operation) do
    %{
      title: "can take the logarithm of a negative value",
      detail:
        "Its operand is a difference of two values that cannot be negative. Where they are " <>
          "close, rounding takes the difference below zero, and the logarithm there is NaN.",
      label: "takes the logarithm of a difference rounding can take below 0",
      help: "keep the difference above zero first (Nx.max of it and a positive epsilon)",
      frame: "takes the difference:"
    }
  end

  def hazard("exp_overflow", _cause, _operation) do
    %{
      title: "can overflow its exponentials",
      detail:
        "It normalizes exponentials of values not shifted down by their maximum first: above " <>
          "about 88 in f32 and bf16 (11 in f16) the exponential overflows to infinity, and " <>
          "infinity over infinity, or its logarithm, is not finite.",
      label: "normalizes exponentials not shifted by their maximum",
      help:
        "subtract the maximum first (Nx.subtract(x, Nx.reduce_max(x, axes: [...], keep_axes: true))), which leaves the result as it is, or use Nx.logsumexp/2",
      frame: "can overflow to infinity:"
    }
  end

  def hazard("unchecked_root", _cause, operation) do
    {_negative, title, verb} = root(operation)

    %{
      title: title,
      detail:
        "Nothing in how the operand is computed keeps it from going below zero, and no test " <>
          "on the way to the call and no select checks it. The result for a negative value " <>
          "is NaN.",
      label: "#{verb} a value that can be below 0",
      help: "check the operand first, or clamp it at zero (Nx.max of it and 0)",
      frame: "can make it negative:"
    }
  end

  def hazard(kind, cause, operation), do: zero_hazard(kind, cause, operation)

  # A division, a logarithm or a differentiated root or norm of a value
  # that can be zero, or a division or logarithm of one nothing checks, by
  # kind (nil for another kind), for the cause of the zero. Option
  # `:help`, what to change, for the kind's own. A cause that makes a
  # value never negative zero only now and then (a sample drawn at its
  # minimum, a product that underflows) is described more briefly.
  @spec zero_hazard(String.t(), String.t(), String.t(), keyword()) :: map() | nil
  def zero_hazard(kind, cause, operation, options \\ []) do
    sampled = Causes.sampled(cause)
    why = sampled || Causes.zero(cause)

    case zero_wording(kind, why, Causes.zero_value(cause), operation, sampled != nil) do
      nil -> nil
      wording -> Map.update!(wording, :help, &Keyword.get(options, :help, &1))
    end
  end

  defp zero_wording("divide_by_zero", why, value, operation, _sampled) do
    %{
      title: "can divide by zero",
      detail:
        "The divisor cannot be negative, but it can be zero: #{why}. " <>
          "Nx gives an infinity or a NaN there, and an integer quotient or remainder raises.",
      label: divides(operation, value),
      help:
        "keep the divisor away from zero: add a positive epsilon to it, or take Nx.max of it and a positive floor (1 for a count)",
      frame: "can make the divisor 0:"
    }
  end

  defp zero_wording("infinite_gradient", why, value, operation, _sampled) do
    {title, derivative, help} = differentiated(without_arity(operation))

    %{
      title: "has #{title} gradient where its result is zero",
      detail:
        "A grad differentiates it, and its result cannot be negative but can be zero: " <>
          "#{why}. The derivative #{derivative}, and the gradient carries it back into " <>
          "everything before it.",
      label: "differentiated at #{value}",
      help: help,
      frame: "can make the result 0:"
    }
  end

  defp zero_wording("log_of_zero", why, value, _operation, _sampled) do
    %{
      title: "can take the logarithm of zero",
      detail:
        "Its operand cannot be negative, but it can be zero: #{why}. " <>
          "The logarithm of zero is negative infinity.",
      label: "takes the logarithm of #{value}",
      help:
        "keep the operand away from zero: add a positive epsilon to it, or take Nx.max of it and a positive floor (1 for a count)",
      frame: "can make the operand 0:"
    }
  end

  defp zero_wording("unchecked_divisor", why, value, operation, sampled) do
    %{
      title: "divides by a value nothing checks is nonzero",
      detail:
        "The divisor can be zero as far as the code shows: #{why}, and #{checks(sampled)} " <>
          "keeps it from zero. Nx gives an infinity or a NaN there, and an integer quotient or " <>
          "remainder raises.",
      label: divides(operation, value),
      help:
        "check the divisor first (a guard, or Nx.select on Nx.equal(divisor, 0)), or keep it from zero with a positive epsilon",
      frame: "can make the divisor 0:"
    }
  end

  defp zero_wording("unchecked_logarithm", why, value, _operation, sampled) do
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
      label: "takes the logarithm of #{value}",
      help:
        "check the operand first (a guard, or Nx.select on Nx.greater(operand, 0)), or take Nx.max of it and a positive epsilon",
      frame: "can make the operand 0:"
    }
  end

  defp zero_wording(_kind, _why, _value, _operation, _sampled), do: nil

  # How a call whose result is zero is differentiated there: a gradient
  # the title names, its derivative, and how to keep it finite.
  defp differentiated("Nx.LinAlg.norm") do
    {"a NaN", "of a norm there is 0/0, NaN",
     "keep the norm's operand away from zero where it is differentiated, or take a vector's norm with an epsilon inside the root: Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(x, x)), 1.0e-6))"}
  end

  defp differentiated("Nx.standard_deviation") do
    {"a NaN", "of a standard deviation there is 0/0, NaN",
     "take the root of the variance plus an epsilon instead: Nx.sqrt(Nx.add(Nx.variance(x), 1.0e-5))"}
  end

  defp differentiated(_root) do
    {"an infinite", "of a root there is infinite",
     "keep the operand away from zero where it is differentiated, such as Nx.sqrt(Nx.add(x, 1.0e-6))"}
  end

  # What could have kept the operand from zero on the way to the call.
  defp checks(true), do: "no test on the way to the call"
  defp checks(false), do: "no test on the way to the call and no select"

  # What a call does with a divisor that can be 0: a power raises it to a
  # negative exponent, and a reciprocal square root takes its root first.
  defp divides(operation, value) do
    case without_arity(operation) do
      name when name in ["Nx.pow", "Nx.Defn.Kernel.**"] -> "raises to a negative power #{value}"
      "Nx.rsqrt" -> "takes the reciprocal square root of #{value}"
      _division -> "divides by #{value}"
    end
  end

  # What a root does to a negative operand, as a title says it where one
  # can be and where nothing checks it, and as a label names it: a square
  # root, its reciprocal, or a power written as a fraction.
  defp root(operation) do
    case without_arity(operation) do
      "Nx.sqrt" ->
        {"can take the square root of a negative value",
         "takes the square root of a value nothing checks is not negative",
         "takes the square root of"}

      "Nx.rsqrt" ->
        {"can take the reciprocal square root of a negative value",
         "takes the reciprocal square root of a value nothing checks is not negative",
         "takes the reciprocal square root of"}

      _power ->
        {"can raise a negative value to a fractional power",
         "raises a value nothing checks is not negative to a fractional power",
         "raises to a fractional power"}
    end
  end

  defp defined(operation), do: "#{without_arity(operation)} #{domain(operation).defined}"

  defp edge(operation), do: domain(operation).edge || "the edge of its domain"

  defp domain(operation), do: Map.get(@domains, without_arity(operation), @unknown_domain)
end
