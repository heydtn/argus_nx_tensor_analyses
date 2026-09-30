defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Causes do
  @moduledoc false
  # How an operand comes to a value its call is not defined at, by the
  # cause the rules name: a table for each value a finding is about, which
  # copies the texts it shares with another table and writes its own where
  # its findings say it otherwise, and for a zero and a negative value a
  # table of what the value is made of, which a label names at the call. A
  # cause a table does not describe still reads.

  # How a value comes to be zero.
  @zero %{
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

  # What a value that can be zero is made of, as a label names it at the
  # call: the value's maker and where it is zero.
  @zero_value %{
    "square" => "a value made of a square, 0 where its operand is",
    "absolute" => "a value made of an absolute value, 0 where its operand is",
    "root" => "a value made of a square root, 0 where its operand is",
    "norm" => "a value made of a norm, 0 for a zero vector",
    "comparison" => "a value made of a comparison, 0 where it does not hold",
    "index" => "a value made of an iota or index, which starts at 0",
    "identity" => "a value made of an identity matrix, 0 off its diagonal",
    "spread" => "a value made of a spread, 0 where every value is the same",
    "clamp" => "a value clamped at 0",
    "remainder" => "a value made of a remainder, 0 where the division is whole",
    "quotient" => "a value made of an integer quotient, 0 where the dividend is smaller",
    "round" => "a value made of a rounding, 0 from -1 to 1",
    "zero" => "a written 0",
    "input" => "a value from an input, which can be 0",
    "cancel" => "a sum or difference whose terms can cancel to 0",
    "negative" => "a value that can be below 0",
    "sample" => "a random sample, at its minimum of 0 now and then",
    "product_underflow" => "a product of fractions, which underflows to 0"
  }

  # How an operand comes to the edge of a function's domain, or past it.
  @domain @zero
          |> Map.take(["input"])
          |> Map.merge(%{
            "rounding" =>
              "it is within ±1 only before rounding, as a cosine similarity or a vector over its norm is, and rounding can take it just past",
            "saturation" =>
              "it is made of a tanh, erf or sigmoid, which lies within ±1 and rounds to exactly ±1 for large inputs",
            "trigonometric" =>
              "it is made of a sine or cosine, which lies within ±1 and reaches both",
            "clip" => "it is clipped to a bound at the edge",
            "written" => "a written number puts it there",
            "size" => "it is a size, which is at least 1",
            "index" => "it is made of an index or an iota, which counts up from zero",
            "comparison" => "it is made of a comparison, which is 0 or 1",
            "sign" => "it is a sign, which is -1, 0 or 1",
            "identity" => "it is made of an identity matrix, which is 0 or 1",
            "unbounded" => "its math does not keep it within the domain"
          })

  # How the operand of a function a grad differentiates comes to zero, or
  # to the edge of the function's domain.
  @gradient @zero
            |> Map.take(
              ~w(square absolute root norm index identity spread remainder round zero cancel)
            )
            |> Map.merge(Map.take(@domain, ~w(clip saturation trigonometric written sign)))
            |> Map.merge(%{
              "comparison" => "it is made of a comparison, which is 0 where it does not hold",
              "clamp" => "it is clamped at zero",
              "quotient" =>
                "it is an integer quotient, which is zero where the dividend is smaller",
              "input" => "both come from inputs, or from values the analysis cannot follow",
              "rounding" =>
                "it is within ±1 only before rounding, as a cosine similarity is, and reaches ±1 where its vectors line up",
              "size" => "it is a size, which can be 1"
            })

  # How a value comes to be exactly 1.
  @one @domain
       |> Map.take(~w(comparison identity sign))
       |> Map.merge(Map.take(@gradient, ["size"]))
       |> Map.merge(%{
         "clip" => "it is clipped to a bound of 1",
         "written" => "it is written as 1",
         "index" => "it is made of an index or an iota, which counts through 1",
         "saturation" => "it is made of a tanh, erf or sigmoid, which rounds to exactly 1",
         "trigonometric" => "it is made of a sine or cosine, which reaches 1",
         "cosh" => "it is a cosh, which is 1 at zero",
         "rounding" => "it is a ratio that is 1 where its operands meet"
       })

  # How a value goes below zero.
  @negative %{
    "written" => "a written negative number reaches it",
    "subtract" =>
      "it is a subtraction, which goes below zero where it takes more than there is, as one less than an index or position of 0 does",
    "negate" => "it is a negation, which is negative wherever its operand is positive",
    "remainder" =>
      "it is a remainder, which keeps the sign of what it divides: rem(-1, 3) is -1 in Nx and Elixir, where Python's % gives 2",
    "ddof" => "it is a variance whose ddof is past its count, which is negative"
  }

  # What a value that can go below zero is made of, as a label names it at
  # the call.
  @negative_value %{
    "written" => "a written negative number",
    "subtract" => "a subtraction, as 0 - 1 is -1",
    "negate" => "a negation of a positive value",
    "remainder" => "a remainder of a negative dividend",
    "ddof" => "a variance whose ddof is past its count"
  }

  # How a value never below zero comes to be exactly zero now and then.
  @sampled %{
    "sample" =>
      "it is a random sample, which is exactly its minimum now and then: a uniform sample " <>
        "from zero is zero once in 2^7 elements in bf16, 2^10 in f16 and 2^23 in f32, and an " <>
        "integer sample from zero once in as many draws as its range is wide",
    "product_underflow" =>
      "it is a product of fractions over many elements, which underflows to zero: 0.01 to " <>
        "the 23rd is below the smallest f32"
  }

  @spec zero(String.t()) :: String.t()
  def zero(cause), do: Map.get(@zero, cause, "its math lets it be zero")

  @spec zero_value(String.t()) :: String.t()
  def zero_value(cause), do: Map.get(@zero_value, cause, "a value its math lets be 0")

  @spec domain(String.t()) :: String.t()
  def domain(cause), do: Map.get(@domain, cause, "its math takes it there")

  @spec gradient(String.t()) :: String.t()
  def gradient(cause), do: Map.get(@gradient, cause, "its math takes it there")

  @spec one(String.t()) :: String.t()
  def one(cause), do: Map.get(@one, cause, "its math takes it there")

  @spec negative(String.t()) :: String.t()
  def negative(cause), do: Map.get(@negative, cause, "its math takes it below zero")

  @spec negative_value(String.t()) :: String.t()
  def negative_value(cause), do: Map.get(@negative_value, cause, "a value its math takes below 0")

  # nil for a cause that does not make a value zero now and then.
  @spec sampled(String.t()) :: String.t() | nil
  def sampled(cause), do: Map.get(@sampled, cause)
end
