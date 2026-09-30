defmodule ArgusNxTensorAnalyses.EMLX.Wording do
  @moduledoc false
  # What the findings of `priv/emlx.dl` say: what EMLX computes otherwise
  # than BinaryBackend and EXLA, and tensors of two backends that meet.

  import ArgusNxTensorAnalyses.Text

  # What a call over tensors of two backends says, the tensor of `backend`
  # put there by the call `shown`.
  @spec mixed_backends(String.t(), String.t(), String.t()) :: map()
  def mixed_backends(backend, other, shown) do
    %{
      title: "gets tensors of #{backend} and #{other}, which cannot meet",
      detail:
        "Nx raises Nx.Defn.IncompatibleBackendsError for a call over tensors of two backends, " <>
          "unless one of them is Nx.BinaryBackend.#{compiled(shown, backend)}",
      label: "gets #{backend} and #{other} here",
      help:
        "move one tensor to the other's backend first, such as Nx.backend_transfer(tensor, EMLX.Backend)"
    }
  end

  # What a compiled function's results are on, where one puts the tensor
  # on its backend.
  defp compiled(shown, backend) do
    if String.starts_with?(shown, ["Nx.Defn.", "EXLA."]),
      do: " A function compiled for #{backend} returns its tensors there, whatever it is handed.",
      else: ""
  end

  # What a frame at the call `shown` that makes a tensor on `backend` says.
  @spec placed_by(String.t(), String.t()) :: String.t()
  def placed_by(shown, backend) do
    if String.starts_with?(shown, ["Nx.backend_", "Nx.Defn.", "EXLA.", "Nx.with_default"]),
      do: "puts a tensor on #{backend}:",
      else: "makes a tensor on #{backend}:"
  end

  # What each kind of divergence at a call of `operation` says: its title,
  # why, the label at the call, what to change, and the frame at its
  # origin; nil for a kind it does not describe. `cause` is how, where the
  # rules say it (`tensor_emlx_divergence`'s column).
  @spec divergence(String.t(), String.t(), String.t(), String.t()) :: map() | nil
  def divergence("narrowed_type", "f64", _cause, _operation) do
    %{
      title: "makes an f64 tensor, which EMLX keeps as f32",
      detail:
        "EMLX has no 64-bit float: on either device it stores an f64 tensor as f32, while the " <>
          "tensor still says f64, so what is computed from it is computed at f32 precision " <>
          "(0.1 reads back as 0.10000000149011612, and 1 + 1.0e-10 as 1.0). BinaryBackend and " <>
          "EXLA compute in f64.",
      label: "makes f64 here",
      help:
        "compute it where f64 exists (Nx.Defn.jit(fun, compiler: EXLA), or backend: EXLA.Backend) and move the result to EMLX as f32, or make it f32",
      frame: ""
    }
  end

  def divergence("narrowed_type", type, _cause, _operation) do
    %{
      title: "makes #{article(type)} #{type} tensor, which EMLX keeps as c64",
      detail:
        "EMLX has no 128-bit complex type: it stores #{article(type)} #{type} tensor as c64, " <>
          "while the tensor still says #{type}, computes at c64 precision, and raises reading " <>
          "it back (Nx.to_number/1, Nx.to_list/1: no function clause matching in " <>
          "EMLX.Backend.maybe_modify_binary/3).",
      label: "makes #{type} here",
      help: "make it c64, or compute it on a backend that has #{type}",
      frame: ""
    }
  end

  def divergence("narrowed_transfer", "c128", _cause, _operation) do
    %{
      title: "moves a c128 tensor onto EMLX, which raises",
      detail:
        "EMLX has no 128-bit complex type, and it raises taking a c128 tensor in: no " <>
          "function clause matching in EMLX.Backend.maybe_modify_binary/3.",
      label: "moves c128 onto EMLX here",
      help:
        "make it c64 before it moves, as in Nx.backend_transfer(Nx.as_type(t, :c64), EMLX.Backend)",
      frame: "makes it c128:"
    }
  end

  def divergence("narrowed_transfer", type, _cause, _operation) do
    %{
      title: "moves #{article(type)} #{type} tensor onto EMLX, which keeps it as f32",
      detail:
        "EMLX has no 64-bit float: the tensor it takes in still says #{type}, but EMLX holds " <>
          "it as f32 from here on, and computes with it at f32 precision (0.1 reads back as " <>
          "0.10000000149011612, and 1 + 1.0e-10 as 1.0).",
      label: "moves #{type} onto EMLX here",
      help:
        "make it f32 before it moves, as in Nx.backend_transfer(Nx.as_type(t, :f32), EMLX.Backend), so its type says what EMLX holds",
      frame: "makes it #{type}:"
    }
  end

  def divergence("negative_remainder", "divisor", cause, _operation) do
    %{
      title: "takes a remainder by a divisor that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, then subtracts the divisor " <>
          "where the dividend is negative, so with a negative divisor it is wrong wherever the " <>
          "remainder is not zero or the dividend is negative: #{divisor_example(cause)}.",
      label: "divides by #{negative_operand(cause)}",
      help:
        "divide by a positive number (the absolute value, flipping the result's sign where it should), or run this remainder off EMLX",
      frame: "can take the divisor below zero:"
    }
  end

  def divergence("negative_remainder", _dividend, cause, _operation) do
    %{
      title: "takes a remainder of a dividend that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, then subtracts the divisor " <>
          "where the dividend is negative, so where a negative dividend divides exactly it " <>
          "gives minus the divisor rather than 0: remainder(-6, 2) is -2 on EMLX, 0 on " <>
          "BinaryBackend and EXLA.",
      label: "takes the remainder of #{negative_operand(cause)}",
      help:
        "keep the dividend from going negative (add a multiple of the divisor first), or select 0 where the result equals minus the divisor",
      frame: "can take the dividend below zero:"
    }
  end

  def divergence("negative_integer_power", _exponent, cause, _operation) do
    exponent = if written?(cause), do: cause, else: "-1"

    label =
      if written?(cause),
        do: "raises integers to the power #{cause}",
        else: "raises integers to #{negative_operand(cause)}"

    %{
      title: "raises an integer to an exponent that can be negative, which hangs EMLX",
      detail:
        "An integer power with a negative exponent, as 2 to the power #{exponent}, runs " <>
          "forever on EMLX's CPU device when run eagerly, and is 0 on its GPU and under its " <>
          "compiler, even for a base of 1. EXLA gives 0 (1 for a base of 1), and BinaryBackend " <>
          "raises ArithmeticError.",
      label: label,
      help:
        "make the base a float (Nx.as_type(base, :f32), or 2.0 rather than 2) for a fractional result, or keep the exponent from going negative",
      frame: "can take the exponent below zero:"
    }
  end

  def divergence("round_half", cause, how, _operation) do
    %{
      title: "rounds values that can lie on a half, which EMLX rounds to even",
      detail:
        "EMLX rounds a value halfway between two integers to the even one, and BinaryBackend " <>
          "and EXLA away from zero: 2.5 rounds to 2 on EMLX and to 3 on them, and 0.5 to 0 and " <>
          "to 1. #{half_cause(cause)}.",
      label: "rounds #{rounded(cause, how)}",
      help:
        "round the halves the way you mean explicitly, such as Nx.floor(Nx.add(x, 0.5)) to round them up",
      frame: "can put the rounded value on a half:"
    }
  end

  def divergence("wrapped_shift", detail, _cause, _operation) do
    [type | amount] = String.split(detail, " ", parts: 2)
    amount = Enum.join(amount)
    width = if type in ~w(u32 u64 s64), do: 64, else: 32

    %{
      title: "shifts #{article(type)} #{type} by #{amount}, which EMLX takes modulo #{width}",
      detail:
        "EMLX shifts #{article(type)} #{type} in #{width} bits and takes the amount modulo " <>
          "#{width}, so a shift by #{amount} is #{wrapped(amount, width)} there. BinaryBackend and EXLA " <>
          "shift every bit out and give 0, or -1 for a negative number shifted right: " <>
          "left_shift(1, 33) of an s32 is 2 on EMLX and 0 on them.",
      label: "shifts by #{amount} here",
      help:
        "keep the amount below #{width}, or select the result for amounts past the type's width (0, or -1 shifting a negative number right)",
      frame: ""
    }
  end

  def divergence(_kind, _detail, _cause, _operation), do: nil

  # A negative operand as a label names it, by how the rules say it is
  # negative: the number written, or what its cause makes it.
  defp negative_operand("input"), do: "a value an input can make negative"
  defp negative_operand("subtract"), do: "a difference that can go below zero"
  defp negative_operand("written"), do: "a written negative number"
  defp negative_operand("sign"), do: "a sign, which is -1 for a negative value"
  defp negative_operand("clip"), do: "a value clipped to a negative bound"

  defp negative_operand(cause),
    do: if(written?(cause), do: cause, else: "a value its math can take below zero")

  # A cause that is the number written as the operand.
  defp written?(cause), do: String.starts_with?(cause, "-")

  # What EMLX and the others give for a remainder by the written divisor,
  # or by two divisors where it is not a written integer.
  defp divisor_example(cause) do
    with {divisor, ""} when divisor < 0 <- Integer.parse(cause),
         dividend when dividend != nil <-
           Enum.find([1, -1, 7, -7], &(emlx_remainder(&1, divisor) != rem(&1, divisor))) do
      "remainder(#{dividend}, #{divisor}) is #{emlx_remainder(dividend, divisor)} on EMLX, " <>
        "#{rem(dividend, divisor)} on BinaryBackend and EXLA"
    else
      _unwritten ->
        "remainder(7, -2) is -1 and remainder(-7, -3) is 2 on EMLX, 1 and -1 on BinaryBackend and EXLA"
    end
  end

  # A remainder of integers as EMLX computes it: MLX's, which has the
  # divisor's sign, less the divisor where the dividend is negative.
  defp emlx_remainder(dividend, divisor) do
    floored = Integer.mod(dividend, divisor)
    if dividend < 0, do: floored - divisor, else: floored
  end

  # What a round rounds, by the class and number that put it on a half.
  defp rounded("mean", _how), do: "a mean of integers"

  defp rounded(_cause, how) do
    case String.split(how, " ", parts: 2) do
      ["divide", number] -> "an integer divided by #{number}"
      ["multiply", number] -> "an integer times #{number}"
      ["add", number] -> "an integer plus #{number}"
      ["subtract", number] -> "the difference of an integer and #{number}"
      _unread -> "a value that can lie on a half"
    end
  end

  # How the rounded value lands on a half, by the cause the program names.
  defp half_cause("halved"),
    do: "An integer halved lies on a half where it is odd, as 5 / 2 is 2.5"

  defp half_cause("half_added"), do: "An integer plus a half always lies on a half"

  defp half_cause("mean"),
    do: "A mean of an even count of integers can lie on a half, as the mean of 2 and 3 is 2.5"

  defp half_cause(_cause), do: "The value rounded can lie on a half"

  # A shift's amount as EMLX takes it, modulo the width it shifts in.
  defp wrapped(amount, width) do
    case Integer.parse(amount) do
      {value, ""} -> "one by #{rem(value, width)}"
      _unread -> "one by its amount modulo #{width}"
    end
  end
end
