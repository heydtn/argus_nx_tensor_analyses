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

  # What each kind of divergence says: its title, why, the label at the
  # call, what to change, and the frame at its origin; nil for a kind it
  # does not describe.
  @spec divergence(String.t(), String.t()) :: map() | nil
  def divergence("narrowed_type", "f64") do
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

  def divergence("narrowed_type", type) do
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

  def divergence("negative_remainder", "divisor") do
    %{
      title: "takes a remainder by a divisor that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, and subtracts the divisor " <>
          "where the dividend is negative: with a negative divisor that is wrong wherever the " <>
          "remainder is not zero, and wherever the dividend is negative. remainder(7, -2) is -1 " <>
          "and remainder(-7, -3) is 2 on EMLX, 1 and -1 on BinaryBackend and EXLA.",
      label: "the divisor can be negative here",
      help:
        "divide by a positive number (the absolute value, flipping the result's sign where it should), or run this remainder off EMLX",
      frame: "the divisor can be negative because of this:"
    }
  end

  def divergence("negative_remainder", _dividend) do
    %{
      title: "takes a remainder of a dividend that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, and subtracts the divisor " <>
          "where the dividend is negative: where a negative dividend divides exactly, EMLX " <>
          "gives minus the divisor instead of 0. remainder(-6, 2) is -2 on EMLX, 0 on " <>
          "BinaryBackend and EXLA.",
      label: "the dividend can be negative here",
      help:
        "keep the dividend from going negative (add a multiple of the divisor first), or select 0 where the result equals minus the divisor",
      frame: "the dividend can be negative because of this:"
    }
  end

  def divergence("negative_integer_power", _exponent) do
    %{
      title: "raises an integer to an exponent that can be negative, which hangs EMLX",
      detail:
        "An integer power with a negative exponent runs forever on EMLX's CPU device when run " <>
          "eagerly, and gives 0 on its GPU and under its compiler, for a base of 1 too. EXLA " <>
          "gives 0 (1 for a base of 1), and BinaryBackend raises.",
      label: "the exponent can be negative here",
      help:
        "make the base a float (Nx.as_type(base, :f32), or 2.0 rather than 2) for a fractional result, or keep the exponent from going negative",
      frame: "the exponent can be negative because of this:"
    }
  end

  def divergence("round_half", cause) do
    %{
      title: "rounds values that can lie on a half, which EMLX rounds to even",
      detail:
        "EMLX rounds a value halfway between two integers to the even one (0.5 to 0, 2.5 to 2), " <>
          "BinaryBackend and EXLA away from zero (to 1 and 3), and #{half_cause(cause)}.",
      label: "rounds here",
      help:
        "round the halves the way you mean explicitly, such as Nx.floor(Nx.add(x, 0.5)) to round them up",
      frame: "puts it on a half:"
    }
  end

  def divergence("wrapped_shift", detail) do
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

  def divergence(_kind, _detail), do: nil

  # A shift's amount as EMLX takes it, modulo the width it shifts in.
  defp wrapped(amount, width) do
    case Integer.parse(amount) do
      {value, ""} -> "one by #{rem(value, width)}"
      _unread -> "one by its amount modulo #{width}"
    end
  end

  # How the rounded value lands on a half, by the cause the program names.
  defp half_cause("halved"), do: "the value rounded is an integer halved, which lies on halves"
  defp half_cause("half_added"), do: "the value rounded is an integer plus a half, always a half"

  defp half_cause("mean"),
    do: "the value rounded is a mean of integers, which lies on a half for an even count"

  defp half_cause(_cause), do: "the value rounded can lie on a half"
end
