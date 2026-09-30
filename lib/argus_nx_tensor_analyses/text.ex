defmodule ArgusNxTensorAnalyses.Text do
  @moduledoc false
  # How the findings of both analyses write what they share: the article
  # before a type, a function without its arity, an argument's position,
  # what an integer type holds and makes of an integer, a list in a
  # sentence.

  @ordinals ~w(first second third fourth fifth)

  # The article before a type's name as it is read: an f64, a u8.
  @spec article(String.t()) :: String.t()
  def article(name), do: if(String.starts_with?(name, ["f", "s"]), do: "an", else: "a")

  # The function a finding names, without its arity: `Nx.take/2` as
  # `Nx.take`.
  @spec without_arity(String.t()) :: String.t()
  def without_arity(operation), do: String.replace(operation, ~r{/\d+$}, "")

  # An argument's position, counted from 0, as a reader counts it: a word
  # for each of the first `spelled` positions, and `6th` past them.
  @spec ordinal(String.t(), 1..5) :: String.t()
  def ordinal(position, spelled) do
    index = String.to_integer(position)
    if index in 0..(spelled - 1), do: Enum.at(@ordinals, index), else: "#{index + 1}th"
  end

  # The integers an integer type holds, by its name: `u8` as `{0, 255}`.
  @spec integer_range(String.t()) :: {integer(), integer()}
  def integer_range(type) do
    {family, size} = integer_type(type)

    case family do
      "s" -> {-Integer.pow(2, size - 1), Integer.pow(2, size - 1) - 1}
      "u" -> {0, Integer.pow(2, size) - 1}
    end
  end

  # What Nx makes of an integer in an integer type, by its name: its low
  # bits, as 300 is 44 in `u8` and 128 is -128 in `s8`.
  @spec wrapped_integer(integer(), String.t()) :: integer()
  def wrapped_integer(value, type) do
    {low, high} = integer_range(type)
    Integer.mod(value - low, high - low + 1) + low
  end

  defp integer_type(type) do
    [_whole, family, size] = Regex.run(~r/^([su])(\d+)$/, type)
    {family, String.to_integer(size)}
  end

  # Items as a sentence lists them: `a`, `a and b`, `a, b and c`, with
  # `conjunction` before the last.
  @spec join([String.t()], String.t()) :: String.t()
  def join([one], _conjunction), do: one

  def join(items, conjunction) do
    {leading, [last]} = Enum.split(items, -1)
    "#{Enum.join(leading, ", ")} #{conjunction} #{last}"
  end
end
