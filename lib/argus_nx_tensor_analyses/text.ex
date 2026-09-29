defmodule ArgusNxTensorAnalyses.Text do
  @moduledoc false
  # How the findings of both analyses write what they share: the article
  # before a type, a function without its arity, an argument's position,
  # a list in a sentence.

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

  # Items as a sentence lists them: `a`, `a and b`, `a, b and c`, with
  # `conjunction` before the last.
  @spec join([String.t()], String.t()) :: String.t()
  def join([one], _conjunction), do: one

  def join(items, conjunction) do
    {leading, [last]} = Enum.split(items, -1)
    "#{Enum.join(leading, ", ")} #{conjunction} #{last}"
  end
end
