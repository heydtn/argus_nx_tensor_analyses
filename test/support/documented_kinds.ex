defmodule ArgusNxTensorAnalyses.DocumentedKinds do
  @moduledoc false
  # The kinds of finding `docs/checks.md` documents: the first column of
  # each row of its tables.

  @checks Path.expand("../../docs/checks.md", __DIR__)

  @spec documented() :: [String.t()]
  def documented, do: @checks |> File.read!() |> kinds() |> Enum.sort()

  # The kinds each section documents, by its heading (the text after
  # `## `), in order; a section runs to the next such heading, its
  # subsections within it.
  @spec sections() :: [{String.t(), [String.t()]}]
  def sections do
    for section <- @checks |> File.read!() |> String.split(~r/^## /m) |> tl() do
      [heading, body] = String.split(section, "\n", parts: 2)
      {heading, kinds(body)}
    end
  end

  defp kinds(text) do
    ~r/^\| `([a-z0-9_]+)` \|/m
    |> Regex.scan(text, capture: :all_but_first)
    |> List.flatten()
  end
end
