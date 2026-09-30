defmodule ArgusNxTensorAnalyses.DocumentedKinds do
  @moduledoc false
  # The kinds of finding `docs/checks.md` documents: the first column of
  # each row of its tables.

  @checks Path.expand("../../docs/checks.md", __DIR__)

  @spec documented() :: [String.t()]
  def documented do
    ~r/^\| `([a-z0-9_]+)` \|/m
    |> Regex.scan(File.read!(@checks), capture: :all_but_first)
    |> List.flatten()
    |> Enum.sort()
  end
end
