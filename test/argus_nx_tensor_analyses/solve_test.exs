defmodule ArgusNxTensorAnalyses.SolveTest do
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.Solve

  test "the Argus files a program is solved after exist" do
    for path <- Solve.argus_includes(), do: assert(File.regular?(path), path)
  end

  # `priv/argus.dl` copies the Argus words the rules read beyond the files
  # a program is solved after. Each must still be Argus's own, so that an
  # Argus release that changes one fails here rather than solving on.
  test "the Argus words priv/argus.dl copies are Argus's, word for word" do
    argus =
      :argus_beam
      |> Application.app_dir("priv/dl/**/*.dl")
      |> Path.wildcard()
      |> Enum.map_join(" ", &normalized(File.read!(&1)))

    copied =
      :argus_nx_tensor_analyses
      |> Application.app_dir("priv/argus.dl")
      |> File.read!()
      |> statements()

    assert copied != []

    for statement <- copied do
      assert String.contains?(argus, statement), "not in Argus: #{statement}"
    end
  end

  # A program's statements, each with its comments dropped and its
  # whitespace collapsed: a statement starts on a line with no indent.
  defp statements(text) do
    text
    |> uncommented()
    |> String.split("\n")
    |> Enum.chunk_while(
      [],
      fn line, statement ->
        cond do
          String.trim(line) == "" ->
            {:cont, statement}

          statement != [] and not String.starts_with?(line, [" ", "\t"]) ->
            {:cont, Enum.reverse(statement), [line]}

          true ->
            {:cont, [line | statement]}
        end
      end,
      fn
        [] -> {:cont, []}
        statement -> {:cont, Enum.reverse(statement), []}
      end
    )
    |> Enum.map(&normalized(Enum.join(&1, " ")))
  end

  defp normalized(text) do
    text
    |> uncommented()
    |> String.split()
    |> Enum.join(" ")
  end

  defp uncommented(text), do: String.replace(text, ~r{//[^\n]*}, "")
end
