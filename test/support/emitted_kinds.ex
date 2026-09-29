defmodule ArgusNxTensorAnalyses.EmittedKinds do
  @moduledoc false
  # The kinds of finding the rules under `priv/` can emit into a relation:
  # the string literals in its declared `kind` column across its clauses'
  # heads, and where a head takes the kind from a variable, the literals
  # that variable takes from the positive atoms of the body, followed
  # through those relations' heads in turn.

  @priv Path.expand("../../priv", __DIR__)

  # Each relation's kinds, sorted, read from the rules as they are now.
  @spec emitted([String.t()]) :: %{String.t() => [String.t()]}
  def emitted(relations) do
    statements =
      "#{@priv}/**/*.dl"
      |> Path.wildcard()
      |> Enum.flat_map(&(&1 |> File.read!() |> uncommented() |> statements()))

    clauses = for {:clause, heads, body} <- statements, head <- heads, do: {head, body}
    columns = for {:decl, name, names} <- statements, into: %{}, do: {name, names}

    Map.new(relations, fn relation ->
      column = columns |> Map.fetch!(relation) |> Enum.find_index(&(&1 == "kind"))
      {relation, clauses |> literals({relation, column}, MapSet.new()) |> Enum.sort()}
    end)
  end

  # The literals the column of the relation takes in the heads of its
  # clauses, a column taken from a variable followed into the body.
  defp literals(clauses, {relation, column} = place, seen) do
    seen = MapSet.put(seen, place)

    for {{^relation, arguments}, body} <- clauses,
        argument = Enum.at(arguments, column),
        argument != nil,
        literal <- argument_literals(clauses, argument, body, seen),
        into: MapSet.new(),
        do: literal
  end

  defp argument_literals(clauses, argument, body, seen) do
    cond do
      literal = string_literal(argument) ->
        [literal]

      Regex.match?(~r/^[a-z][A-Za-z0-9_]*$/, argument) ->
        for {:atom, name, arguments} <- body,
            {^argument, position} <- Enum.with_index(arguments),
            not MapSet.member?(seen, {name, position}),
            literal <- literals(clauses, {name, position}, seen),
            do: literal

      true ->
        []
    end
  end

  defp string_literal(text) do
    case Regex.run(~r/^"((?:[^"\\]|\\.)*)"$/s, text) do
      [_whole, literal] -> literal
      nil -> nil
    end
  end

  # The text without its comments, strings kept whole.
  defp uncommented(text), do: text |> String.to_charlist() |> strip([]) |> to_string()

  defp strip([], kept), do: Enum.reverse(kept)
  defp strip([?/, ?/ | rest], kept), do: rest |> Enum.drop_while(&(&1 != ?\n)) |> strip(kept)
  defp strip([?/, ?* | rest], kept), do: rest |> block_comment() |> strip(kept)
  defp strip([?" | rest], kept), do: string(rest, [?" | kept])
  defp strip([character | rest], kept), do: strip(rest, [character | kept])

  defp block_comment([?*, ?/ | rest]), do: rest
  defp block_comment([_character | rest]), do: block_comment(rest)
  defp block_comment([]), do: []

  defp string([?\\, character | rest], kept), do: string(rest, [character, ?\\ | kept])
  defp string([?" | rest], kept), do: strip(rest, [?" | kept])
  defp string([character | rest], kept), do: string(rest, [character | kept])
  defp string([], kept), do: Enum.reverse(kept)

  # The declarations and clauses of a file: a directive holds its line; a
  # clause starts at the start of a line and runs to the `.` that ends
  # it.
  defp statements(text) do
    {statements, _open} =
      text
      |> String.split("\n")
      |> Enum.reduce({[], nil}, fn
        line, {statements, nil} ->
          cond do
            String.starts_with?(line, ".decl ") -> {[declaration(line) | statements], nil}
            Regex.match?(~r/^[A-Za-z_$]/, line) -> close(statements, line)
            true -> {statements, nil}
          end

        line, {statements, open} ->
          close(statements, open <> "\n" <> line)
      end)

    statements |> Enum.reject(&is_nil/1) |> Enum.reverse()
  end

  # A clause whose text ends at its `.`, or one still open.
  defp close(statements, text) do
    trimmed = String.trim_trailing(text)

    if String.ends_with?(trimmed, ".") and balanced?(trimmed),
      do: {[clause(String.trim_trailing(trimmed, ".")) | statements], nil},
      else: {statements, text}
  end

  defp balanced?(text), do: text |> String.to_charlist() |> open_brackets(0) == 0

  defp open_brackets([], depth), do: depth

  defp open_brackets([?" | rest], depth),
    do: rest |> quoted([]) |> elem(1) |> open_brackets(depth)

  defp open_brackets([character | rest], depth) when character in ~c"([{",
    do: open_brackets(rest, depth + 1)

  defp open_brackets([character | rest], depth) when character in ~c")]}",
    do: open_brackets(rest, depth - 1)

  defp open_brackets([_character | rest], depth), do: open_brackets(rest, depth)

  defp declaration(line) do
    case Regex.run(~r/^\.decl\s+([\w.]+)\s*\((.*)\)/, line) do
      [_whole, name, columns] ->
        names =
          for column <- split_top(columns, [?,]),
              do: column |> String.split(":") |> hd() |> String.trim()

        {:decl, name, names}

      nil ->
        nil
    end
  end

  defp clause(text) do
    {heads, body} =
      case split_top(text, [":-"]) do
        [heads, body] -> {heads, body}
        [heads] -> {heads, ""}
      end

    {:clause, for({:atom, name, arguments} <- literals_of(heads), do: {name, arguments}),
     literals_of(body)}
  end

  # The atoms of a head or body, negated ones and constraints left out.
  defp literals_of(text) do
    for literal <- split_top(text, [?,, ?;]),
        [_whole, name, arguments] <-
          [Regex.run(~r/^([A-Za-z_$][\w.$]*)\s*\((.*)\)$/s, String.trim(literal))],
        do: {:atom, name, Enum.map(split_top(arguments, [?,]), &String.trim/1)}
  end

  # The text split at each separator outside brackets and strings; `":-"`
  # separates by two characters.
  defp split_top(text, separators) do
    text
    |> String.to_charlist()
    |> split_top(separators, 0, [], [])
    |> Enum.map(&to_string/1)
  end

  defp split_top([], _separators, _depth, part, parts),
    do: Enum.reverse([Enum.reverse(part) | parts])

  defp split_top([?" | rest], separators, depth, part, parts) do
    {quoted, rest} = quoted(rest, [?"])
    split_top(rest, separators, depth, Enum.reverse(quoted) ++ part, parts)
  end

  defp split_top([character | rest], separators, depth, part, parts) when character in ~c"([{",
    do: split_top(rest, separators, depth + 1, [character | part], parts)

  defp split_top([character | rest], separators, depth, part, parts) when character in ~c")]}",
    do: split_top(rest, separators, depth - 1, [character | part], parts)

  defp split_top([?:, ?- | rest], [":-"] = separators, 0, part, parts),
    do: split_top(rest, separators, 0, [], [Enum.reverse(part) | parts])

  defp split_top([character | rest], separators, 0, part, parts) do
    if character in separators,
      do: split_top(rest, separators, 0, [], [Enum.reverse(part) | parts]),
      else: split_top(rest, separators, 0, [character | part], parts)
  end

  defp split_top([character | rest], separators, depth, part, parts),
    do: split_top(rest, separators, depth, [character | part], parts)

  defp quoted([?\\, character | rest], kept), do: quoted(rest, [character, ?\\ | kept])
  defp quoted([?" | rest], kept), do: {Enum.reverse([?" | kept]), rest}
  defp quoted([character | rest], kept), do: quoted(rest, [character | kept])
  defp quoted([], kept), do: {Enum.reverse(kept), []}
end
