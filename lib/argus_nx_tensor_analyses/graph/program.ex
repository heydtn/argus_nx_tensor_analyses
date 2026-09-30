defmodule ArgusNxTensorAnalyses.Graph.Program do
  @moduledoc false
  # A Datalog program as a solve reads it: the value of the graph's
  # `rules` and `stage0_rules` inputs.
  #
  # A program is its roots, the files a solve includes in order, each with
  # every file it includes (`Argus.Souffle.Program.program_files/1`, which
  # resolves an include as Souffle does). Each file is named as it is
  # included and by a digest of its text without comments
  # (`Argus.Souffle.Program.uncommented/1`): a comment is not an edit.
  #
  # And the relations a solve of it loads: every relation it declares an
  # input (`.input name`) and names anywhere outside that declaration.
  # Souffle loads no input no rule reads, so an input named nowhere else
  # decides nothing a solve writes, and its rows are no part of what a
  # solve is keyed on: a file of declarations declares every relation of
  # the facts, `line_info` among them, which moves with every comment in
  # the code analyzed, and a program reads a few. Naming is read as text,
  # a whole word, so a name in a string or a field counts too: a solve may
  # be keyed on more than it loads, never on less. Where the text does not
  # say plainly — an input declared with parameters or several to a line,
  # one inside a component, a macro that could spell a name — every
  # relation counts. Souffle's own answer
  # (`Argus.Souffle.input_relations/2`) is exact, but it compiles the
  # whole program to say, which is most of what solving it costs, and a
  # run would pay it before it could tell whether it needs to solve at
  # all.

  alias Argus.Souffle.Program
  alias Roux.Blob

  @type t :: %{
          roots: [Path.t()],
          files: [[{String.t(), binary()}]] | {:unreadable, term()},
          reads: [String.t()] | :all
        }

  @doc false
  # The program made of `roots`. A file that cannot be read leaves the
  # program unreadable (`files: {:unreadable, reason}`), which a solve then
  # reports as Souffle does.
  @spec read([Path.t()]) :: t()
  def read(roots) do
    trees =
      for root <- roots do
        for {spelled, path} <- Program.program_files(root),
            do: {spelled, path |> File.read!() |> Program.uncommented()}
      end

    files = for tree <- trees, do: for({spelled, text} <- tree, do: {spelled, Blob.digest(text)})

    %{
      roots: roots,
      files: files,
      reads: trees |> Enum.concat() |> Enum.map(&elem(&1, 1)) |> reads()
    }
  rescue
    error in File.Error -> %{roots: roots, files: {:unreadable, error.path}, reads: :all}
  end

  # The relations the texts declare an input and name outside the
  # declaration, sorted, or `:all`.
  defp reads(texts) do
    parsed = Enum.map(texts, &inputs/1)

    if Enum.any?(parsed, &(&1 == :all)) do
      :all
    else
      declared = parsed |> Enum.flat_map(&elem(&1, 0)) |> Enum.uniq()
      named(declared, Enum.map(parsed, &elem(&1, 1)))
    end
  end

  # A text's declared inputs and the text left once each declaration's
  # name is taken out, or `:all`.
  defp inputs(text) do
    text
    |> String.split("\n")
    |> Enum.reduce_while({[], [], false}, fn line, {declared, kept, component?} ->
      trimmed = String.trim_leading(line)

      cond do
        String.starts_with?(trimmed, "#define") ->
          {:halt, :all}

        String.starts_with?(trimmed, ".input") ->
          case Regex.run(~r/^\.input\s+([A-Za-z_][A-Za-z0-9_]*)\s*$/, trimmed) do
            [_line, relation] -> {:cont, {[relation | declared], kept, component?}}
            nil -> {:halt, :all}
          end

        # A `.decl` line without the name it declares: its fields and
        # whatever follows them are still read.
        String.starts_with?(trimmed, ".decl") ->
          stripped = String.replace(trimmed, ~r/^\.decl\s+[A-Za-z_][A-Za-z0-9_]*/, "")
          {:cont, {declared, [stripped | kept], component?}}

        true ->
          component? =
            component? or (String.starts_with?(trimmed, ".comp") and trimmed =~ ~r/^\.comp\s/)

          {:cont, {declared, [line | kept], component?}}
      end
    end)
    |> case do
      :all -> :all
      # An input declared inside a component is the instance's, by another name.
      {[_input | _inputs], _kept, true} -> :all
      {declared, kept, _component?} -> {declared, kept |> Enum.reverse() |> Enum.join("\n")}
    end
  end

  defp named([], _texts), do: []

  defp named(declared, texts) do
    pattern = :binary.compile_pattern(declared)

    for text <- texts,
        {start, length} <- :binary.matches(text, pattern),
        word?(text, start, length),
        uniq: true,
        do: binary_part(text, start, length)
  end

  # Whether the name at `start` is a whole word: no letter, digit or
  # underscore just before or after it. A name inside a longer one is not
  # the relation.
  defp word?(text, start, length) do
    not word_byte?(text, start - 1) and not word_byte?(text, start + length)
  end

  defp word_byte?(text, at) when at < 0 or at >= byte_size(text), do: false

  defp word_byte?(text, at) do
    byte = :binary.at(text, at)
    byte == ?_ or byte in ?a..?z or byte in ?A..?Z or byte in ?0..?9
  end
end
