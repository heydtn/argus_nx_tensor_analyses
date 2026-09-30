defmodule ArgusNxTensorAnalyses.Graph.Locate do
  @moduledoc false
  # Findings placed in the source: `located(program)`, each finding at its
  # call's line and each related frame at its own, as the runner places
  # them (`ArgusNxTensorAnalyses.Solve.place/3`), from the program's
  # `line_info` and each module's beam (its source file, and the line its
  # module is declared on).
  #
  # Findings are line-free, so an edit that only moves lines runs this
  # alone: `line_info` moves, and every finding is placed again. So does
  # any edit to a beam, which may move where its module's source is.

  use Roux.Query, code: true

  alias ArgusNxTensorAnalyses.Graph.Relations
  alias ArgusNxTensorAnalyses.Solve
  alias Roux.Blob
  alias Roux.Runtime

  defquery :located,
    key: program,
    store: :blob,
    transient: &match?({:error, _reason}, &1),
    returns: {:ok, [Argus.Located.t()]} | {:error, term()} do
    paths = Runtime.input(db, :program, program)
    Enum.each(paths, &Runtime.input(db, :beam, &1))

    with {:ok, findings} <- Runtime.query(db, :findings, program),
         {:ok, lines} <- lines(db, program) do
      beams = Solve.beams(paths)
      {:ok, Enum.map(findings, &Solve.place(&1, beams, lines))}
    end
  end

  # Each instruction's line and each function's first (`Argus.Lines`), from
  # the program's `line_info`.
  defp lines(db, program) do
    with {:ok, digest} <- Runtime.query(db, :relation, {program, "line_info"}),
         {:ok, %{"line_info" => file}} <- Relations.files(db, program, [{"line_info", digest}]),
         {:ok, content} <- Blob.get(db.blob, file) do
      {:ok, Argus.Lines.from_facts(%{line_info: Argus.Tsv.decode(content)})}
    else
      :miss -> {:error, {:relation_missing, "line_info"}}
      {:error, _reason} = error -> error
    end
  end
end
