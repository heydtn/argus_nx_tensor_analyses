defmodule ArgusNxTensorAnalyses.Graph.Findings do
  @moduledoc false
  # What the analysis found: `findings(program)`, built by Argus from the
  # solve's rows (`Argus.Findings.build/2`: each relation's rows
  # deduplicated by its key, evidence rows as related frames), or the
  # solve's error. The rows are built category by category
  # (`ArgusNxTensorAnalyses.Categories`), and each finding is reported
  # under its category, its `analysis` and `concern` in place of the
  # engine's name. That needs a relation's key, and an evidence
  # relation's join, to hold its kind: the rows of one finding are then in
  # one category, and a category's rows deduplicate as the whole
  # relation's do. Anchors are instructions, functions and modules, never
  # lines: an edit that only moves lines leaves them as they were, and
  # nothing past them runs.
  #
  # Built by the analysis's code (its output relations, `finding/2` and
  # `evidence/2`), which Argus calls by name: the `analysis` input names it
  # with the digest of that code, so an edit to the analysis's wording
  # builds its findings again, and solves nothing.

  use Roux.Query, code: true

  alias ArgusNxTensorAnalyses.Categories
  alias Roux.Blob
  alias Roux.Runtime

  defquery :findings,
    key: program,
    store: :blob,
    transient: &match?({:error, _reason}, &1),
    returns: {:ok, [Argus.Findings.finding()]} | {:error, term()} do
    with {:ok, digest} <- Runtime.query(db, :solve, program),
         {:ok, rows} <- rows(db.blob, digest) do
      %{module: analysis} = Runtime.input(db, :analysis, :all)

      findings =
        for {category, category_rows} <- Categories.split(analysis, rows),
            finding <- Argus.Findings.build(analysis, category_rows),
            do: %{finding | analysis: category, concern: category}

      {:ok, findings}
    end
  end

  @doc false
  # The rows of a solve's output relations, by name, from the blob store.
  @spec rows(Blob.t(), Blob.digest()) :: {:ok, map()} | {:error, term()}
  def rows(store, digest) do
    case Blob.get_term(store, digest) do
      {:ok, rows} -> {:ok, rows}
      :miss -> {:error, {:rows_missing, digest}}
    end
  end
end
