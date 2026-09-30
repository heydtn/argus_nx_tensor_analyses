defmodule ArgusNxTensorAnalyses.Graph.Inputs do
  @moduledoc false
  # What the query graph (`ArgusNxTensorAnalyses.Graph`) is a function of:
  # the inputs a run sets before it demands anything, and nothing a query
  # computes.
  #
  #   * `program` (`:beams`) — the beams to extract, as the absolute paths
  #     of their files, in the order the caller gave them: the order of
  #     every relation's rows.
  #   * `beam` (a beam's path) — `%{hash: digest}`, the SHA-256 of the
  #     file's bytes.
  #   * `extraction` (`:all`) — `%{extractors: modules, code: digest,
  #     relations: names}`: the extractors that run after Argus's base
  #     over every module, the digest of the code extraction runs
  #     (`Roux.Code`: `Argus.Pipeline`, the extractors, which the pipeline
  #     reaches only by name, and Argus's schema, which it reads by name),
  #     and every relation extraction gives a file, rows or not.
  #
  # Durability: `program` and `beam` move with the code analyzed and are
  # `:medium`; the rest move with the toolchain and are `:high`, so an
  # edit to the code analyzed never walks what only they reach.

  use Roux.Query

  definput(:program, durability: :medium)
  definput(:beam, durability: :medium)
  definput(:extraction, durability: :high)
end
