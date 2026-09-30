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
  #   * `rules` (`:all`) — the program the analysis solves, after the Argus
  #     files it builds on (`ArgusNxTensorAnalyses.Solve.argus_includes/0`),
  #     as a solve reads it (`ArgusNxTensorAnalyses.Graph.Program`): its
  #     files by their text without comments, and the relations it loads.
  #   * `stage0_rules` (`:all`) — Argus's `stage0.dl`, likewise.
  #   * `solver` (`:all`) — `%{path: executable, version: banner, digest:
  #     digest}`, the souffle on `PATH` by its version and the SHA-256 of
  #     its file, or nil without one.
  #   * `options` (`:all`) — the relations the analysis's options fill
  #     (`unsupported_type`, `float_type`), each by its file's text.
  #   * `analysis` (`:all`) — `%{module: analysis, code: digest}`: the
  #     `Argus.Analysis` whose findings are built from the rows, and the
  #     digest of its code, which Argus calls by name.
  #
  # Durability: `program` and `beam` move with the code analyzed and are
  # `:medium`; the rest move with the toolchain or the analysis's
  # configuration and are `:high`, so an edit to the code analyzed never
  # walks what only they reach.

  use Roux.Query

  definput(:program, durability: :medium)
  definput(:beam, durability: :medium)
  definput(:extraction, durability: :high)
  definput(:rules, durability: :high)
  definput(:stage0_rules, durability: :high)
  definput(:solver, durability: :high)
  definput(:options, durability: :high)
  definput(:analysis, durability: :high)
end
