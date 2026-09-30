defmodule ArgusNxTensorAnalyses.Graph do
  @moduledoc false
  # The analyses as a roux query graph: extraction, stage 0 and the solve,
  # each a memoized query keyed by what it read, so a run computes again
  # only what an edit reached.
  #
  #     program(:beams) ─ beam(path) ─ extraction(:all)   [Graph.Inputs]
  #          │
  #     module_facts(path)          one segment per module in the blob
  #          │                      store
  #     program_relations(:beams)   a digest per relation, over the
  #          │                      program's modules as one fan-out
  #     relation({:beams, r})       ── cutoff: per relation
  #          │
  #     stage0(:beams) ─ stage0_output({:beams, r})   ── cutoff
  #          │                      stage0_rules(:all), solver(:all)
  #     solve(:beams)               rules(:all), options(:all): one
  #                                 Souffle solve, kept by its key
  #
  # What a query is keyed by: the inputs it reads, the queries it demands,
  # and the code it runs. Each query module's code version is the digest
  # of the code it reaches (`use Roux.Query, code: true`); the code the
  # graph reaches only by name, the analysis's extractors, is an input
  # (`extraction`), versioned the same way (`Roux.Code`). A solve reads
  # the relations its program loads, and not `line_info`, which moves with
  # every comment in the code analyzed and which only placement reads.

  alias ArgusNxTensorAnalyses.Graph
  alias Roux.Blob
  alias Roux.Input
  alias Roux.Runtime
  alias Roux.Session

  @modules [Graph.Inputs, Graph.Extraction, Graph.Relations, Graph.Solve]

  # The program's id: a session holds one set of beams.
  @program :beams

  @doc false
  # Runs `demand` over the graph, set for the analysis's extractors over
  # `modules` (atoms or `.beam` paths) and the program made of `roots`
  # (the files a solve includes, in order), with `options` as the
  # relations the analysis's options fill: in a session kept under `cache`,
  # or in one that keeps nothing.
  @spec run(
          module(),
          [module() | Path.t()],
          [Path.t()],
          %{String.t() => String.t()},
          Path.t() | nil,
          (Roux.Database.t() -> result)
        ) :: result | {:error, term()}
        when result: var
  def run(analysis, modules, roots, options, cache, demand) do
    with {:ok, paths} <- Argus.Pipeline.Disassemble.resolve_paths(modules) do
      session = open(cache)

      try do
        paths = paths |> Enum.map(&Path.expand/1) |> Enum.uniq()
        sources = set_program(session.db, paths, analysis.extractors(), session.sources)
        set_rules(session.db, roots, options)
        result = demand.(session.db)
        commit(session, sources)
        result
      after
        Session.close(session)
      end
    end
  end

  # Opens a session over the graph: kept under `cache` (its manifest, and
  # the blob store its facts and solves are kept in), or, for nil, in a
  # blob store of its own that closing it removes, which keeps nothing.
  defp open(nil), do: Session.open(modules: @modules, blob: Blob.temporary())

  defp open(cache) do
    File.mkdir_p!(cache)

    Session.open(
      modules: @modules,
      manifest: Path.join(cache, "manifest"),
      blob: Path.join(cache, "store")
    )
  end

  # Keeps what the session's run computed (`Roux.Session.commit/3`), with
  # `sources` as its beams' metadata. A kept store is collected at most
  # once a day (`Roux.Blob.maybe_gc/2`): what the manifest names stays, and
  # so does what a recent run used.
  defp commit(%Session{manifest: nil}, _sources), do: :ok

  defp commit(session, sources) do
    {_status, _session} = Session.commit(session, sources)
    _collected = Blob.maybe_gc(session.blob)
    :ok
  end

  # Sets what the graph extracts: the beams at `paths` (absolute, in the
  # order their rows are written) and the analysis's extractors, with the
  # digest of the code extraction runs. `sources` is the beams' metadata
  # the session's last run left (`Roux.Sources`); returns the metadata to
  # commit.
  defp set_program(db, paths, extractors, sources) do
    %{meta: meta} =
      Roux.Sources.sync(db, :beam, Map.new(paths, &{&1, &1}), sources,
        hash: &Blob.digest/1,
        value: fn %{hash: hash} -> %{hash: hash} end
      )

    :ok = Input.set(db, :program, @program, paths)

    :ok =
      Input.set(db, :extraction, :all, %{
        extractors: extractors,
        code: extraction_code(extractors, db.blob),
        relations: extracted_relations(extractors)
      })

    meta
  end

  # The code extraction runs, by digest: `Argus.Pipeline` and the
  # extractors, which the pipeline calls by name, and Argus's schema, whose
  # relations it reads by name. A module compiled in memory has no object
  # code to read, and is named for this VM alone.
  defp extraction_code(extractors, store) do
    roots = [Argus.Pipeline | extractors] ++ schema_modules()

    case Roux.Code.digest(roots, store: store) do
      {:ok, digest} -> digest
      {:error, reason} -> {:unversioned, reason, vm_token()}
    end
  end

  defp schema_modules do
    for module <- Application.spec(:argus_beam, :modules),
        String.starts_with?(Atom.to_string(module), "Elixir.Argus.Schema."),
        do: module
  end

  defp vm_token do
    key = {__MODULE__, :vm_token}

    case :persistent_term.get(key, nil) do
      nil ->
        token = :crypto.strong_rand_bytes(16)
        :persistent_term.put(key, token)
        token

      token ->
        token
    end
  end

  # Every relation extraction gives a file, rows or not: Argus's schema
  # and each extractor's.
  defp extracted_relations(extractors) do
    (Argus.Schema.names() ++ Enum.flat_map(extractors, & &1.relations()))
    |> Enum.map(&to_string/1)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # Sets what the graph solves: the program and stage 0 as a solve reads
  # them, the solver, and the options' relations.
  defp set_rules(db, roots, options) do
    :ok = Input.set(db, :rules, :all, Graph.Program.read(roots))

    :ok =
      Input.set(db, :stage0_rules, :all, Graph.Program.read([Argus.Analysis.stage0_rules_path()]))

    :ok = Input.set(db, :solver, :all, solver(db.blob))
    :ok = Input.set(db, :options, :all, options)
  end

  # The souffle on `PATH`, by its version, as Argus names it in its own
  # keys, and by the file it runs: a build can print an empty version
  # (`Version: `), and its file still tells it from another. Kept while
  # the file's stamp holds (`Roux.Stamp`).
  defp solver(store) do
    case Argus.Souffle.executable() do
      nil ->
        nil

      path ->
        Roux.Stamp.memo(
          {__MODULE__, :solver, path},
          [path],
          fn ->
            %{
              path: path,
              version: Argus.Souffle.version(path),
              digest: path |> File.read!() |> Blob.digest()
            }
          end,
          store: store
        )
    end
  end

  @doc false
  # The rows of the program's output relations, by name.
  @spec rows(Roux.Database.t()) :: {:ok, map()} | {:error, term()}
  def rows(db) do
    with {:ok, digest} <- Runtime.query(db, :solve, @program) do
      case Blob.get_term(db.blob, digest) do
        {:ok, rows} -> {:ok, rows}
        :miss -> {:error, {:rows_missing, digest}}
      end
    end
  end

  @doc false
  # Each instruction's line and each function's first (`Argus.Lines`), from
  # the program's `line_info`.
  @spec lines(Roux.Database.t()) :: {:ok, Argus.Lines.t()} | {:error, term()}
  def lines(db) do
    with {:ok, digest} <- Runtime.query(db, :relation, {@program, "line_info"}),
         {:ok, %{"line_info" => file}} <-
           Graph.Relations.files(db, @program, [{"line_info", digest}]),
         {:ok, content} <- Blob.get(db.blob, file) do
      {:ok, Argus.Lines.from_facts(%{line_info: Argus.Tsv.decode(content)})}
    else
      :miss -> {:error, {:relation_missing, "line_info"}}
      {:error, _reason} = error -> error
    end
  end
end
