defmodule ArgusNxTensorAnalyses.Graph do
  @moduledoc false
  # The analyses as a roux query graph: each module's facts a memoized
  # query keyed by what it read, so a run computes again only what an edit
  # reached.
  #
  #     program(:beams) ─ beam(path) ─ extraction(:all)   [Graph.Inputs]
  #          │
  #     module_facts(path)          one segment per module in the blob
  #          │                      store
  #     program_relations(:beams)   a digest per relation, over the
  #          │                      program's modules as one fan-out
  #     relation({:beams, r})       ── cutoff: per relation
  #
  # What a query is keyed by: the inputs it reads, the queries it demands,
  # and the code it runs. Each query module's code version is the digest
  # of the code it reaches (`use Roux.Query, code: true`); the code the
  # graph reaches only by name, the analysis's extractors, is an input
  # (`extraction`), versioned the same way (`Roux.Code`).

  alias ArgusNxTensorAnalyses.Graph
  alias Roux.Blob
  alias Roux.Input
  alias Roux.Runtime
  alias Roux.Session

  @modules [Graph.Inputs, Graph.Extraction, Graph.Relations]

  # The program's id: a session holds one set of beams.
  @program :beams

  @doc false
  # Opens a session over the graph: kept under `cache` (its manifest, and
  # the blob store its facts and solves are kept in), or, for nil, in a
  # blob store of its own that closing it removes, which keeps nothing.
  @spec open(Path.t() | nil) :: Session.t()
  def open(nil), do: Session.open(modules: @modules, blob: Blob.temporary())

  def open(cache) do
    File.mkdir_p!(cache)

    Session.open(
      modules: @modules,
      manifest: Path.join(cache, "manifest"),
      blob: Path.join(cache, "store")
    )
  end

  @doc false
  # Keeps what the session's run computed (`Roux.Session.commit/3`), with
  # `sources` as its beams' metadata. A kept store is collected at most
  # once a day (`Roux.Blob.maybe_gc/2`): what the manifest names stays, and
  # so does what a recent run used.
  @spec commit(Session.t(), map()) :: :ok
  def commit(%Session{manifest: nil}, _sources), do: :ok

  def commit(session, sources) do
    {_status, _session} = Session.commit(session, sources)
    _collected = Blob.maybe_gc(session.blob)
    :ok
  end

  @doc false
  # Sets what the graph extracts: the beams at `paths` (absolute, in the
  # order their rows are written) and the analysis's extractors, with the
  # digest of the code extraction runs. `sources` is the beams' metadata
  # the session's last run left (`Roux.Sources`); returns the metadata to
  # commit.
  @spec set_program(Roux.Database.t(), [Path.t()], [module()], map()) :: map()
  def set_program(db, paths, extractors, sources) do
    %{meta: meta} =
      Roux.Sources.sync(db, :beam, Map.new(paths, &{&1, &1}), sources,
        hash: &hash/1,
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

  defp hash(bytes), do: :sha256 |> :crypto.hash(bytes) |> Base.encode16(case: :lower)

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

  @doc false
  # Writes every relation's file into `directory`, as
  # `Argus.Pipeline.run/3` writes them: one per relation extraction gives a
  # file, and one per other relation a module has rows for.
  @spec write_facts(Roux.Database.t(), Path.t()) :: :ok | {:error, term()}
  def write_facts(db, directory) do
    %{relations: extracted} = Input.get(db, :extraction, :all)

    with {:ok, digests} <- Runtime.query(db, :program_relations, @program),
         relations = Enum.uniq(extracted ++ Map.keys(digests)),
         named = Enum.map(relations, &{&1, relation_digest(db, &1)}),
         {:ok, files} <- Graph.Relations.files(db, @program, named) do
      Enum.reduce_while(files, :ok, fn {relation, file}, :ok ->
        case Blob.link(db.blob, file, Path.join(directory, relation <> ".facts")) do
          :ok -> {:cont, :ok}
          {:error, reason} -> {:halt, {:error, {:write_failed, relation, reason}}}
        end
      end)
    end
  end

  defp relation_digest(db, relation) do
    {:ok, digest} = Runtime.query(db, :relation, {@program, relation})
    digest
  end
end
