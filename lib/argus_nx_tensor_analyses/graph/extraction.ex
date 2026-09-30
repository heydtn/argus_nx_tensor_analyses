defmodule ArgusNxTensorAnalyses.Graph.Extraction do
  @moduledoc false
  # A module's facts: `module_facts(path)`, the rows Argus's base and the
  # analysis's extractors give the beam at `path`, as `Argus.Pipeline.run/3`
  # writes them (each relation's bytes, producer by producer, the base
  # first), kept in the blob store as one segment, and named by each
  # relation's chunk digest.
  #
  # Keyed by the beam's content (`beam`), by the extractors and the code
  # extraction runs (`extraction`), and by this query's code: an edit to an
  # extractor moves that code, and every module is extracted again; an edit
  # to the rules, or to code extraction does not run (the findings'
  # wording), extracts nothing. A module that outlived the pipeline's
  # per-module timeout is its one error row, which depends on the
  # machine's load: it is not kept, nor is anything that read it, and the
  # next run extracts it again.

  use Roux.Query, code: true

  alias Roux.Blob
  alias Roux.Runtime

  defquery :module_facts,
    key: path,
    store: :blob,
    transient: &match?({:ok, %{lost: true}}, &1),
    returns: {:ok, map()} | {:error, term()} do
    _content = Runtime.input(db, :beam, path)
    %{extractors: extractors} = Runtime.input(db, :extraction, :all)

    :telemetry.execute([:argus_nx_tensor_analyses, :graph, :extract], %{}, %{
      path: path,
      store: db.blob.root
    })

    with {:ok, %{status: status, chunks: chunks}} <- extract(path, extractors),
         {:ok, segment} <- put_segment(db.blob, chunks) do
      if segment, do: Runtime.hold(segment)

      {:ok,
       %{
         segment: segment,
         relations: Map.new(chunks, fn {relation, bytes} -> {relation, Blob.digest(bytes)} end),
         lost: status == :lost
       }}
    end
  end

  # The module's rows from the base and each extractor, in that order, each
  # relation's bytes joined as `Argus.Pipeline.run/3` appends them to its
  # file; a relation no producer has rows for is left out.
  defp extract(path, extractors) do
    producers = [:base | Enum.uniq(extractors)]

    with {:ok, extraction} <- Argus.Pipeline.extract_module(path, producers: producers) do
      chunks =
        for producer <- producers,
            {relation, bytes} <- Map.get(extraction.facts, producer, %{}),
            reduce: %{} do
          chunks -> Map.update(chunks, Atom.to_string(relation), bytes, &(&1 <> bytes))
        end

      {:ok, %{status: extraction.status, chunks: chunks}}
    end
  end

  defp put_segment(_store, chunks) when chunks == %{}, do: {:ok, nil}
  defp put_segment(store, chunks), do: Blob.put_term(store, chunks)

  @doc false
  # The chunks of `relations` in the module at `path` whose facts are
  # `facts` (`module_facts`'s value), read from its segment. A segment the
  # store lost (collected, or removed by hand) is made again by extracting
  # the module once more, which the same beam under the same code does
  # byte for byte: `{:error, {:facts_lost, path}}` when it does not.
  @spec chunks(Roux.Database.t(), Path.t(), map(), [String.t()]) ::
          {:ok, %{String.t() => binary()}} | {:error, term()}
  def chunks(_db, _path, %{segment: nil}, _relations), do: {:ok, %{}}

  def chunks(db, path, %{segment: segment}, relations) do
    case Blob.get_term(db.blob, segment) do
      {:ok, chunks} ->
        {:ok, Map.take(chunks, relations)}

      :miss ->
        %{extractors: extractors} =
          Runtime.untracked(fn -> Runtime.input(db, :extraction, :all) end)

        with {:ok, %{chunks: chunks}} <- extract(path, extractors),
             {:ok, ^segment} <- put_segment(db.blob, chunks) do
          {:ok, Map.take(chunks, relations)}
        else
          _other -> {:error, {:facts_lost, path}}
        end
    end
  end
end
