defmodule ArgusNxTensorAnalyses.Graph.Relations do
  @moduledoc false
  # The program's relations, each named by a digest, and made into a file
  # only when a solve needs it.
  #
  #   * `program_relations(program)` — each relation any of the program's
  #     modules has rows for, and a digest of it: of its modules' chunk
  #     digests, in the program's order. The modules are extracted, or
  #     found up to date, side by side.
  #   * `relation({program, relation})` — one relation's digest: an edit
  #     that leaves a relation's rows as they were leaves it equal, and
  #     nothing that reads only such relations runs again.
  #
  # A relation's file is its modules' chunks joined in the program's
  # order (`files/3`): assembled in one pass over the segments of the
  # modules that hold it, put into the blob store, and remembered under
  # the relation's digest, so a later solve that reads it links it.

  use Roux.Query, code: true

  alias ArgusNxTensorAnalyses.Graph.Extraction
  alias Roux.Blob
  alias Roux.Runtime

  # The digest of a relation no module has rows for: an empty file.
  @empty "empty"

  defquery :program_relations,
    key: program,
    returns: {:ok, %{optional(String.t()) => String.t()}} | {:error, term()} do
    keys = Runtime.input(db, :program, program)

    facts =
      Runtime.parallel(db, Enum.map(keys, &{:module_facts, &1}),
        max_concurrency: System.schedulers_online()
      )

    # A beam the pipeline cannot read fails the whole program, the first
    # one in its order, as it fails `Argus.Pipeline.run/3`.
    case Enum.find(facts, &match?({:error, _reason}, &1)) do
      nil ->
        {:ok,
         facts
         |> Enum.flat_map(fn {:ok, %{relations: relations}} -> Enum.to_list(relations) end)
         |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
         |> Map.new(fn {relation, digests} -> {relation, merkle(digests)} end)}

      error ->
        error
    end
  end

  defquery :relation,
    key: {program, relation},
    returns: {:ok, String.t()} | {:error, term()} do
    with {:ok, digests} <- Runtime.query(db, :program_relations, program),
         do: {:ok, Map.get(digests, relation, @empty)}
  end

  @doc false
  # The digest of a relation no module has rows for.
  @spec empty() :: String.t()
  def empty, do: @empty

  defp merkle(digests) do
    digests
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
    |> then(&"modules:#{length(digests)}:#{&1}")
  end

  @doc false
  # The blob store entries holding the files of `relations` (each with the
  # digest `relation` gave it) for `program`: each relation's name to its
  # file's digest. Remembered by the relation's digest; the ones not
  # remembered are assembled together, in one pass over the program's
  # modules. Reads the graph without edges: its caller depends on each
  # relation's digest already.
  @spec files(Roux.Database.t(), term(), [{String.t(), String.t()}]) ::
          {:ok, %{String.t() => Blob.digest()}} | {:error, term()}
  def files(db, program, relations) do
    store = db.blob

    {known, missing} =
      Enum.reduce(relations, {%{}, []}, fn {relation, digest}, {known, missing} ->
        case remembered(store, digest) do
          {:ok, file} -> {Map.put(known, relation, file), missing}
          _miss -> {known, [{relation, digest} | missing]}
        end
      end)

    with {:ok, made} <- assemble(db, program, Enum.reverse(missing)) do
      for {relation, digest} <- missing,
          do: _ = Blob.remember(store, remember_key(digest), Map.fetch!(made, relation))

      {:ok, Map.merge(known, made)}
    end
  end

  defp remembered(store, @empty), do: Blob.put(store, "")

  defp remembered(store, digest) do
    with {:ok, file} <- Blob.recall(store, remember_key(digest)),
         true <- Blob.member?(store, file) do
      {:ok, file}
    else
      _missing -> :miss
    end
  end

  defp remember_key(digest), do: {__MODULE__, :file, digest}

  defp assemble(_db, _program, []), do: {:ok, %{}}

  # Each module's chunks of the missing relations, appended to their files
  # in a scratch directory on the store's file system, then moved into it.
  defp assemble(db, program, missing) do
    store = db.blob
    relations = Enum.map(missing, &elem(&1, 0))
    wanted = MapSet.new(relations)
    keys = Runtime.untracked(fn -> Runtime.input(db, :program, program) end)

    Blob.scratch(store, fn directory ->
      devices =
        Map.new(Enum.with_index(relations), fn {relation, index} ->
          path = Path.join(directory, Integer.to_string(index))
          {:ok, device} = File.open(path, [:write, :raw, :binary, :delayed_write])
          {relation, {path, device}}
        end)

      written =
        try do
          Enum.reduce_while(keys, :ok, fn key, :ok ->
            case Runtime.untracked(fn -> Runtime.query(db, :module_facts, key) end) do
              {:ok, %{relations: held} = facts} ->
                if Enum.any?(Map.keys(held), &MapSet.member?(wanted, &1)),
                  do: append(db, key, facts, relations, devices),
                  else: {:cont, :ok}

              {:error, _reason} = error ->
                {:halt, error}
            end
          end)
        after
          Enum.each(devices, fn {_relation, {_path, device}} -> File.close(device) end)
        end

      with :ok <- written do
        Enum.reduce_while(devices, {:ok, %{}}, fn {relation, {path, _device}}, {:ok, made} ->
          case Blob.adopt(store, path) do
            {:ok, file} -> {:cont, {:ok, Map.put(made, relation, file)}}
            {:error, reason} -> {:halt, {:error, {:relation_write_failed, relation, reason}}}
          end
        end)
      end
    end)
  end

  defp append(db, key, facts, relations, devices) do
    case Extraction.chunks(db, key, facts, relations) do
      {:ok, chunks} ->
        for {relation, bytes} <- chunks do
          {_path, device} = Map.fetch!(devices, relation)
          :ok = IO.binwrite(device, bytes)
        end

        {:cont, :ok}

      {:error, _reason} = error ->
        {:halt, error}
    end
  end
end
