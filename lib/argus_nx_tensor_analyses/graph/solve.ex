defmodule ArgusNxTensorAnalyses.Graph.Solve do
  @moduledoc false
  # The solves: Argus's stage 0, and the analysis's program.
  #
  #   * `stage0(program)` — the call graph (`Argus.Analysis.derive_stage0/2`)
  #     over the relations `stage0.dl` reads: each output's file in the
  #     blob store, by name.
  #   * `stage0_output({program, relation})` — one output's digest: an
  #     edit that moves instructions but no call leaves the call graph as
  #     it was, and a solve that reads only it and other relations that
  #     did not move is valid without running.
  #   * `solve(program)` — the analysis's program (`rules`), after the
  #     Argus files it builds on, solved over the relations it reads
  #     (`ArgusNxTensorAnalyses.Graph.Program`): the digest of its output
  #     rows in the blob store. Stage 0 is demanded only for the
  #     relations of it the program reads, and the solve told it is
  #     provided: left to itself, `Argus.Analysis.run_rules/3` would also
  #     learn whether the program reads Argus's process points-to, which
  #     it never does, by compiling the program.
  #
  # Each solve is kept in the blob store's action cache under its key:
  # the code that runs it (the query's code version), the solver (its
  # version and its file: a build can print an empty version), the
  # program's files without their comments, and each relation it reads by
  # digest. A solve whose key is kept is read back, and nothing is written
  # for it; one that is not runs in a scratch directory of its own, where
  # only the relations it reads are placed, as links to their files.
  #
  # A solve that fails (the solver's error, a program that cannot be
  # read) is a value, `{:error, reason}`, and a transient one: it is not
  # kept, and neither is anything that read it, so the next run solves
  # again.

  use Roux.Query, code: true

  alias ArgusNxTensorAnalyses.Graph.Relations
  alias Roux.Blob
  alias Roux.Runtime

  @stage0_relations Argus.Analysis.stage0_relations()

  defquery :stage0,
    key: program,
    transient: &match?({:error, _reason}, &1),
    returns: {:ok, %{String.t() => Blob.digest()}} | {:error, term()} do
    rules = Runtime.input(db, :stage0_rules, :all)

    solved(db, program, :stage0, rules, fn facts, _directory ->
      with :ok <- Argus.Analysis.derive_stage0(facts) do
        Enum.reduce_while(@stage0_relations, {:ok, %{}}, fn relation, {:ok, outputs} ->
          case Blob.adopt(db.blob, Path.join(facts, relation <> ".facts")) do
            {:ok, digest} -> {:cont, {:ok, Map.put(outputs, relation, digest)}}
            {:error, reason} -> {:halt, {:error, {:stage0, {:missing_output, relation, reason}}}}
          end
        end)
      end
    end)
  end

  defquery :stage0_output,
    key: {program, relation},
    returns: {:ok, Blob.digest()} | {:error, term()} do
    case Runtime.query(db, :stage0, program) do
      {:ok, %{^relation => digest}} -> {:ok, digest}
      {:ok, _outputs} -> {:error, {:stage0, {:missing_output, relation}}}
      {:error, _reason} = error -> error
    end
  end

  defquery :solve,
    key: program,
    transient: &match?({:error, _reason}, &1),
    returns: {:ok, Blob.digest()} | {:error, term()} do
    rules = Runtime.input(db, :rules, :all)

    solved(db, program, :rules, rules, fn facts, directory ->
      wrapper = Path.join(directory, "program.dl")
      File.write!(wrapper, Enum.map(rules.roots, &~s(.include "#{&1}"\n)))

      with {:ok, rows} <- Argus.Analysis.run_rules(facts, {:custom, wrapper}, stage0: :provided),
           do: Blob.put_term(db.blob, rows)
    end)
  end

  # A solve of `rules` over the relations it reads, kept by its key, or
  # run by `solve` over a directory of them (`solve.(facts, directory)`,
  # the facts directory inside the solve's own): `{:ok, value}`, whose
  # digests the entry holds.
  defp solved(db, program, stage, rules, solve) do
    with {:ok, inputs} <- inputs(db, program, stage, rules) do
      solver = Runtime.input(db, :solver, :all)
      key = key(stage, rules, solver, inputs)

      result =
        case key && recall(db.blob, key) do
          {:ok, value} -> {:ok, value}
          _missing -> solve_and_keep(db, program, stage, key, inputs, solve)
        end

      with {:ok, value} <- result do
        Runtime.hold(digests(value))
        {:ok, value}
      end
    end
  end

  # What a solve is kept under, or nil for one that is never kept: without
  # a solver, or with a program that cannot be read, it only fails.
  defp key(_stage, _rules, nil, _inputs), do: nil
  defp key(_stage, %{files: {:unreadable, _path}}, _solver, _inputs), do: nil

  defp key(stage, rules, solver, inputs) do
    {__MODULE__, stage, Runtime.code_version(), {solver.version, solver.digest}, rules.files,
     Enum.map(inputs, fn {relation, source} -> {relation, identity(source)} end)}
  end

  defp identity({:text, text}), do: {:text, :crypto.hash(:sha256, text)}
  defp identity(source), do: source

  # A kept solve whose every blob is still in the store; one that lost an
  # entry to a collection is solved again.
  defp recall(store, key) do
    with {:ok, value} <- Blob.recall(store, key),
         true <- Enum.all?(digests(value), &Blob.member?(store, &1)) do
      {:ok, value}
    else
      _missing -> :miss
    end
  end

  defp digests(outputs) when is_map(outputs), do: Map.values(outputs)
  defp digests(rows) when is_binary(rows), do: [rows]

  defp solve_and_keep(db, program, stage, key, inputs, solve) do
    :telemetry.execute([:argus_nx_tensor_analyses, :graph, :solve], %{}, %{
      stage: stage,
      store: db.blob.root
    })

    result =
      Blob.scratch(db.blob, fn directory ->
        facts = Path.join(directory, "facts")
        File.mkdir!(facts)
        with :ok <- place(db, program, inputs, facts), do: solve.(facts, directory)
      end)

    with {:ok, value} <- result do
      if key, do: _ = Blob.remember(db.blob, key, value)
      {:ok, value}
    end
  end

  # The relations a solve at `stage` reads, each with where its file comes
  # from: `{:relation, digest}` for the facts extracted, `{:stage0,
  # digest}` for stage 0's outputs, `{:text, text}` for the options'
  # relations, or `:absent` for a relation nothing gives a file, which the
  # solver then refuses to load.
  defp inputs(db, program, stage, rules) do
    extraction = Runtime.input(db, :extraction, :all)
    options = if stage == :rules, do: Runtime.input(db, :options, :all), else: %{}
    staged = if stage == :rules, do: @stage0_relations, else: []

    relations =
      case rules.reads do
        :all -> every_relation(db, program, extraction, options, staged)
        reads -> reads
      end

    Enum.reduce_while(relations, {:ok, []}, fn relation, {:ok, inputs} ->
      case source(db, program, relation, extraction, options, staged) do
        {:error, _reason} = error -> {:halt, error}
        source -> {:cont, {:ok, [{relation, source} | inputs]}}
      end
    end)
    |> case do
      {:ok, inputs} -> {:ok, Enum.reverse(inputs)}
      error -> error
    end
  end

  defp every_relation(db, program, extraction, options, staged) do
    extracted =
      case Runtime.query(db, :program_relations, program) do
        {:ok, digests} -> Map.keys(digests)
        {:error, _reason} -> []
      end

    (extraction.relations ++ extracted ++ Map.keys(options) ++ staged)
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp source(db, program, relation, extraction, options, staged) do
    cond do
      relation in staged ->
        with {:ok, digest} <- Runtime.query(db, :stage0_output, {program, relation}),
             do: {:stage0, digest}

      Map.has_key?(options, relation) ->
        {:text, Map.fetch!(options, relation)}

      true ->
        with {:ok, digest} <- Runtime.query(db, :relation, {program, relation}) do
          # A relation no module has rows for has a file when extraction
          # gives it one.
          if digest == Relations.empty() and relation not in extraction.relations,
            do: :absent,
            else: {:relation, digest}
        end
    end
  end

  # Each input's file in `facts`: a link to its entry in the store, or its
  # text written.
  defp place(db, program, inputs, facts) do
    extracted = for {relation, {:relation, digest}} <- inputs, do: {relation, digest}

    with {:ok, files} <- Relations.files(db, program, extracted) do
      Enum.reduce_while(inputs, :ok, fn {relation, source}, :ok ->
        path = Path.join(facts, relation <> ".facts")

        placed =
          case source do
            {:relation, _digest} -> Blob.link(db.blob, Map.fetch!(files, relation), path)
            {:stage0, digest} -> Blob.link(db.blob, digest, path)
            {:text, text} -> File.write(path, text)
            :absent -> :ok
          end

        case placed do
          :ok -> {:cont, :ok}
          {:error, reason} -> {:halt, {:error, {:input_failed, relation, reason}}}
        end
      end)
    end
  end
end
