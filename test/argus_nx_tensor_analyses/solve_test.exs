defmodule ArgusNxTensorAnalyses.SolveTest do
  # Not async: a test puts another souffle first on `PATH`, and another
  # compiles a new version of an extractor, and both are the whole VM's.
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO

  alias ArgusNxTensorAnalyses.Solve
  alias ArgusNxTensorAnalyses.TensorShapes

  # An analysis whose extractor a test compiles, and compiles again.
  defmodule Analysis do
    def extractors, do: [ArgusNxTensorAnalyses.SolveTest.Extractor]
  end

  test "the Argus files a program is solved after exist" do
    for path <- Solve.argus_includes(), do: assert(File.regular?(path), path)
  end

  # `priv/argus.dl` copies the Argus words the rules read beyond the files
  # a program is solved after. Each must still be Argus's own, so that an
  # Argus release that changes one fails here rather than solving on.
  test "the Argus words priv/argus.dl copies are Argus's, word for word" do
    argus =
      :argus_beam
      |> Application.app_dir("priv/dl/**/*.dl")
      |> Path.wildcard()
      |> Enum.map_join(" ", &normalized(File.read!(&1)))

    copied =
      :argus_nx_tensor_analyses
      |> Application.app_dir("priv/argus.dl")
      |> File.read!()
      |> statements()

    assert copied != []

    for statement <- copied do
      assert String.contains?(argus, statement), "not in Argus: #{statement}"
    end
  end

  # A program's statements, each with its comments dropped and its
  # whitespace collapsed: a statement starts on a line with no indent.
  defp statements(text) do
    text
    |> uncommented()
    |> String.split("\n")
    |> Enum.chunk_while(
      [],
      fn line, statement ->
        cond do
          String.trim(line) == "" ->
            {:cont, statement}

          statement != [] and not String.starts_with?(line, [" ", "\t"]) ->
            {:cont, Enum.reverse(statement), [line]}

          true ->
            {:cont, [line | statement]}
        end
      end,
      fn
        [] -> {:cont, []}
        statement -> {:cont, Enum.reverse(statement), []}
      end
    )
    |> Enum.map(&normalized(Enum.join(&1, " ")))
  end

  defp normalized(text) do
    text
    |> uncommented()
    |> String.split()
    |> Enum.join(" ")
  end

  defp uncommented(text), do: String.replace(text, ~r{//[^\n]*}, "")

  @program """
  .include "words.dl"

  // Stage 0's call graph.
  .decl call_edge(caller: symbol, callee: symbol)
  .input call_edge

  .decl calls(caller: symbol, callee: symbol)
  calls(caller, callee) :- call_edge(caller, callee).
  .output calls

  .decl unsupported_type(type: symbol)
  .input unsupported_type

  .decl unsupported(type: symbol)
  unsupported(type) :- unsupported_type(type).
  .output unsupported
  """

  @words """
  // A function its module exports, from Argus's facts.
  .decl exported(func: symbol)
  exported(func) :- function_def(func, _, _, _, 1).
  .output exported
  """

  describe "with a cache" do
    # Three small modules and a small program over them, in a directory of
    # the test's own: the program reads Argus's facts through an include,
    # stage 0's call graph, and an option's relation.
    setup do
      directory = Path.join(System.tmp_dir!(), "solve_test_#{System.unique_integer([:positive])}")
      File.mkdir_p!(directory)
      on_exit(fn -> File.rm_rf!(directory) end)

      beams =
        for {name, source} <- modules(),
            do: compile_erlang(directory, name, source)

      program = Path.join(directory, "program.dl")
      File.write!(program, @program)
      File.write!(Path.join(directory, "words.dl"), @words)

      %{
        directory: directory,
        beams: beams,
        program: program,
        cache: Path.join(directory, "cache")
      }
    end

    test "a run that changed nothing extracts and solves nothing", context do
      {rows, events} = solve(context)

      assert events == [
               extract: "solve_test_first.beam",
               extract: "solve_test_second.beam",
               extract: "solve_test_third.beam",
               solve: :rules,
               solve: :stage0
             ]

      assert [":solve_test_first:run/1", ":solve_test_first:helper/1"] in rows["calls"]
      assert {^rows, []} = solve(context)
    end

    test "a changed beam extracts that module alone", context do
      solve(context)
      compile_erlang(context.directory, :solve_test_second, second(["run/1", "other/0"]))

      {rows, events} = solve(context)
      assert for({:extract, beam} <- events, do: beam) == ["solve_test_second.beam"]
      assert [":solve_test_second:other/0"] in rows["exported"]
    end

    test "a change that only moves lines extracts the module and solves nothing", context do
      {rows, _events} = solve(context)
      compile_erlang(context.directory, :solve_test_first, "\n\n\n" <> first())

      assert {^rows, [extract: "solve_test_first.beam"]} = solve(context)
    end

    test "an edit to an extractor's code extracts every module again", context do
      directory = Path.join(context.directory, "extractor")
      File.mkdir_p!(directory)
      Code.prepend_path(directory)
      on_exit(fn -> Code.delete_path(directory) end)
      compile_extractor(directory, "1")

      solve = fn ->
        Solve.solve(Analysis, context.beams, context.program, cache: context.cache)
      end

      {_rows, _events} = events(context.cache, solve)
      assert {_rows, []} = events(context.cache, solve)

      # The code versions this VM memoized are of the code it started
      # with, as a run's are.
      compile_extractor(directory, "2")
      Roux.Code.forget()

      {_rows, events} = events(context.cache, solve)

      # Its rows are no relation the program reads: nothing is solved.
      assert events == [
               extract: "solve_test_first.beam",
               extract: "solve_test_second.beam",
               extract: "solve_test_third.beam"
             ]
    end

    test "a changed rule file, or a file it includes, solves again", context do
      solve(context)

      File.write!(
        context.program,
        @program <>
          """
          .decl caller(func: symbol)
          caller(func) :- call_edge(func, _).
          .output caller
          """
      )

      {rows, events} = solve(context)
      assert events == [solve: :rules]
      assert [":solve_test_first:run/1"] in rows["caller"]

      File.write!(
        Path.join(context.directory, "words.dl"),
        String.replace(@words, ", 1)", ", 0)")
      )

      {rows, events} = solve(context)
      assert events == [solve: :rules]
      assert [":solve_test_first:helper/1"] in rows["exported"]
    end

    test "a comment in the rules, or a blank line, solves nothing", context do
      {rows, _events} = solve(context)
      File.write!(context.program, "// Why the program is.\n\n" <> @program)
      File.write!(Path.join(context.directory, "words.dl"), @words <> "\n  // A last word.\n")

      assert {^rows, []} = solve(context)
    end

    test "another solver solves again", context do
      solve(context)

      # The same souffle, by another file.
      bin = Path.join(context.directory, "bin")
      File.mkdir_p!(bin)
      wrapper = Path.join(bin, "souffle")
      File.write!(wrapper, "#!/bin/sh\nexec '#{Argus.Souffle.executable()}' \"$@\"\n")
      File.chmod!(wrapper, 0o755)
      path = System.get_env("PATH")

      try do
        System.put_env("PATH", bin <> ":" <> path)
        assert {_rows, [solve: :rules, solve: :stage0]} = solve(context)
      after
        System.put_env("PATH", path)
      end
    end

    test "other options solve again, and ones solved before are read back", context do
      {rows, _events} = solve(context)
      assert rows["unsupported"] == []

      {typed, events} = solve(context, unsupported_types: [:f64, {:c, 128}])
      assert events == [solve: :rules]
      assert typed["unsupported"] == [["c128"], ["f64"]]

      assert {^rows, []} = solve(context)
    end

    test "a run that lost its manifest reads its solves back from the store", context do
      {rows, _events} = solve(context)
      File.rm!(Path.join(context.cache, "manifest"))

      assert {^rows,
              [
                extract: "solve_test_first.beam",
                extract: "solve_test_second.beam",
                extract: "solve_test_third.beam"
              ]} = solve(context)
    end
  end

  defp modules do
    [
      solve_test_first: first(),
      solve_test_second: second(["run/1"]),
      solve_test_third: """
      -module(solve_test_third).
      -export([run/0]).
      run() -> solve_test_first:run(2).
      """
    ]
  end

  defp first do
    """
    -module(solve_test_first).
    -export([run/1]).
    run(Value) -> helper(Value) + 1.
    helper(Value) -> Value * 2.
    """
  end

  defp second(exports) do
    """
    -module(solve_test_second).
    -export([#{Enum.join(exports, ", ")}]).
    run(Value) -> lists:reverse(Value).
    other() -> ok.
    """
  end

  defp compile_erlang(directory, name, source) do
    file = Path.join(directory, "#{name}.erl")
    File.write!(file, source)

    {:ok, ^name} =
      :compile.file(String.to_charlist(file), [
        :report_errors,
        outdir: String.to_charlist(directory)
      ])

    Path.join(directory, "#{name}.beam")
  end

  # An extractor of one relation, which no program here reads, whose rows
  # carry `version`.
  defp compile_extractor(directory, version) do
    file = Path.join(directory, "extractor.ex")

    File.write!(file, """
    defmodule ArgusNxTensorAnalyses.SolveTest.Extractor do
      @behaviour Argus.Extractor

      @impl true
      def relations, do: [:solve_test_version]

      @impl true
      def extract(module_data),
        do: %{solve_test_version: [[inspect(module_data.module), "#{version}"]]}
    end
    """)

    capture_io(:stderr, fn ->
      {:ok, _modules, _diagnostics} =
        Kernel.ParallelCompiler.compile_to_path([file], directory, return_diagnostics: true)
    end)
  end

  # The program solved over the modules with the tensor shapes analysis's
  # extractors, and what the run extracted and solved.
  defp solve(context, options \\ []) do
    events(context.cache, fn ->
      Solve.solve(
        TensorShapes,
        context.beams,
        context.program,
        [cache: context.cache] ++ options
      )
    end)
  end

  # `solve`'s rows, and each module the graph kept under `cache`
  # extracted and each solve it ran meanwhile, sorted.
  defp events(cache, solve) do
    handler = make_ref()
    store = Path.expand(Path.join(cache, "store"))

    :telemetry.attach_many(
      handler,
      [
        [:argus_nx_tensor_analyses, :graph, :extract],
        [:argus_nx_tensor_analyses, :graph, :solve]
      ],
      &__MODULE__.forward/4,
      {self(), handler, store}
    )

    try do
      {:ok, rows} = solve.()
      {rows, received(handler, [])}
    after
      :telemetry.detach(handler)
    end
  end

  @doc false
  def forward(event, _measurements, %{store: store} = metadata, {test, handler, store}),
    do: send(test, {handler, List.last(event), metadata})

  def forward(_event, _measurements, _metadata, _config), do: :ok

  defp received(handler, events) do
    receive do
      {^handler, :extract, %{path: path}} ->
        received(handler, [{:extract, Path.basename(path)} | events])

      {^handler, :solve, %{stage: stage}} ->
        received(handler, [{:solve, stage} | events])
    after
      0 -> Enum.sort(events)
    end
  end
end
