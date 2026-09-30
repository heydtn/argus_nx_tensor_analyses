defmodule ArgusNxTensorAnalyses.Graph.ProgramTest do
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.Graph.Program
  alias ArgusNxTensorAnalyses.Solve

  unless Argus.Souffle.available?() do
    @moduletag skip: "souffle is not on PATH"
  end

  setup do
    directory = Path.join(System.tmp_dir!(), "program_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(directory)
    on_exit(fn -> File.rm_rf!(directory) end)
    %{directory: directory}
  end

  test "Argus's stage 0 reads every relation Souffle loads of it" do
    stage0 = Argus.Analysis.stage0_rules_path()
    {:ok, loaded} = Argus.Souffle.input_relations(stage0)

    assert loaded -- Program.read([stage0]).reads == []
  end

  test "a program reads every relation Souffle loads of it, and not the ones it only declares",
       %{directory: directory} do
    {reads, loaded} =
      read(directory, """
      .decl call_edge(caller: symbol, callee: symbol)
      .input call_edge

      .decl leaf(func: symbol)
      leaf(func) :- function_def(func, _, _, _, _), !call_edge(func, _).
      .output leaf
      """)

    assert loaded -- reads == []
    assert "function_def" in reads and "call_edge" in reads
    refute "line_info" in reads
  end

  test "a relation named anywhere outside its declaration is read", %{directory: directory} do
    {reads, loaded} =
      read(directory, """
      .decl said(text: symbol)
      said("line_info").
      .output said
      """)

    assert loaded -- reads == []
    assert "line_info" in reads
  end

  test "an input the text does not declare plainly makes every relation read",
       %{directory: directory} do
    for text <- [
          ".decl moved(func: symbol)\n.input moved(IO=file, filename=\"elsewhere.facts\")\n",
          ".decl one(x: symbol)\n.decl two(x: symbol)\n.input one, two\n",
          ".comp Inputs {\n  .decl given(x: symbol)\n  .input given\n}\n.init inputs = Inputs\n",
          "#define READ(name) name\n"
        ] do
      assert Program.read(roots(directory, text)).reads == :all, text
    end
  end

  test "a comment or a blank line leaves the program's files as they were",
       %{directory: directory} do
    text =
      ".decl leaf(func: symbol)\nleaf(func) :- function_def(func, _, _, _, _).\n.output leaf\n"

    before = Program.read(roots(directory, text))
    after_comment = Program.read(roots(directory, "// Leaves.\n\n" <> text <> "  // done\n"))

    assert after_comment.files == before.files

    assert after_comment.files !=
             Program.read(roots(directory, text <> "leaf(\"more\").\n")).files
  end

  test "a program that includes a file that is not there cannot be read",
       %{directory: directory} do
    assert %{files: {:unreadable, _path}, reads: :all} =
             Program.read(roots(directory, ".include \"missing.dl\"\n"))
  end

  # The program `text` after Argus's files, as the runner solves it: the
  # relations its text says it reads, and those Souffle loads.
  defp read(directory, text) do
    roots = roots(directory, text)
    wrapper = Path.join(directory, "wrapper.dl")
    File.write!(wrapper, Enum.map(roots, &~s(.include "#{&1}"\n)))
    {:ok, loaded} = Argus.Souffle.input_relations(wrapper)
    {Program.read(roots).reads, loaded}
  end

  defp roots(directory, text) do
    program = Path.join(directory, "program.dl")
    File.write!(program, text)
    Solve.argus_includes() ++ [program]
  end
end
