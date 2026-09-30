defmodule ArgusNxTensorAnalyses.TensorAnalysisCase do
  @moduledoc false
  # A test module of the analyses: fixtures compiled into a directory of
  # their own and solved once, for the module's tests to read what the
  # analysis finds in them and what Nx does running them. A solve's rows
  # are cached under the build path, and a run whose facts, rules, Argus
  # and solver are unchanged reads them back.
  #
  # Not async: compiling the fixtures loads them into the VM, and silences
  # the compiler by capturing `:stderr`, which every process shares.

  use ExUnit.CaseTemplate

  import ExUnit.Assertions
  import ExUnit.CaptureIO

  using do
    quote do
      use ExUnit.Case, async: false

      import ArgusNxTensorAnalyses.TensorAnalysisCase

      unless Argus.Souffle.available?() do
        @moduletag skip: "souffle is not on PATH"
      end
    end
  end

  # Compiles the fixtures' source into a directory of its own under the
  # build path, emptied first so that no beam of an earlier run is left
  # in it, and keeps its beams as `name` where
  # `ArgusNxTensorAnalyses.FixtureBeams` keeps them: `%{directory, source,
  # beams}`. Call it from `setup_all`.
  #
  # The directory is the checkout's, the same on every run: the path the
  # fixtures are compiled from can reach their facts, which a solve's
  # cache is keyed on.
  @spec compile_fixtures(String.t(), String.t()) :: %{atom() => term()}
  def compile_fixtures(name, source) do
    directory = suite_path(["fixtures", name])
    File.rm_rf!(directory)
    File.mkdir_p!(directory)
    path = Path.join(directory, "fixtures.ex")
    File.write!(path, source)

    capture_io(:stderr, fn ->
      {:ok, _modules, _diagnostics} =
        Kernel.ParallelCompiler.compile_to_path([path], directory, return_diagnostics: true)
    end)

    ArgusNxTensorAnalyses.FixtureBeams.keep(directory, name)
    %{directory: directory, source: path, beams: Path.wildcard(Path.join(directory, "*.beam"))}
  end

  # Runs each solve at once, since each spends most of its time in Souffle
  # compiling its program, and gives each result by its name. A solve is a
  # function of the directory to cache its rows in, giving `{:ok,
  # result}`; each has its own, named for `group` and the solve, since a
  # cache keeps only its latest rows.
  @spec solve_concurrently(String.t(), keyword((Path.t() -> {:ok, term()}))) ::
          %{atom() => term()}
  def solve_concurrently(group, solves) do
    solves
    |> Task.async_stream(
      fn {name, solve} -> {name, solve.(suite_path(["solves", group, Atom.to_string(name)]))} end,
      max_concurrency: length(solves),
      timeout: :infinity
    )
    |> Map.new(fn {:ok, {name, {:ok, result}}} -> {name, result} end)
  end

  # A path under the build path, which each checkout has its own of.
  defp suite_path(parts), do: Path.join([Mix.Project.build_path(), "suite" | parts])

  # A function as the rules name it, `Module:name/arity`, and a `defn`'s
  # body, which the compiler names `__defn:name__`.
  @spec function_id(module(), atom() | String.t(), non_neg_integer()) :: String.t()
  def function_id(module, name, arity), do: "#{inspect(module)}:#{name}/#{arity}"

  @spec defn_id(module(), atom() | String.t(), non_neg_integer()) :: String.t()
  def defn_id(module, name, arity), do: "#{inspect(module)}:__defn:#{name}__/#{arity}"

  # The rows of the relations, in order, in the functions `owner` takes (a
  # function's name, or a test of it), each as the fields asked for: one
  # field's value, or a tuple of several. Every row is kept, repeated or
  # not. `:relation` is a field too.
  @spec findings_for(
          map(),
          String.t() | [String.t()],
          String.t() | (String.t() -> boolean()),
          atom() | [atom()]
        ) ::
          list()
  def findings_for(rows, relations, owner, fields) do
    owns? = if is_function(owner, 1), do: owner, else: &(&1 == owner)

    for relation <- List.wrap(relations),
        names = Map.fetch!(field_names(), relation),
        func = Enum.find_index(names, &(&1 == :func)),
        row <- Map.get(rows, relation, []),
        owns?.(Enum.at(row, func)),
        do: row |> named(names, relation) |> project(fields)
  end

  # The rows of a shape relation in the function, and in the functions it
  # calls with the shapes it gives them (its `_via` relation's call
  # frames), as `findings_for/4` gives them.
  @spec reached_findings(map(), String.t(), String.t(), atom() | [atom()]) :: list()
  def reached_findings(rows, relation, function, fields) do
    reached =
      for [id, kind, _order, _call, ^function, "call" | _frame] <-
            Map.get(rows, relation <> "_via", []),
          do: {id, kind}

    names = Map.fetch!(field_names(), relation)

    for row <- Map.get(rows, relation, []),
        finding = named(row, names, relation),
        finding.func == function or {finding.id, finding.kind} in reached,
        do: project(finding, fields)
  end

  # A row of an output relation of either analysis by its fields' names.
  defp named(row, names, relation),
    do: names |> Enum.zip(row) |> Map.new() |> Map.put(:relation, relation)

  defp field_names do
    for analysis <- ArgusNxTensorAnalyses.analyses(),
        %{name: name, fields: fields} <- analysis.output_relations(),
        into: %{},
        do: {Atom.to_string(name), Enum.map(fields, &elem(&1, 0))}
  end

  defp project(finding, fields) when is_list(fields),
    do: fields |> Enum.map(&Map.fetch!(finding, &1)) |> List.to_tuple()

  defp project(finding, field), do: Map.fetch!(finding, field)

  # The placed findings in the module.
  @spec placed_in([Argus.Located.t()], module()) :: [Argus.Located.t()]
  def placed_in(placed, module), do: Enum.filter(placed, &(&1.finding.module == module))

  # Asserts the findings found hold the one expected, or do not.
  @spec assert_finding(list(), term()) :: true
  def assert_finding(found, expected),
    do: assert(expected in found, "expected #{inspect(expected)}, found #{inspect(found)}")

  @spec refute_finding(list(), term()) :: false
  def refute_finding(found, unexpected),
    do: refute(unexpected in found, "expected no #{inspect(unexpected)}, found #{inspect(found)}")

  # What a fixture does run on the binary backend: `{:returns, value}`, or
  # `{:raises, exception}`.
  @spec outcome_on_binary_backend(module(), atom(), list()) ::
          {:returns, term()} | {:raises, Exception.t()}
  def outcome_on_binary_backend(module, function, arguments) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      {:returns, apply(module, function, arguments)}
    end)
  rescue
    error -> {:raises, error}
  end

  # An outcome as a case expects it: `:nonfinite` where the value holds an
  # infinity or a NaN, `:finite` where it does not, and `:raises` where it
  # raises, but that an ArithmeticError (an integer divided by zero) is
  # `arithmetic`.
  @spec classify_outcome({:returns, term()} | {:raises, Exception.t()}, atom()) :: atom()
  def classify_outcome({:returns, value}, _arithmetic),
    do: if(nonfinite?(value), do: :nonfinite, else: :finite)

  def classify_outcome({:raises, %ArithmeticError{}}, arithmetic), do: arithmetic
  def classify_outcome({:raises, _error}, _arithmetic), do: :raises

  # Whether a value holds an infinity or a NaN: a tensor's element, or one
  # of a tuple's, list's or map's.
  @spec nonfinite?(term()) :: boolean()
  def nonfinite?(%Nx.Tensor{} = tensor) do
    tensor
    |> Nx.is_nan()
    |> Nx.logical_or(Nx.is_infinity(tensor))
    |> Nx.any()
    |> Nx.to_number() == 1
  end

  def nonfinite?(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> nonfinite?()
  def nonfinite?(list) when is_list(list), do: Enum.any?(list, &nonfinite?/1)

  def nonfinite?(map) when is_map(map) and not is_struct(map),
    do: map |> Map.values() |> nonfinite?()

  def nonfinite?(_other), do: false

  # The source file's line, as written.
  @spec source_line(Path.t(), pos_integer()) :: String.t()
  def source_line(source, line),
    do: source |> File.read!() |> String.split("\n") |> Enum.at(line - 1)
end
