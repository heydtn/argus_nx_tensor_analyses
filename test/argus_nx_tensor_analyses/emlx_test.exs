defmodule ArgusNxTensorAnalyses.EMLXTest do
  # Each case below is compiled into a function of one module, and the
  # analysis of the compiled module has to find what the case expects.
  # This package does not depend on EMLX or EXLA, so the cases are not
  # run: what EMLX does with each is recorded in the analysis's findings'
  # wording, confirmed against EMLX 0.4.2.
  use ArgusNxTensorAnalyses.TensorAnalysisCase

  alias ArgusNxTensorAnalyses.EMLX
  alias ArgusNxTensorAnalyses.TensorShapes

  @fixtures ArgusNxTensorAnalyses.EMLXTest.Fixtures
  @on_exla ArgusNxTensorAnalyses.EMLXTest.OnExla

  @helpers """
    defn double(x), do: x * 2
    defn f64_table(n), do: Nx.iota({4}, type: :f64) * n
    defn wrap(x), do: rem(x - 1, 4)
    defn inverse(x), do: x ** -1
    defn add_iota(x), do: x + Nx.iota({2}, type: :f32)
    defn shift_far(x), do: x <<< 33
    defn rope_tables(n), do: %{cos: Nx.as_type(Nx.cos(Nx.iota({4}, type: :f64) * n), :f32)}
    defp on_exla(tensor), do: Nx.backend_transfer(tensor, EXLA.Backend)
    defp combine(left, right), do: Nx.add(left, right)
    defp tables(n), do: %{cos: Nx.iota({n}, type: :f32), sin: Nx.iota({n}, type: :f32)}
  """

  # Each case is the body of a function of `t`, an input the analysis does
  # not follow, and what the analysis finds in it (and in its closures):
  # `{:finds, kind, detail, certain}`, a divergence; `{:mixed, backend,
  # other, origin_operation}`, tensors of two backends; `:quiet`, nothing.
  @cases [
    # types EMLX does not have
    {{:finds, "narrowed_type", "f64", "1"}, "Nx.iota({3}, type: :f64)"},
    {{:finds, "narrowed_type", "f64", "1"}, "Nx.tensor([1.0], type: {:f, 64})"},
    {{:finds, "narrowed_type", "f64", "1"}, "Nx.as_type(t, :f64)"},
    {{:finds, "narrowed_type", "f64", "1"}, "Nx.Constants.pi({:f, 64})"},
    {:quiet, "Nx.iota({3}, type: :f32)"},
    # a call whose options Nx rejects makes no tensor of the type they name
    {:quiet, "Nx.Constants.pi(:f32, type: :f64)"},
    {:quiet, "Nx.iota({3}, type: :f64, backend: EXLA.Backend)"},
    {:quiet, "Nx.Defn.jit(&f64_table/1, compiler: EXLA).(t)"},
    # a jit's options are its compiler's, and name no tensor's type
    {:quiet, "Nx.add(Nx.Defn.jit(fn x -> x end, type: :f64).(t), 1)"},
    {:quiet, "Nx.as_type(Nx.backend_transfer(t, EXLA.Backend), :f64)"},
    # listed as unsupported, so nx_types reports it
    {:quiet, "Nx.tensor([1.0], type: :c128)"},
    # tensors of those types moved onto EMLX
    {{:finds, "narrowed_transfer", "f64", "1"},
     "Nx.backend_transfer(Nx.iota({3}, type: :f64, backend: EXLA.Backend), EMLX.Backend)"},
    {{:finds, "narrowed_transfer", "f64", "1"},
     "Nx.backend_copy(Nx.as_type(Nx.backend_transfer(t, EXLA.Backend), :f64), EMLX.Backend)"},
    {{:finds, "narrowed_transfer", "c128", "1"},
     "Nx.backend_transfer(Nx.as_type(Nx.backend_transfer(t, EXLA.Backend), :c128), {EMLX.Backend, device: :gpu})"},
    {:quiet,
     "Nx.backend_transfer(Nx.iota({3}, type: :f64, backend: EXLA.Backend), Nx.BinaryBackend)"},
    {{:finds, "narrowed_type", "f64", "1"},
     "Nx.backend_transfer(Nx.iota({3}, type: :f64), EMLX.Backend)"},
    # f64 tables built on EXLA and cast to f32 before they move
    {:quiet,
     "table = Nx.Defn.jit(&f64_table/1, compiler: EXLA).(t)\nNx.backend_transfer(Nx.as_type(table, :f32), EMLX.Backend)"},
    {:quiet,
     "tables = Nx.Defn.jit(&rope_tables/1, compiler: EXLA).(t)\nNx.backend_transfer(tables.cos, EMLX.Backend)"},
    # remainders of negative numbers
    {{:finds, "negative_remainder", "dividend", "1"},
     "Nx.remainder(Nx.subtract(Nx.iota({4}), 2), 3)"},
    {{:finds, "negative_remainder", "divisor", "1"}, "Nx.remainder(Nx.iota({4}), -3)"},
    {{:finds, "negative_remainder", "divisor", "1"},
     "Nx.remainder(Nx.iota({4}), Nx.subtract(Nx.iota({4}), 2))"},
    {{:finds, "negative_remainder", "dividend", "0"}, "Nx.remainder(t, 3)"},
    {:quiet, "Nx.remainder(Nx.iota({4}), 3)"},
    {:quiet, "Nx.remainder(Nx.abs(t), 3)"},
    # integer powers with negative exponents
    {{:finds, "negative_integer_power", "exponent", "1"}, "Nx.pow(Nx.iota({3}), -1)"},
    {{:finds, "negative_integer_power", "exponent", "1"},
     "Nx.pow(2, Nx.subtract(Nx.iota({3}), 2))"},
    {:quiet, "Nx.pow(Nx.iota({3}), 2)"},
    {:quiet, "Nx.pow(Nx.iota({3}, type: :f32), -1)"},
    {:quiet, "Nx.pow(Nx.iota({3}), -0.5)"},
    # rounding halves
    {{:finds, "round_half", "halved", "1"}, "Nx.round(Nx.divide(Nx.iota({5}), 2))"},
    {{:finds, "round_half", "halved", "1"}, "Nx.round(Nx.multiply(Nx.iota({5}), 0.5))"},
    {{:finds, "round_half", "half_added", "1"}, "Nx.round(Nx.add(Nx.iota({5}), 0.5))"},
    {{:finds, "round_half", "mean", "1"}, "Nx.round(Nx.mean(Nx.iota({4})))"},
    {:quiet, "Nx.round(Nx.divide(Nx.iota({5}), 3))"},
    {:quiet, "Nx.round(Nx.multiply(Nx.iota({5}, type: :f32), 0.5))"},
    {:quiet, "Nx.round(t)"},
    {:quiet, "Nx.round(Nx.divide(t, 2))"},
    # shifts past the width EMLX shifts the type in
    {{:finds, "wrapped_shift", "s32 33", "1"}, "Nx.left_shift(Nx.tensor(1), 33)"},
    {{:finds, "wrapped_shift", "s32 32", "1"}, "Nx.right_shift(Nx.iota({3}), 32)"},
    {{:finds, "wrapped_shift", "u8 32", "1"}, "Nx.left_shift(Nx.tensor(1, type: :u8), 32)"},
    {{:finds, "wrapped_shift", "s64 64", "1"}, "Nx.left_shift(Nx.tensor(1, type: :s64), 64)"},
    {{:finds, "wrapped_shift", "u32 65", "1"}, "Nx.left_shift(Nx.tensor(1, type: :u32), 65)"},
    {{:finds, "wrapped_shift", "u16 40", "1"}, "Nx.left_shift(Nx.tensor(1, type: :u16), 40)"},
    # wrapped to an amount the type shifts every bit out at, as the others do
    {:quiet, "Nx.left_shift(Nx.tensor(1, type: :u8), 40)"},
    {:quiet, "Nx.left_shift(Nx.tensor(1, type: :u32), 96)"},
    {:quiet, "Nx.right_shift(Nx.tensor(-8), 63)"},
    {:quiet, "Nx.left_shift(Nx.tensor(1), 31)"},
    {:quiet, "Nx.left_shift(Nx.tensor(1, type: :u8), 8)"},
    {:quiet, "Nx.left_shift(Nx.tensor(1, type: :u32), 32)"},
    # tensors of two backends
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.tensor/2"},
     "Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.tensor/2"},
     "Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.tri(1, 1))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.tensor/2"},
     "Nx.add(Nx.tensor([1.0], backend: {EXLA.Backend, client: :host}), Nx.iota({1}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.backend_transfer/2"},
     "Nx.add(Nx.backend_transfer(Nx.iota({2}), EXLA.Backend), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.Defn.jit/2"},
     "compiled = Nx.Defn.jit(&double/1, compiler: EXLA)\nNx.multiply(compiled.(Nx.iota({2})), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.Defn.jit_apply/3"},
     "Nx.add(Nx.Defn.jit_apply(&double/1, [Nx.iota({2})], compiler: EXLA), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "EXLA.jit/1"},
     "Nx.add(EXLA.jit(&double/1).(Nx.iota({2})), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.with_default_backend/2"},
     "Nx.add(Nx.with_default_backend(EXLA.Backend, fn -> Nx.iota({2}) end), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.tensor/2"},
     "Nx.concatenate([Nx.tensor([1.0], backend: EXLA.Backend), Nx.tensor([2.0])])"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.Defn.jit/2"},
     "tables = Nx.Defn.jit(&tables/1, compiler: EXLA).(4)\nNx.multiply(tables.cos, Nx.iota({4}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.backend_transfer/2"},
     "Nx.add(on_exla(Nx.iota({2})), Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.backend_transfer/2"},
     "held = %{cos: Nx.backend_transfer(Nx.iota({2}), EXLA.Backend)}\nNx.add(held.cos, Nx.iota({2}))"},
    {{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.backend_transfer/2"},
     "t = if Nx.to_number(Nx.sum(t)) > 0, do: Nx.backend_transfer(t, EXLA.Backend), else: Nx.iota({2})\nNx.add(t, Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.backend_transfer(Nx.iota({2}), EMLX.Backend), Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.Defn.jit(&double/1, compiler: EMLX).(Nx.iota({2})), Nx.iota({2}))"},
    # calls in code Nx traces with the compiler configured outside the code
    {:quiet,
     "Nx.Defn.jit(fn x -> Nx.add(x, Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))) end).(t)"},
    {:quiet,
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.multiply(x, Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1})))) end)"},
    {:quiet,
     "Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.tensor([1.0], backend: Nx.BinaryBackend))"},
    {:quiet, "Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), 1)"},
    {:quiet, "Nx.add(Nx.iota({2}), Nx.bit_size(Nx.tensor([1.0], backend: EXLA.Backend)))"},
    {:quiet,
     "Nx.add(Nx.backend_transfer(Nx.iota({2}), EXLA.Backend), Nx.tensor([1, 2], backend: EXLA.Backend))"},
    {:quiet,
     "t = if Nx.to_number(Nx.sum(t)) > 0, do: Nx.backend_transfer(t, EXLA.Backend), else: Nx.iota({2})\nNx.add(t, 1)"},
    {:quiet,
     "Nx.with_default_backend(EXLA.Backend, fn -> Nx.add(Nx.backend_transfer(t, EXLA.Backend), Nx.iota({2})) end)"},
    {:quiet, "Nx.add(Nx.iota(Nx.backend_transfer(t, EXLA.Backend)), Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.broadcast(0.0, Nx.backend_transfer(t, EXLA.Backend)), Nx.iota({2}))"},
    {:quiet,
     "Nx.add(Nx.tensor(Nx.backend_transfer(t, EXLA.Backend)), Nx.backend_transfer(t, EXLA.Backend))"}
  ]

  # A module of its own, solved apart, whose code sets EXLA as the default
  # backend.
  @on_exla_source """
  defmodule #{inspect(@on_exla)} do
    def start, do: Nx.default_backend(EXLA.Backend)

    def emlx_meets_default do
      moved = Nx.backend_transfer(Nx.iota({2}), EMLX.Backend)
      Nx.add(moved, Nx.iota({2}))
    end

    def exla_meets_default, do: Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))
    def f64_on_default, do: Nx.iota({3}, type: :f64)
  end
  """

  setup_all do
    %{beams: beams} = compile_fixtures("emlx", fixture_source())

    %{source: on_exla_source, beams: on_exla_beams} =
      compile_fixtures("emlx_on_exla", @on_exla_source)

    # c128 listed as unsupported: nx_types reports its tensors, and emlx
    # leaves them to it.
    solved =
      solve_concurrently("emlx",
        rows: &EMLX.solve(beams, unsupported_types: [:c128], cache: &1),
        nx_rows:
          &TensorShapes.solve(beams, TensorShapes.rules_file(),
            unsupported_types: [:c128],
            cache: &1
          ),
        categorized: &categorized(beams, &1),
        on_exla: &EMLX.run(on_exla_beams, cache: &1)
      )

    Map.put(solved, :on_exla_source, on_exla_source)
  end

  for {{expectation, body}, index} <- Enum.with_index(@cases, 1) do
    test "#{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      found = findings(rows, unquote(index))

      case unquote(Macro.escape(expectation)) do
        :quiet ->
          assert found == [], "expected nothing, found #{inspect(found)}"

        {:finds, kind, detail, certain} ->
          assert_finding(found, {:divergence, kind, detail, certain})

        {:mixed, backend, other, origin} ->
          assert_finding(found, {:mixed, backend, other, origin})
      end
    end
  end

  test "a type listed as unsupported is an nx_types finding", %{rows: rows} do
    function = case_id(Enum.find_index(@cases, &(elem(&1, 1) =~ ":c128")) + 1)

    assert [{"Nx.tensor/2", "unsupported_type", "c128"}] =
             findings_for(rows, "tensor_type_error", function, [:operation, :kind, :subject])
  end

  test "a tensor made f64 on EMLX is reported where it is made, not where it moves", %{
    rows: rows
  } do
    index = Enum.find_index(@cases, &(elem(&1, 1) =~ "Nx.iota({3}, type: :f64), EMLX")) + 1

    assert [{:divergence, "narrowed_type", "f64", "1"}] = findings(rows, index)
  end

  test "a remainder in a defn whose dividend its math makes negative", %{rows: rows} do
    assert {"negative_remainder", "dividend", "1", "Nx.Defn.Kernel.-/2"} in divergences_in(
             rows,
             defn_id(@fixtures, :wrap, 1)
           )
  end

  test "a power in a defn whose exponent is written negative", %{rows: rows} do
    assert {"negative_integer_power", "exponent", "1", ""} in divergences_in(
             rows,
             defn_id(@fixtures, :inverse, 1)
           )
  end

  test "a shift in a defn past the width EMLX shifts in", %{rows: rows} do
    assert {"wrapped_shift", "s32 33", "1", ""} in divergences_in(
             rows,
             defn_id(@fixtures, :shift_far, 1)
           )
  end

  test "an f64 tensor a function jitted with EXLA makes is not EMLX's", %{rows: rows} do
    assert divergences_in(rows, defn_id(@fixtures, :f64_table, 1)) == []
  end

  test "a helper that adds tensors of two backends its caller hands it", %{rows: rows} do
    function = function_id(@fixtures, :combine, 2)

    assert [{"Nx.add/2", "EXLA.Backend", origin, "Nx.tensor/2"}] =
             findings_for(rows, "tensor_emlx_mixed_backends", function, [
               :operation,
               :backend,
               :origin,
               :origin_operation
             ])

    assert origin =~ "#{inspect(@fixtures)}:calls_combine/0#"
  end

  # Its argument is on EXLA and its iota on the default backend, but a
  # defn's calls build an expression, and whether they mix backends is up
  # to the compiler that runs it, which the code does not name.
  test "a defn's body is traced, and mixes no backends", %{rows: rows} do
    function = defn_id(@fixtures, :add_iota, 1)

    assert findings_for(rows, "tensor_emlx_mixed_backends", function, :id) == []
  end

  test "a mix is reported where it happens, not where its result goes", %{rows: rows} do
    index = length(@cases) + 1

    assert [{:mixed, "EXLA.Backend", "EMLX.Backend", "Nx.tensor/2"}] = findings(rows, index)
    assert [["Nx.add/2"]] = mixed_operations(rows, index)
  end

  test "a tensor on EMLX meets one made on the default the code sets", %{on_exla: on_exla} do
    titles = Enum.map(on_exla, & &1.finding.title)

    assert "Nx.add/2 gets tensors of EXLA.Backend and EMLX.Backend, which cannot meet" in titles
    assert length(titles) == 1, inspect(titles)
  end

  test "run/2 places a mix at its call, and each tensor at the call that places it", %{
    on_exla: [located],
    on_exla_source: source
  } do
    at = &(source |> source_line(&1.line) |> String.trim())

    assert located.file == source
    assert at.(located) == "Nx.add(moved, Nx.iota({2}))"

    assert [
             %{label: "makes a tensor on EXLA.Backend: Nx.iota/1"},
             %{label: "puts a tensor on EMLX.Backend: Nx.backend_transfer/2"}
           ] = located.finding.related

    assert Enum.map(located.related, at) == [
             "Nx.add(moved, Nx.iota({2}))",
             "moved = Nx.backend_transfer(Nx.iota({2}), EMLX.Backend)"
           ]
  end

  test "the program outputs the Nx engine's rows as the Nx program does", %{
    rows: rows,
    nx_rows: nx_rows
  } do
    assert nx_rows |> nx_report_rows() |> Map.values() |> Enum.concat() != []
    assert nx_report_rows(rows) == nx_report_rows(nx_rows)
  end

  test "a run of every category solves this program alone", %{categorized: categorized} do
    %{found: found, kept: kept} = categorized

    assert kept == ["emlx"]
    assert found |> Map.keys() |> Enum.sort() == Enum.sort(ArgusNxTensorAnalyses.analyses())

    for {category, placed} <- found, %{finding: finding} <- placed do
      assert {finding.analysis, finding.concern} == {category, category}
    end

    assert found.emlx != []
    assert Enum.any?(found.nx_types, &(&1.finding.title =~ "c128"))
  end

  # Every category's findings in the modules, and what the run keeps under
  # `cache`.
  defp categorized(beams, cache) do
    with {:ok, found} <-
           ArgusNxTensorAnalyses.run(beams, [:all], unsupported_types: [:c128], cache: cache),
         do: {:ok, %{found: found, kept: File.ls!(cache)}}
  end

  defp case_id(index), do: function_id(@fixtures, "case_#{index}", 1)

  # The findings in the case's function and its closures: divergences as
  # `{:divergence, kind, detail, certain}`, mixes as `{:mixed, backend,
  # other, origin_operation}`.
  defp findings(rows, index) do
    function = case_id(index)
    closure = "#{inspect(@fixtures)}:-case_#{index}/1-fun-"
    in_case? = &(&1 == function or String.starts_with?(&1, closure))

    divergences =
      for {kind, detail, certain} <-
            findings_for(rows, "tensor_emlx_divergence", in_case?, [:kind, :detail, :certain]),
          uniq: true,
          do: {:divergence, kind, detail, certain}

    mixes =
      for {backend, other, shown} <-
            findings_for(rows, "tensor_emlx_mixed_backends", in_case?, [
              :backend,
              :other,
              :origin_operation
            ]),
          uniq: true,
          do: {:mixed, backend, other, shown}

    divergences ++ mixes
  end

  defp mixed_operations(rows, index) do
    for operation <- findings_for(rows, "tensor_emlx_mixed_backends", case_id(index), :operation),
        uniq: true,
        do: [operation]
  end

  defp divergences_in(rows, function) do
    rows
    |> findings_for("tensor_emlx_divergence", function, [
      :kind,
      :detail,
      :certain,
      :origin_operation
    ])
    |> Enum.uniq()
  end

  defp fixture_source do
    cases =
      @cases
      |> Enum.with_index(1)
      |> Enum.map_join("\n", fn {{_expectation, body}, index} ->
        "  def case_#{index}(t) do\n#{body}\n  end\n"
      end)

    follow_on = """
      def case_#{length(@cases) + 1}(_t) do
        mixed = Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))
        Nx.multiply(Nx.exp(mixed), Nx.iota({1}))
      end

      def calls_combine, do: combine(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))
      def calls_wrap, do: wrap(Nx.iota({4}))
      def calls_inverse, do: inverse(Nx.iota({3}))
      def calls_shift_far, do: shift_far(Nx.iota({3}))
      def calls_add_iota(t), do: add_iota(Nx.backend_transfer(t, EXLA.Backend))
    """

    """
    defmodule #{inspect(@fixtures)} do
      import Nx.Defn

    #{@helpers}
    #{cases}
    #{follow_on}
    end
    """
  end
end
