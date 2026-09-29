defmodule ArgusNxTensorAnalyses.TensorShapes.ShapeFlow do
  @moduledoc """
  Where the values each function hands on come from: the per-function half
  of the tensor shape analysis. `priv/tensor_shapes.dl` chains these
  summaries across functions and computes shapes over them.

  A value is a set of sources:

    * `param` — the function's parameter at a position;
    * `result` — what the call at a site returned, whatever it calls;
    * `literal` — a literal, spelled as `Argus.Extractor.Terms.spell/1`
      spells it, the empty list as `[]`;
    * `object` — a term this function built, named by the instruction
      that built it;
    * `load` — a field read from a term, named by the read.

  The facts:

    * `flow_argument(site, func, position, source_kind, source)` — the
      call at `site` may be handed the source as its argument at
      `position`;
    * `flow_dynamic(site, func, arity)` — the call at `site` calls a
      value: a fun (`call_fun`) or a module and function (`apply`);
    * `flow_callee(site, func, role, source_kind, source)` — what such a
      call calls: its `fun`, or the `module` and `function` it applies;
    * `flow_return(func, at, source_kind, source)` — `func` may return
      the source from the instruction `at`, a tail call's result as a
      `result` source;
    * `flow_guarded(at, func, node)` — every path to the call or return
      `at` passes the tests of `node` (none for `""`);
    * `flow_guard_node(node, func, parent, source_kind, source, literal,
      polarity)` — the tests of `node` are its parent's and this one: the
      source was (`eq`) or was not (`ne`) the literal, as `select_val` and
      `is_eq_exact` test it;
    * `flow_object(func, object, shape, arity)` — `func` builds `object`:
      a `tuple` of `arity` elements, a `list` cell, a `map`, an
      `operation` on its operands (`flow_operation` names it), or a
      `closure` over `arity` captured values (`flow_closure` names its
      code);
    * `flow_operation(object, operator)` — the BIF an operation applies;
    * `flow_closure(object, target)` — the function a closure runs, which
      takes the captured values first;
    * `flow_field(func, object, selector, source_kind, source)` — the
      field `selector` of `object` may hold the source: `{i}` for a
      tuple's element, an operation's operand or a closure's captured
      value at position `i` (0-based), `head` and `tail` for a list
      cell's, a map's key as spelled, `*` for a key not known;
    * `flow_base(func, object, source_kind, source)` — `object` updates
      the source, whose fields it keeps where it sets none
      (`flow_sets`);
    * `flow_sets(object, selector)` — a field an update sets;
    * `flow_load(func, load, selector, source_kind, source)` — `load`
      reads the field `selector` of the source.

  Values are solved per function over its reaching definitions
  (`Argus.Extractor.ValueFlow`), as `Argus.Extractors.PidFlow` solves its
  own. Nothing here knows Nx: which values are tensors, and of what shape,
  is the Datalog's to say.
  """

  @behaviour Argus.Extractor

  alias Argus.Extractor.CallSites
  alias Argus.Extractor.Helpers
  alias Argus.Extractor.Terms
  alias Argus.Extractor.ValueFlow
  alias Argus.Instr
  alias Argus.InstrId

  # The BIFs whose results the shape rules may compute: integer arithmetic
  # on sizes, and the sizes of terms.
  @operations [:*, :+, :-, :div, :rem, :abs, :tuple_size, :length, :map_size]

  @impl true
  def relations,
    do: [
      :flow_argument,
      :flow_dynamic,
      :flow_callee,
      :flow_return,
      :flow_guarded,
      :flow_guard_node,
      :flow_object,
      :flow_operation,
      :flow_closure,
      :flow_field,
      :flow_base,
      :flow_sets,
      :flow_load
    ]

  @impl true
  def extract(module_data) do
    case Helpers.reaching(module_data) do
      nil ->
        %{}

      reaching ->
        reads = ValueFlow.reads_by_function(reaching)
        sites = sites_by_function(module_data)

        module_data.functions
        |> Enum.reduce(%{}, fn {:function, name, arity, entry, instructions}, facts ->
          id = InstrId.func_id(module_data.module, name, arity)

          summarize(facts, %{
            id: id,
            entry: entry,
            code: List.to_tuple(instructions),
            reads: Map.get(reads, id, %{}),
            sites: Map.get(sites, id, %{})
          })
        end)
        |> Map.new(fn {relation, rows} -> {relation, rows |> Enum.uniq() |> Enum.sort()} end)
    end
  end

  defp sites_by_function(module_data) do
    module_data
    |> CallSites.for_module()
    |> Enum.reduce(%{}, fn %{func_id: function, idx: index} = site, acc ->
      Map.update(acc, function, %{index => site}, &Map.put(&1, index, site))
    end)
  end

  defp summarize(facts, function) do
    indexes = Enum.to_list(0..(tuple_size(function.code) - 1)//1)

    {outs, nil} =
      ValueFlow.solve(indexes, function.reads, nil, fn index, outs, nil ->
        {writes(function, index, outs), nil, []}
      end)

    function = Map.put(function, :guards, path_guards(function, outs))
    Enum.reduce(indexes, facts, &emit(&2, function, &1, outs))
  end

  # ── What an instruction writes ───────────────────────────────────────

  defp writes(function, index, outs) do
    instruction = elem(function.code, index)

    if Map.has_key?(function.sites, index) or dynamic_call(instruction) != nil do
      for destination <- Instr.defs(instruction),
          do: {register(destination), token(:result, id(function, index))}
    else
      written(function, index, outs, instruction)
    end
  end

  defp written(function, index, outs, {:move, source, destination}),
    do: write(destination, value(function, index, outs, source))

  defp written(function, index, outs, {:swap, left, right}),
    do:
      write(left, value(function, index, outs, right)) ++
        write(right, value(function, index, outs, left))

  defp written(function, index, outs, {:trim, _shift, _remaining} = trim) do
    Enum.flat_map(Instr.defs(trim), fn destination ->
      write(destination, value(function, index, outs, Instr.copy_source(trim, destination)))
    end)
  end

  defp written(function, index, _outs, {:put_tuple2, destination, _elements}),
    do: write(destination, token(:object, id(function, index)))

  defp written(function, index, _outs, {:put_list, _head, _tail, destination}),
    do: write(destination, token(:object, id(function, index)))

  defp written(function, index, _outs, {operation, _fail, _source, destination, _live, _pairs})
       when operation in [:put_map_assoc, :put_map_exact],
       do: write(destination, token(:object, id(function, index)))

  defp written(
         function,
         index,
         _outs,
         {:update_record, _hint, _size, _source, destination, _updates}
       ),
       do: write(destination, token(:object, id(function, index)))

  defp written(function, index, _outs, {:make_fun3, _target, _index, _uniq, destination, _env}),
    do: write(destination, token(:object, id(function, index)))

  defp written(function, index, _outs, {:get_tuple_element, _source, position, destination}),
    do: write(destination, load_token(function, index, "{#{position}}"))

  defp written(function, index, _outs, {:get_map_elements, _fail, _source, {:list, pairs}}) do
    pairs
    |> Enum.chunk_every(2)
    |> Enum.flat_map(fn [key, destination] ->
      write(destination, load_token(function, index, selector(key)))
    end)
  end

  defp written(function, index, _outs, {:get_list, _source, head, tail}),
    do:
      write(head, load_token(function, index, "head")) ++
        write(tail, load_token(function, index, "tail"))

  defp written(function, index, _outs, {:get_hd, _source, head}),
    do: write(head, load_token(function, index, "head"))

  defp written(function, index, _outs, {:get_tl, _source, tail}),
    do: write(tail, load_token(function, index, "tail"))

  defp written(function, index, _outs, {:bif, name, _fail, arguments, destination}),
    do: bif_written(function, index, name, arguments, destination)

  defp written(function, index, _outs, {:gc_bif, name, _fail, _live, arguments, destination}),
    do: bif_written(function, index, name, arguments, destination)

  defp written(_function, _index, _outs, _instruction), do: []

  defp bif_written(function, index, name, arguments, destination) do
    case bif_read(name, arguments) do
      {:load, _term, selector} -> write(destination, load_token(function, index, selector))
      :operation -> write(destination, token(:object, id(function, index)))
      :none -> []
    end
  end

  # What a BIF reads: a field of a term, or an operation on its operands.
  defp bif_read(:element, [{:integer, position}, term]), do: {:load, term, "{#{position - 1}}"}
  defp bif_read(:map_get, [key, term]), do: {:load, term, selector(key)}
  defp bif_read(:hd, [term]), do: {:load, term, "head"}
  defp bif_read(:tl, [term]), do: {:load, term, "tail"}
  defp bif_read(name, _arguments) when name in @operations, do: :operation
  defp bif_read(_name, _arguments), do: :none

  # ── The facts an instruction emits ───────────────────────────────────

  defp emit(facts, function, index, outs) do
    instruction = elem(function.code, index)

    case {Map.fetch(function.sites, index), dynamic_call(instruction)} do
      {{:ok, %{mfa: {_module, _name, arity}}}, _dynamic} ->
        emit_call(facts, function, index, outs, instruction, arity)

      {:error, {arity, callee}} ->
        site = id(function, index)

        callee
        |> Enum.reduce(
          facts
          |> add(:flow_dynamic, [site, function.id, Integer.to_string(arity)])
          |> emit_call(function, index, outs, instruction, arity),
          fn {role, operand}, acc ->
            sourced(
              acc,
              :flow_callee,
              [site, function.id, role],
              value(function, index, outs, operand)
            )
          end
        )

      {:error, nil} ->
        emit_instruction(facts, function, index, outs, instruction)
    end
  end

  defp emit_call(facts, function, index, outs, instruction, arity) do
    site = id(function, index)

    facts =
      0..(arity - 1)//1
      |> Enum.reduce(facts, fn position, acc ->
        sourced(
          acc,
          :flow_argument,
          [site, function.id, Integer.to_string(position)],
          value(function, index, outs, {:x, position})
        )
      end)
      |> guarded(function, index, site)

    if Instr.tail_call?(instruction),
      do: add(facts, :flow_return, [function.id, site, "result", site]),
      else: facts
  end

  # A call whose callee is a value: its arity, and the operands that name
  # what it calls, the fun it runs or the module and function it applies.
  defp dynamic_call({:call_fun, arity}), do: {arity, [{"fun", {:x, arity}}]}
  defp dynamic_call({:call_fun2, _tag, arity, fun}), do: {arity, [{"fun", fun}]}

  defp dynamic_call({:apply, arity}),
    do: {arity, [{"module", {:x, arity}}, {"function", {:x, arity + 1}}]}

  defp dynamic_call({:apply_last, arity, _deallocate}),
    do: {arity, [{"module", {:x, arity}}, {"function", {:x, arity + 1}}]}

  defp dynamic_call(_instruction), do: nil

  defp emit_instruction(facts, function, index, outs, :return) do
    at = id(function, index)

    facts
    |> sourced(:flow_return, [function.id, at], value(function, index, outs, {:x, 0}))
    |> guarded(function, index, at)
  end

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:put_tuple2, _destination, {:list, elements}}
       ) do
    object = id(function, index)

    elements
    |> Enum.with_index()
    |> Enum.reduce(
      add(facts, :flow_object, [function.id, object, "tuple", Integer.to_string(length(elements))]),
      fn {element, position}, acc ->
        sourced(
          acc,
          :flow_field,
          [function.id, object, "{#{position}}"],
          value(function, index, outs, element)
        )
      end
    )
  end

  defp emit_instruction(facts, function, index, outs, {:put_list, head, tail, _destination}) do
    object = id(function, index)

    facts
    |> add(:flow_object, [function.id, object, "list", "0"])
    |> sourced(:flow_field, [function.id, object, "head"], value(function, index, outs, head))
    |> sourced(:flow_field, [function.id, object, "tail"], value(function, index, outs, tail))
  end

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {operation, _fail, source, _destination, _live, {:list, pairs}}
       )
       when operation in [:put_map_assoc, :put_map_exact] do
    object = id(function, index)

    facts =
      facts
      |> add(:flow_object, [function.id, object, "map", "0"])
      |> sourced(:flow_base, [function.id, object], value(function, index, outs, source))

    pairs
    |> Enum.chunk_every(2)
    |> Enum.reduce(facts, fn [key, field], acc ->
      updated(acc, function, object, selector(key), value(function, index, outs, field))
    end)
  end

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:update_record, _hint, size, source, _destination, {:list, updates}}
       ) do
    object = id(function, index)

    facts =
      facts
      |> add(:flow_object, [function.id, object, "tuple", Integer.to_string(size)])
      |> sourced(:flow_base, [function.id, object], value(function, index, outs, source))

    updates
    |> Enum.chunk_every(2)
    |> Enum.reduce(facts, fn
      [{:integer, position}, field], acc ->
        updated(acc, function, object, "{#{position - 1}}", value(function, index, outs, field))

      _other, acc ->
        acc
    end)
  end

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:make_fun3, {module, name, arity}, _index, _uniq, _destination, {:list, env}}
       ) do
    object = id(function, index)

    env
    |> Enum.with_index()
    |> Enum.reduce(
      facts
      |> add(:flow_object, [function.id, object, "closure", Integer.to_string(length(env))])
      |> add(:flow_closure, [object, InstrId.func_id(module, name, arity)]),
      fn {captured, position}, acc ->
        sourced(
          acc,
          :flow_field,
          [function.id, object, "{#{position}}"],
          value(function, index, outs, captured)
        )
      end
    )
  end

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:get_tuple_element, source, position, _destination}
       ),
       do: loaded(facts, function, index, "{#{position}}", value(function, index, outs, source))

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:get_map_elements, _fail, source, {:list, pairs}}
       ) do
    term = value(function, index, outs, source)

    pairs
    |> Enum.chunk_every(2)
    |> Enum.reduce(facts, fn [key, _destination], acc ->
      loaded(acc, function, index, selector(key), term)
    end)
  end

  defp emit_instruction(facts, function, index, outs, {:get_list, source, _head, _tail}) do
    term = value(function, index, outs, source)

    facts
    |> loaded(function, index, "head", term)
    |> loaded(function, index, "tail", term)
  end

  defp emit_instruction(facts, function, index, outs, {:get_hd, source, _head}),
    do: loaded(facts, function, index, "head", value(function, index, outs, source))

  defp emit_instruction(facts, function, index, outs, {:get_tl, source, _tail}),
    do: loaded(facts, function, index, "tail", value(function, index, outs, source))

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:bif, name, _fail, arguments, _destination}
       ),
       do: emit_bif(facts, function, index, outs, name, arguments)

  defp emit_instruction(
         facts,
         function,
         index,
         outs,
         {:gc_bif, name, _fail, _live, arguments, _destination}
       ),
       do: emit_bif(facts, function, index, outs, name, arguments)

  defp emit_instruction(facts, _function, _index, _outs, _instruction), do: facts

  defp emit_bif(facts, function, index, outs, name, arguments) do
    case bif_read(name, arguments) do
      {:load, term, selector} ->
        loaded(facts, function, index, selector, value(function, index, outs, term))

      :operation ->
        object = id(function, index)

        facts =
          facts
          |> add(:flow_object, [
            function.id,
            object,
            "operation",
            Integer.to_string(length(arguments))
          ])
          |> add(:flow_operation, [object, Atom.to_string(name)])

        arguments
        |> Enum.with_index()
        |> Enum.reduce(facts, fn {argument, position}, acc ->
          sourced(
            acc,
            :flow_field,
            [function.id, object, "{#{position}}"],
            value(function, index, outs, argument)
          )
        end)

      :none ->
        facts
    end
  end

  # The tests on the path to `at`, sorted, as a chain of nodes: calls on
  # one branch pass the same tests and share a node, and so do the tests
  # two branches share before they part.
  defp guarded(facts, function, index, at) do
    {facts, node} =
      function.guards
      |> Map.get(index, MapSet.new())
      |> Enum.sort()
      |> Enum.reduce({facts, ""}, fn {kind, source, literal, polarity} = test, {acc, parent} ->
        node = guard_node(function, parent, test)
        row = [node, function.id, parent, kind, source, literal, polarity]
        {add(acc, :flow_guard_node, row), node}
      end)

    add(facts, :flow_guarded, [at, function.id, node])
  end

  defp guard_node(function, parent, test) do
    digest =
      :sha256
      |> :crypto.hash(:erlang.term_to_binary({parent, test}))
      |> binary_part(0, 8)
      |> Base.encode16(case: :lower)

    function.id <> "?" <> digest
  end

  defp updated(facts, function, object, selector, value) do
    facts = sourced(facts, :flow_field, [function.id, object, selector], value)
    if selector == "*", do: facts, else: add(facts, :flow_sets, [object, selector])
  end

  defp loaded(facts, function, index, selector, term),
    do:
      sourced(
        facts,
        :flow_load,
        [function.id, load_id(function, index, selector), selector],
        term
      )

  # One row per source of the value, the columns before it given.
  defp sourced(facts, relation, columns, value) do
    Enum.reduce(value, facts, fn {kind, source}, acc ->
      add(acc, relation, columns ++ [Atom.to_string(kind), source])
    end)
  end

  defp add(facts, relation, row), do: Map.update(facts, relation, [row], &[row | &1])

  # ── The literal tests a path passes ──────────────────────────────────

  # The tests every path from the function's entry to an instruction
  # passes, as `{kind, source, literal, polarity}`: the source was (`eq`)
  # or was not (`ne`) the literal. A test of a value that several sources
  # may give proves nothing about any one of them.
  defp path_guards(function, outs) do
    labels =
      for index <- 0..(tuple_size(function.code) - 1)//1,
          {:label, label} <- [elem(function.code, index)],
          into: %{},
          do: {label, index}

    case Map.fetch(labels, function.entry) do
      {:ok, start} -> spread(function, outs, labels, [start], %{start => MapSet.new()})
      :error -> %{}
    end
  end

  # What holds at an instruction is what holds on every edge into it, so a
  # revisit only ever drops tests.
  defp spread(_function, _outs, _labels, [], held), do: held

  defp spread(function, outs, labels, [index | queue], held) do
    {held, queue} =
      function
      |> edges(outs, labels, index, Map.fetch!(held, index))
      |> Enum.reduce({held, queue}, &meet/2)

    spread(function, outs, labels, queue, held)
  end

  # An edge's tests meet what holds at its target, which is visited again
  # when that drops any.
  defp meet({target, tests}, {held, queue}) do
    case Map.fetch(held, target) do
      :error ->
        {Map.put(held, target, tests), [target | queue]}

      {:ok, before} ->
        kept = MapSet.intersection(before, tests)

        if kept == before,
          do: {held, queue},
          else: {Map.put(held, target, kept), [target | queue]}
    end
  end

  defp edges(function, outs, labels, index, tests) do
    instruction = elem(function.code, index)

    instruction
    |> branches(function, outs, labels, index, tests)
    |> Enum.filter(fn {target, _tests} -> target != nil end)
  end

  defp branches(
         {:select_val, operand, {:f, fail}, {:list, pairs}},
         function,
         outs,
         labels,
         index,
         tests
       ) do
    choices = Enum.chunk_every(pairs, 2)
    tested = tested(function, outs, index, operand)

    otherwise =
      Enum.reduce(choices, tests, fn [choice, _label], acc ->
        with_test(acc, tested, choice, "ne")
      end)

    # A choice's label is also reached as none of the choices that go
    # elsewhere, which is what survives where several choices share it.
    [
      {Map.get(labels, fail), otherwise}
      | for [choice, {:f, label}] <- choices do
          elsewhere =
            for [other, {:f, target}] <- choices, target != label, reduce: tests do
              acc -> with_test(acc, tested, other, "ne")
            end

          {Map.get(labels, label), with_test(elsewhere, tested, choice, "eq")}
        end
    ]
  end

  defp branches({:test, test, {:f, fail}, [left, right]}, function, outs, labels, index, tests)
       when test in [:is_eq_exact, :is_eq, :is_ne_exact, :is_ne] do
    {passed, failed} = if test in [:is_eq_exact, :is_eq], do: {"eq", "ne"}, else: {"ne", "eq"}

    {tested, literal} =
      case {tested(function, outs, index, left), tested(function, outs, index, right)} do
        {nil, nil} -> {nil, nil}
        {nil, token} -> {token, left}
        {token, _other} -> {token, right}
      end

    [
      {index + 1, with_test(tests, tested, literal, passed)},
      {Map.get(labels, fail), with_test(tests, tested, literal, failed)}
    ]
  end

  defp branches(instruction, function, _outs, labels, index, tests) do
    jumps = for label <- Instr.targets(instruction), do: {Map.get(labels, label), tests}

    if Instr.falls_through?(instruction) and index + 1 < tuple_size(function.code),
      do: [{index + 1, tests} | jumps],
      else: jumps
  end

  # The one source an operand's value comes from, other than a literal.
  defp tested(function, outs, index, operand) do
    case function |> value(index, outs, operand) |> MapSet.to_list() do
      [{kind, source}] when kind != :literal -> {Atom.to_string(kind), source}
      _several -> nil
    end
  end

  defp with_test(tests, nil, _literal, _polarity), do: tests

  defp with_test(tests, {kind, source}, literal, polarity) do
    case spell(literal) do
      nil -> tests
      spelled -> MapSet.put(tests, {kind, source, spelled, polarity})
    end
  end

  # ── Values ───────────────────────────────────────────────────────────

  # What an operand holds at an instruction: the sources of the writes that
  # reach a register's read, a parameter where the function's entry does,
  # or the literal an operand spells.
  defp value(function, index, outs, operand) do
    case Instr.register(operand) do
      {file, number} when file in [:x, :y] ->
        register = "#{file}#{number}"

        function.reads
        |> Map.get(index, %{})
        |> Map.get(register, [])
        |> Enum.reduce(MapSet.new(), fn
          {:param, position}, acc ->
            MapSet.put(acc, {:param, Integer.to_string(position)})

          {:def, definition}, acc ->
            MapSet.union(acc, Map.get(outs, {definition, register}, MapSet.new()))
        end)

      literal ->
        case spell(literal) do
          nil -> MapSet.new()
          spelled -> token(:literal, spelled)
        end
    end
  end

  defp spell({:integer, integer}), do: Integer.to_string(integer)
  defp spell({:atom, atom}), do: inspect(atom)
  defp spell({:float, float}), do: Float.to_string(float)
  defp spell({:literal, term}), do: Terms.spell(term)
  defp spell(nil), do: "[]"
  defp spell(_other), do: nil

  # A map key as a field's name: the literal spelled, or `*` for one only
  # known at run time.
  defp selector(key) do
    case Instr.register(key) do
      {file, _number} when file in [:x, :y] -> "*"
      literal -> spell(literal) || "*"
    end
  end

  defp write(operand, value) do
    case register(operand) do
      nil -> []
      register -> [{register, value}]
    end
  end

  defp register(operand) do
    case Instr.register(operand) do
      {file, number} when file in [:x, :y] -> "#{file}#{number}"
      _other -> nil
    end
  end

  defp token(kind, source), do: MapSet.new([{kind, source}])

  defp load_token(function, index, selector), do: token(:load, load_id(function, index, selector))

  defp load_id(function, index, selector), do: id(function, index) <> " " <> selector

  defp id(function, index), do: InstrId.mint(function.id, index)
end
