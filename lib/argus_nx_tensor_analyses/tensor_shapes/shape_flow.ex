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
    * `flow_checked(site, func, position, literal, polarity)` — every path
      to the call at `site` tests its argument at `position` against the
      literal: it was (`eq`) or was not (`ne`) the literal, or was below
      (`lt`), at most (`le`), above (`gt`) or at least (`ge`) it. The test
      read the same value as the argument, whatever its sources;
    * `flow_object(func, object, shape, arity)` — `func` builds `object`:
      a `tuple` of `arity` elements, a `list` cell, a `map`, an
      `operation` on its operands (`flow_operation` names it), or a
      `closure` over `arity` captured values (`flow_closure` names its
      code);
    * `flow_operation(object, operator)` — the BIF an operation applies;
    * `flow_closure(object, target)` — the function a closure runs, which
      takes the arguments it is called with first, then the captured
      values;
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
  own, and the tests on its paths over its control-flow graph
  (`Argus.Cfg`). Nothing here knows Nx: which values are tensors, and of
  what shape, is the Datalog's to say.
  """

  @behaviour Argus.Extractor

  alias Argus.Extractor.CallSites
  alias Argus.Extractor.Facts
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
      :flow_checked,
      :flow_object,
      :flow_operation,
      :flow_closure,
      :flow_field,
      :flow_base,
      :flow_sets,
      :flow_load,
      :flow_operand,
      :flow_literal_element,
      :flow_next
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
        |> Enum.reduce(%{}, fn {:function, name, arity, _entry, instructions}, facts ->
          id = InstrId.func_id(module_data.module, name, arity)

          summarize(facts, %{
            id: id,
            code: List.to_tuple(instructions),
            cfg: Helpers.cfg(module_data, name, arity),
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
    facts = Enum.reduce(indexes, facts, &emit(&2, function, &1, outs))

    indexes
    |> Enum.reduce(facts, &emit_operands(&2, function, &1, outs))
    |> literal_elements(function)
    |> call_order(function)
  end

  # ── What an instruction writes ───────────────────────────────────────
  #
  # `Argus.Instr` says which registers an instruction reads and writes, and
  # copies (`copy_source/2`) are read through it. What the clauses below
  # add is what it does not say: the term an instruction builds or the
  # field it reads, which is what a value here is made of.

  defp writes(function, index, outs) do
    instruction = elem(function.code, index)

    if Map.has_key?(function.sites, index) or dynamic_call(instruction) != nil do
      for destination <- Instr.defs(instruction),
          do: {register(destination), token(:result, id(function, index))}
    else
      case copied(function, index, outs, instruction) do
        [] -> written(function, index, outs, instruction)
        copies -> copies
      end
    end
  end

  # What a copy (`move`, `swap`, `trim`) writes: each register it copies
  # into holds what its source did.
  defp copied(function, index, outs, instruction) do
    for destination <- Instr.defs(instruction),
        source = Instr.copy_source(instruction, destination),
        source != nil,
        copy <- write(destination, value(function, index, outs, source)),
        do: copy
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
          |> Facts.add_fact(:flow_dynamic, [site, function.id, Integer.to_string(arity)])
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
      |> checked(function, index, outs, site, arity)

    if Instr.tail_call?(instruction),
      do: Facts.add_fact(facts, :flow_return, [function.id, site, "result", site]),
      else: facts
  end

  # A call whose callee is a value: its arity, and the operands that name
  # what it calls, the fun it runs or the module and function it applies.
  # `Argus.Instr.uses/1` lists these operands among the arguments; which
  # of them is the callee is read here.
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
      Facts.add_fact(facts, :flow_object, [
        function.id,
        object,
        "tuple",
        Integer.to_string(length(elements))
      ]),
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
    |> Facts.add_fact(:flow_object, [function.id, object, "list", "0"])
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
      |> Facts.add_fact(:flow_object, [function.id, object, "map", "0"])
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
      |> Facts.add_fact(:flow_object, [function.id, object, "tuple", Integer.to_string(size)])
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
      |> Facts.add_fact(:flow_object, [
        function.id,
        object,
        "closure",
        Integer.to_string(length(env))
      ])
      |> Facts.add_fact(:flow_closure, [object, InstrId.func_id(module, name, arity)]),
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
          |> Facts.add_fact(:flow_object, [
            function.id,
            object,
            "operation",
            Integer.to_string(length(arguments))
          ])
          |> Facts.add_fact(:flow_operation, [object, Atom.to_string(name)])

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
      for(
        {:guard, kind, source, literal, polarity} <- held_at(function, index),
        do: {kind, source, literal, polarity}
      )
      |> Enum.sort()
      |> Enum.reduce({facts, ""}, fn {kind, source, literal, polarity} = test, {acc, parent} ->
        node = guard_node(function, parent, test)
        row = [node, function.id, parent, kind, source, literal, polarity]
        {Facts.add_fact(acc, :flow_guard_node, row), node}
      end)

    Facts.add_fact(facts, :flow_guarded, [at, function.id, node])
  end

  # The tests on the path to the call that read the value of one of its
  # arguments.
  defp checked(facts, function, index, outs, site, arity) do
    tests = held_at(function, index)

    for position <- 0..(arity - 1)//1,
        argument = value(function, index, outs, {:x, position}),
        {:check, ^argument, literal, polarity} <- tests,
        reduce: facts do
      acc ->
        Facts.add_fact(acc, :flow_checked, [
          site,
          function.id,
          Integer.to_string(position),
          literal,
          polarity
        ])
    end
  end

  defp held_at(function, index), do: Map.get(function.guards, index, MapSet.new())

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
    if selector == "*", do: facts, else: Facts.add_fact(facts, :flow_sets, [object, selector])
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
      Facts.add_fact(acc, relation, columns ++ [Atom.to_string(kind), source])
    end)
  end

  # ── The literal tests a path passes ──────────────────────────────────

  # The tests every path from the function's entry to an instruction
  # passes, each as a check of the value it read, `{:check, value,
  # literal, polarity}`. A check of a value one source gives, that it was
  # (`eq`) or was not (`ne`) the literal, is a guard of that source too,
  # `{:guard, kind, source, literal, polarity}`: a test of a value that
  # several sources may give proves nothing about any one of them.
  #
  # A forward dataflow over the function's control-flow graph
  # (`Argus.Cfg`), meeting on every edge into a block. A test ends its
  # block, so what holds on a block's entry holds at each of its
  # instructions.
  defp path_guards(%{cfg: nil}, _outs), do: %{}

  defp path_guards(%{cfg: cfg} = function, outs) do
    for {block, tests} <- held(function, outs, [cfg.entry], %{cfg.entry => MapSet.new()}),
        %{range: {first, last}} = Map.fetch!(cfg.blocks, block),
        index <- first..last,
        into: %{},
        do: {index, tests}
  end

  # What holds on a block's entry is what holds on every edge into it, so a
  # revisit only ever drops tests.
  defp held(_function, _outs, [], held), do: held

  defp held(function, outs, [block | queue], held) do
    %{range: {_first, last}, succs: succs} = Map.fetch!(function.cfg.blocks, block)
    tests = Map.fetch!(held, block)

    {held, queue} =
      Enum.reduce(succs, {held, queue}, fn {successor, kind}, acc ->
        meet({successor, edge_tests(function, outs, last, kind, successor, tests)}, acc)
      end)

    held(function, outs, queue, held)
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

  # What holds on the edge of `kind` to `successor` out of the block the
  # instruction at `index` ends.
  defp edge_tests(function, outs, index, kind, successor, tests),
    do: edge_tests(elem(function.code, index), function, outs, index, kind, successor, tests)

  # A select's default passes none of its choices. An arm passes its
  # choice where it is the only one into its block, and fails the choices
  # that go elsewhere.
  defp edge_tests(
         {:select_val, operand, _fail, {:list, pairs}},
         function,
         outs,
         index,
         kind,
         successor,
         tests
       ) do
    choices = for [choice, {:f, label}] <- Enum.chunk_every(pairs, 2), do: {choice, label}
    tested = tested(function, outs, index, operand)

    case kind do
      :select_fail ->
        Enum.reduce(choices, tests, fn {choice, _label}, acc ->
          with_test(acc, tested, choice, "ne")
        end)

      {:select_arm, _spelled} ->
        {here, elsewhere} =
          Enum.split_with(choices, fn {_choice, label} ->
            Map.get(function.cfg.labels, label) == successor
          end)

        failed =
          Enum.reduce(elsewhere, tests, fn {choice, _label}, acc ->
            with_test(acc, tested, choice, "ne")
          end)

        case here do
          [{choice, _label}] -> with_test(failed, tested, choice, "eq")
          _several -> failed
        end

      _other ->
        tests
    end
  end

  defp edge_tests(
         {:test, test, _fail, [left, right]},
         function,
         outs,
         index,
         kind,
         _successor,
         tests
       )
       when test in [:is_eq_exact, :is_eq, :is_ne_exact, :is_ne] and
              kind in [:branch_pass, :branch_fail] do
    {passed, failed} = if test in [:is_eq_exact, :is_eq], do: {"eq", "ne"}, else: {"ne", "eq"}

    {tested, literal} =
      case {tested(function, outs, index, left), tested(function, outs, index, right)} do
        {nil, nil} -> {nil, nil}
        {nil, token} -> {token, left}
        {token, _other} -> {token, right}
      end

    with_test(tests, tested, literal, if(kind == :branch_pass, do: passed, else: failed))
  end

  # An order test against a literal: `is_lt(a, b)` passes where a < b and
  # fails where a >= b, `is_ge` the other way round. The test is kept on
  # the value's side: `lt`, `le`, `gt` or `ge` of the literal. The rules
  # read it only as a check.
  defp edge_tests(
         {:test, test, _fail, [left, right]},
         function,
         outs,
         index,
         kind,
         _successor,
         tests
       )
       when test in [:is_lt, :is_ge] and kind in [:branch_pass, :branch_fail] do
    below = test == :is_lt == (kind == :branch_pass)

    case {tested(function, outs, index, left), tested(function, outs, index, right)} do
      {nil, nil} -> tests
      {nil, token} -> with_test(tests, token, left, if(below, do: "gt", else: "le"))
      {token, _other} -> with_test(tests, token, right, if(below, do: "lt", else: "ge"))
    end
  end

  defp edge_tests(_instruction, _function, _outs, _index, _kind, _successor, tests), do: tests

  # The value a register operand holds, none for a literal.
  defp tested(function, outs, index, operand) do
    if register(operand), do: value(function, index, outs, operand)
  end

  defp with_test(tests, nil, _literal, _polarity), do: tests

  defp with_test(tests, value, literal, polarity) do
    case spell(literal) do
      nil ->
        tests

      spelled ->
        tests
        |> MapSet.put({:check, value, spelled, polarity})
        |> with_guard(value, spelled, polarity)
    end
  end

  defp with_guard(tests, value, spelled, polarity) when polarity in ["eq", "ne"] do
    case MapSet.to_list(value) do
      [{kind, source}] when kind != :literal ->
        MapSet.put(tests, {:guard, Atom.to_string(kind), source, spelled, polarity})

      _several ->
        tests
    end
  end

  defp with_guard(tests, _value, _spelled, _polarity), do: tests

  # ── Values ───────────────────────────────────────────────────────────

  # What an operand holds at an instruction: the sources of the writes that
  # reach a register's read, a parameter where the function's entry does,
  # or the literal an operand spells.
  defp value(function, index, outs, operand) do
    case register(operand) do
      nil ->
        literal(operand)

      register ->
        function.reads
        |> ValueFlow.inputs(outs, index, &{:param, Integer.to_string(&1)})
        |> Map.get(register, MapSet.new())
    end
  end

  defp literal(operand) do
    case operand |> Instr.register() |> spell() do
      nil -> MapSet.new()
      spelled -> token(:literal, spelled)
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

  # ── The operands of Elixir's operators and tests ─────────────────────
  #
  # `flow_operand(site, func, operator, position, source_kind, source)`:
  # the instruction at `site` applies an Elixir operator or test to the
  # source as its operand at `position`, a literal operand as a `literal`
  # source. The operator is an arithmetic, comparison or boolean BIF
  # outside a guard (where it would fail the guard rather than raise), by
  # its name (`*`, `<`, `not`); `fconv`, which takes a term into float
  # arithmetic (`/`, and the other operators on floats); a test that
  # compares two terms, by the test's name (`is_lt`, `is_eq_exact`); or
  # `select_val`, whose value is its operand 0 and whose choices, in order,
  # the literals after it.

  @host_operators [:+, :-, :*, :/, :div, :rem, :abs, :float, :trunc, :round, :ceil, :floor] ++
                    [:bnot, :band, :bor, :bxor, :bsl, :bsr, :not, :and, :or, :xor] ++
                    [:<, :>, :"=<", :>=, :==, :"/=", :"=:=", :"=/=", :min, :max]

  @comparison_tests [:is_lt, :is_ge, :is_eq, :is_ne, :is_eq_exact, :is_ne_exact]

  defp emit_operands(facts, function, index, outs),
    do: operands(facts, function, index, outs, elem(function.code, index))

  defp operands(facts, function, index, outs, {:bif, name, {:f, 0}, arguments, _destination})
       when name in @host_operators,
       do: operand_rows(facts, function, index, outs, Atom.to_string(name), arguments)

  defp operands(
         facts,
         function,
         index,
         outs,
         {:gc_bif, name, {:f, 0}, _live, arguments, _destination}
       )
       when name in @host_operators,
       do: operand_rows(facts, function, index, outs, Atom.to_string(name), arguments)

  defp operands(facts, function, index, outs, {:fconv, source, _destination}),
    do: operand_rows(facts, function, index, outs, "fconv", [source])

  defp operands(facts, function, index, outs, {:test, test, _fail, [_left, _right] = arguments})
       when test in @comparison_tests,
       do: operand_rows(facts, function, index, outs, Atom.to_string(test), arguments)

  defp operands(facts, function, index, outs, {:select_val, operand, _fail, {:list, pairs}}) do
    choices = for [choice, _label] <- Enum.chunk_every(pairs, 2), do: choice
    operand_rows(facts, function, index, outs, "select_val", [operand | choices])
  end

  defp operands(facts, _function, _index, _outs, _instruction), do: facts

  defp operand_rows(facts, function, index, outs, operator, arguments) do
    site = id(function, index)

    arguments
    |> Enum.with_index()
    |> Enum.reduce(facts, fn {argument, position}, acc ->
      sourced(
        acc,
        :flow_operand,
        [site, function.id, operator, Integer.to_string(position)],
        value(function, index, outs, argument)
      )
    end)
  end

  # ── The elements of literal lists ────────────────────────────────────
  #
  # `flow_literal_element(literal, index, element)`: the literal, spelled
  # as a `literal` source spells it, is a proper list whose element at
  # `index` (0-based) is `element`, spelled the same way. For every literal
  # list an instruction takes: the compiler folds the constant rest of a
  # list the code builds into one (`[x, 0.0, 1.0]` is `x` in front of
  # `[0.0, 1.0]`), and a list of constants into a single literal, whose
  # elements a rule may read one at a time.

  defp literal_elements(facts, function) do
    function.code
    |> Tuple.to_list()
    |> Enum.flat_map(&literal_lists/1)
    |> Enum.reduce(facts, fn list, acc ->
      spelled = Terms.spell(list)

      list
      |> Enum.with_index()
      |> Enum.reduce(acc, fn {element, index}, rows ->
        Facts.add_fact(rows, :flow_literal_element, [
          spelled,
          Integer.to_string(index),
          Terms.spell(element)
        ])
      end)
    end)
  end

  # The literal lists an instruction's operands hold, but `[]`.
  defp literal_lists({:literal, [_head | _tail] = list}),
    do: if(Terms.proper_list?(list), do: [list], else: [])

  defp literal_lists({:literal, _term}), do: []

  defp literal_lists(operands) when is_tuple(operands),
    do: operands |> Tuple.to_list() |> Enum.flat_map(&literal_lists/1)

  defp literal_lists(operands) when is_list(operands),
    do: operands |> Terms.list_elements() |> Enum.flat_map(&literal_lists/1)

  defp literal_lists(_operand), do: []

  # ── The order of calls and returns ───────────────────────────────────
  #
  # `flow_next(at, func, next)`: control can pass from the call at `at` to
  # the call or return at `next` with no call or return between them, in
  # one trip through `func`. The calls are the sites the facts above name
  # (a function's, or a value's through `call_fun` or `apply`), and the
  # returns the `return` instructions `flow_return` names, so a rule asks
  # whether a use of a value comes after another in the words the value
  # flow uses. The closure of this relation is the order between them: a
  # rule takes it from the few calls it asks about.
  #
  # A trip follows `Argus.Cfg.Function.forward_succs/2`: an edge into a
  # block that dominates its source closes a loop and is no flow. The BEAM
  # loops within a function only in a receive; elsewhere each trip of a
  # loop is a call of its own.

  defp call_order(facts, %{cfg: nil}), do: facts

  defp call_order(facts, function) do
    blocks =
      0..(tuple_size(function.code) - 1)//1
      |> Enum.filter(&ordered_point?(function, &1))
      |> Enum.group_by(&Argus.Cfg.Function.block_at(function.cfg, &1).id)

    firsts = Map.new(blocks, fn {block, [first | _rest]} -> {block, first} end)

    Enum.reduce(blocks, facts, fn {block, points}, acc ->
      last = List.last(points)

      successors =
        function.cfg
        |> Argus.Cfg.Function.forward_succs(Map.fetch!(function.cfg.blocks, block))
        |> first_points(function.cfg, firsts, MapSet.new(), [])

      points
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.concat(for successor <- successors, do: [last, successor])
      |> Enum.reduce(acc, fn [from, to], rows ->
        Facts.add_fact(rows, :flow_next, [id(function, from), function.id, id(function, to)])
      end)
    end)
  end

  # A call or a return: the instructions the value flow names as sites
  # and returns.
  defp ordered_point?(function, index) do
    instruction = elem(function.code, index)

    Map.has_key?(function.sites, index) or dynamic_call(instruction) != nil or
      instruction == :return
  end

  # The first call or return in each block reached from `blocks` through
  # blocks holding none.
  defp first_points([], _cfg, _firsts, _seen, found), do: found

  defp first_points([block | rest], cfg, firsts, seen, found) do
    cond do
      MapSet.member?(seen, block) ->
        first_points(rest, cfg, firsts, seen, found)

      Map.has_key?(firsts, block) ->
        first_points(rest, cfg, firsts, MapSet.put(seen, block), [
          Map.fetch!(firsts, block) | found
        ])

      true ->
        cfg
        |> Argus.Cfg.Function.forward_succs(Map.fetch!(cfg.blocks, block))
        |> Enum.concat(rest)
        |> first_points(cfg, firsts, MapSet.put(seen, block), found)
    end
  end
end
