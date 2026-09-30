defmodule ArgusNxTensorAnalyses.TensorShapes.WordingTest do
  # Every kind of finding the rules can emit, read from them as they are,
  # is worded by exactly one wording module: the one that answers it for a
  # detail, cause and operation made up here, which only a module that
  # words the kind whatever they are does. Each is documented in
  # docs/checks.md too, which documents no other.
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.DocumentedKinds
  alias ArgusNxTensorAnalyses.EmittedKinds
  alias ArgusNxTensorAnalyses.TensorShapes.Wording

  # What a wording module words each relation's kinds with.
  @families %{
    "call_error" => :call_error,
    "nonfinite" => :hazard,
    "unchecked" => :hazard,
    "violation" => :violation,
    "misalignment" => :violation,
    "tensor_type_error" => :type_error
  }

  test "each kind the rules emit is worded by exactly one module" do
    emitted = EmittedKinds.emitted(Map.keys(@families))

    assert emitted |> Map.values() |> Enum.concat() != [],
           "the rules name no kind: read their kinds anew"

    unworded =
      for {relation, kinds} <- emitted,
          kind <- kinds,
          modules = Enum.filter(Wording.modules(), &answers?(&1, @families[relation], kind)),
          length(modules) != 1,
          do: {relation, kind, modules}

    assert unworded == []
  end

  test "docs/checks.md documents each kind the rules emit, and no other" do
    emitted =
      (Map.keys(@families) ++ ["emlx_divergence"])
      |> EmittedKinds.emitted()
      |> Map.values()
      |> Enum.concat()
      |> MapSet.new()

    documented = MapSet.new(DocumentedKinds.documented())

    assert emitted |> MapSet.difference(documented) |> Enum.sort() == []

    # Mixed backends are a relation of their own, with backends in place
    # of a kind, and documented under the relation's name.
    assert documented
           |> MapSet.difference(emitted)
           |> MapSet.delete("tensor_emlx_mixed_backends")
           |> Enum.sort() == []
  end

  # A module answers a kind with a wording, or by raising for the detail
  # made up here, which only its clause for the kind reads.
  defp answers?(module, family, kind) do
    ask(module, family, kind) != nil
  rescue
    _error -> true
  end

  defp ask(module, :call_error, kind), do: module.call_error(kind, "", "")
  defp ask(module, :hazard, kind), do: module.hazard(kind, "", "")
  defp ask(module, :violation, kind), do: module.violation(kind)
  defp ask(module, :type_error, kind), do: module.type_error(kind, "", "", "0", "1")
end
