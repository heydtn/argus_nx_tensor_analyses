defmodule ArgusNxTensorAnalyses.CategoriesTest do
  # Every kind of finding the rules can emit, read from them as they are,
  # is in exactly one category, and a category holds no other kind but a
  # report relation whose rows carry none, which is a kind of its own.
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.Categories
  alias ArgusNxTensorAnalyses.EMLX
  alias ArgusNxTensorAnalyses.EmittedKinds
  alias ArgusNxTensorAnalyses.TensorShapes

  # The relations the rules emit the report relations' kinds from.
  @kind_relations ~w(call_error nonfinite unchecked violation misalignment tensor_type_error
    emlx_divergence)

  test "each kind the rules emit is in exactly one category" do
    emitted = emitted_kinds()

    assert emitted != [], "the rules name no kind: read their kinds anew"

    uncategorized =
      for kind <- emitted,
          categories = Enum.filter(Categories.names(), &(kind in Categories.kinds(&1))),
          length(categories) != 1,
          do: {kind, categories}

    assert uncategorized == []
  end

  test "a category holds no kind the rules do not emit" do
    held = for category <- Categories.names(), kind <- Categories.kinds(category), do: kind

    assert held -- (emitted_kinds() ++ kindless_relations()) == []
  end

  # A finding's rows are the rows its relation's key joins, and its
  # frames the evidence rows its join takes: where both hold the kind,
  # they are in the finding's category.
  test "a finding's rows and frames are of its kind" do
    unjoined =
      for %{fields: fields} = relation <- report_relations(),
          List.keymember?(fields, :kind, 0),
          joined = Map.get(relation, :key) || relation.evidence.on,
          :kind not in joined,
          do: relation.name

    assert unjoined == []
  end

  test "split/2 puts each row in its kind's category, in order" do
    unknown_option = ["M:f/0#1", "M:f/0", "Nx.sum/2", "unknown_option", "axis", "", ""]
    reused_key = ["M:f/0#2", "M:f/0", "Nx.Random.uniform/1", "reused_key", "key", "", ""]
    option_form = ["M:f/0#3", "M:f/0", "Nx.sum/2", "option_form", "axes", "", ""]

    rows = %{
      "tensor_call_error" => [unknown_option, reused_key, option_form],
      "violation" => [["M:f/0#4", "broadcast"]]
    }

    assert Categories.split(TensorShapes, rows) == [
             nx_options: %{"tensor_call_error" => [unknown_option, option_form]},
             nx_random: %{"tensor_call_error" => [reused_key]}
           ]
  end

  test "split/2 puts a row with no kind in its relation's category" do
    mixed = ["M:f/0#1", "M:f/0", "Nx.add/2", "EXLA.Backend", "0" | List.duplicate("", 6)]

    assert Categories.split(EMLX, %{"tensor_emlx_mixed_backends" => [mixed]}) == [
             emlx: %{"tensor_emlx_mixed_backends" => [mixed]}
           ]
  end

  defp emitted_kinds do
    @kind_relations
    |> EmittedKinds.emitted()
    |> Map.values()
    |> Enum.concat()
  end

  # The report relations whose rows carry no kind.
  defp kindless_relations do
    for %{name: name, fields: fields} <- report_relations(),
        not List.keymember?(fields, :kind, 0),
        do: Atom.to_string(name)
  end

  defp report_relations,
    do: Enum.uniq(TensorShapes.output_relations() ++ EMLX.output_relations())
end
