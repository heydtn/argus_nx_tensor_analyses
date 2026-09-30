defmodule ArgusNxTensorAnalysesTest do
  use ExUnit.Case, async: true

  @nx ~w(nx_shapes nx_names nx_math nx_types nx_options nx_indices nx_traced nx_gradients
    nx_containers nx_random nx_freed nx_serving)a

  test "the analyses are the categories, the Nx ones first" do
    assert ArgusNxTensorAnalyses.analyses() == @nx ++ [:emlx]
  end

  test "the default analyses leave out the one about EMLX" do
    assert ArgusNxTensorAnalyses.default_analyses() == @nx
    assert ArgusNxTensorAnalyses.select([:default]) == {:ok, @nx}
  end

  test "a selection takes names and sets, in the analyses' order" do
    assert ArgusNxTensorAnalyses.select([:all]) == {:ok, @nx ++ [:emlx]}
    assert ArgusNxTensorAnalyses.select([:emlx]) == {:ok, [:emlx]}
    assert ArgusNxTensorAnalyses.select([:emlx, :default]) == {:ok, @nx ++ [:emlx]}

    assert ArgusNxTensorAnalyses.select([:nx_serving, :nx_shapes]) ==
             {:ok, [:nx_shapes, :nx_serving]}

    assert ArgusNxTensorAnalyses.select([]) == {:ok, []}
  end

  test "a selection naming an analysis this package lacks is refused" do
    assert {:error, message} = ArgusNxTensorAnalyses.select([:tensor_shapes, :default])
    assert message =~ ":tensor_shapes"
    assert message =~ ":nx_shapes"
    assert message =~ ":emlx"
  end

  test "a run that asks for no category solves nothing" do
    assert ArgusNxTensorAnalyses.run([], []) == {:ok, %{}}
  end
end
