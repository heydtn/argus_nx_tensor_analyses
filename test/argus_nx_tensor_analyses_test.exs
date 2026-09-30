defmodule ArgusNxTensorAnalysesTest do
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.{EMLX, TensorShapes}

  test "the default analyses leave out the one about EMLX" do
    assert ArgusNxTensorAnalyses.default_analyses() == [TensorShapes]
    assert ArgusNxTensorAnalyses.select([:default]) == {:ok, [TensorShapes]}
  end

  test "a selection takes names and sets, in the analyses' order" do
    assert ArgusNxTensorAnalyses.select([:all]) == {:ok, [TensorShapes, EMLX]}
    assert ArgusNxTensorAnalyses.select([:tensor_emlx]) == {:ok, [EMLX]}
    assert ArgusNxTensorAnalyses.select([:tensor_emlx, :default]) == {:ok, [TensorShapes, EMLX]}
    assert ArgusNxTensorAnalyses.select([]) == {:ok, []}
  end

  test "a selection naming an analysis this package lacks is refused" do
    assert {:error, message} = ArgusNxTensorAnalyses.select([:tensor_exla, :default])
    assert message =~ ":tensor_exla"
    assert message =~ ":tensor_emlx"
  end
end
