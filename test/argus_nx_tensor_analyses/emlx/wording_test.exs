defmodule ArgusNxTensorAnalyses.EMLX.WordingTest do
  # Every kind of divergence the rules can emit, read from them as they
  # are, has a wording of its own.
  use ExUnit.Case, async: true

  alias ArgusNxTensorAnalyses.EMLX.Wording
  alias ArgusNxTensorAnalyses.EmittedKinds

  test "each divergence the rules emit is worded" do
    %{"emlx_divergence" => kinds} = EmittedKinds.emitted(["emlx_divergence"])

    assert kinds != [], "the rules name no divergence: read their kinds anew"
    assert for(kind <- kinds, Wording.divergence(kind, "") == nil, do: kind) == []
  end
end
