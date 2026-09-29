defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.ReshapeOrder do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/reshape_order.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @impl true
  def violation("reshape_order") do
    %{
      title: "moves a size past another axis, which scrambles the data",
      why:
        "A reshape keeps the elements in their row-major order and only regroups them: an axis split or merged in place keeps what it indexes, but a size moved past another axis's runs along that axis's elements, and the result's axes mix the two.",
      help:
        "reshape only to split or merge axes in place, and move them with Nx.transpose/2: split and then transpose, or transpose and then merge"
    }
  end

  def violation(_kind), do: nil
end
