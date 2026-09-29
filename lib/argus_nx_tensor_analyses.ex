defmodule ArgusNxTensorAnalyses do
  @moduledoc """
  [Argus](https://github.com/QuinnWilton/argus) analyses of code that uses
  [Nx](https://github.com/elixir-nx/nx), run over a project's compiled
  modules.

    * `ArgusNxTensorAnalyses.TensorShapes` — Nx calls whose operand shapes
      Nx rejects, and calls Nx accepts where the code does not line its
      axes up.

  Argus runs only the analyses it ships, so `mix argus_nx_tensor_analyses`
  runs these beside Argus's own and reports both as one; aliased as
  `argus`, it takes `mix argus`'s place.
  """

  @doc "The analyses this package defines, each an `Argus.Analysis`."
  @spec analyses() :: [module()]
  def analyses, do: [ArgusNxTensorAnalyses.TensorShapes]
end
