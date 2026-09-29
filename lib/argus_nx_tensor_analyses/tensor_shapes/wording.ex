defmodule ArgusNxTensorAnalyses.TensorShapes.Wording do
  @moduledoc false
  # What a finding says, for the kinds `priv/tensor_shapes.dl` and the
  # rules files under `priv/tensor_shapes/` report: each file's wording
  # module answers for its own kinds and returns nil for the rest.
  #
  # A call error, hazard or type error is `%{title, detail, label, help,
  # frame}`, with an optional `:severity` (`:error`, `:warning`, `:info`);
  # a violation is `%{title, why, help}`.

  alias __MODULE__, as: Wording

  @callback call_error(kind :: String.t(), detail :: String.t(), operation :: String.t()) ::
              map() | nil
  @callback hazard(kind :: String.t(), cause :: String.t(), operation :: String.t()) ::
              map() | nil
  @callback violation(kind :: String.t()) :: map() | nil
  @callback type_error(
              kind :: String.t(),
              subject :: String.t(),
              operation :: String.t(),
              position :: String.t(),
              certain :: String.t()
            ) :: map() | nil

  defmacro __using__(_options) do
    quote do
      @behaviour ArgusNxTensorAnalyses.TensorShapes.Wording

      @impl true
      def call_error(_kind, _detail, _operation), do: nil
      @impl true
      def hazard(_kind, _cause, _operation), do: nil
      @impl true
      def violation(_kind), do: nil
      @impl true
      def type_error(_kind, _subject, _operation, _position, _certain), do: nil

      defoverridable call_error: 3, hazard: 3, violation: 1, type_error: 5
    end
  end

  # The wording modules, in the order they are asked. One module words
  # each kind, whatever its detail, cause or operation; a module that words
  # only particular causes or operations of another's kind comes before
  # it.
  @modules [
    Wording.Options,
    Wording.Traced,
    Wording.DefnFlow,
    Wording.Containers,
    Wording.Gradients,
    Wording.Math,
    Wording.Indices,
    Wording.Literals,
    Wording.Access,
    Wording.Tuples,
    Wording.ShapeGaps,
    Wording.ReshapeOrder,
    Wording.Dtypes,
    Wording.Serving,
    Wording.Consumption,
    Wording.LinAlg,
    Wording.Core,
    Wording.Nonfinite
  ]

  @spec modules() :: [module()]
  def modules, do: @modules

  def call_error(kind, detail, operation),
    do: Enum.find_value(@modules, & &1.call_error(kind, detail, operation))

  def hazard(kind, cause, operation),
    do: Enum.find_value(@modules, & &1.hazard(kind, cause, operation))

  def violation(kind), do: Enum.find_value(@modules, & &1.violation(kind))

  def type_error(kind, subject, operation, position, certain),
    do: Enum.find_value(@modules, & &1.type_error(kind, subject, operation, position, certain))
end
