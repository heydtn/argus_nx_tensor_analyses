defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Access do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/access.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @impl true
  def call_error("access_scalar", detail, _operation) do
    %{
      title: "indexes a scalar tensor",
      detail: "A scalar tensor has no axis to index, and Nx raises: #{detail}.",
      label: "indexes a scalar here",
      help: "index the tensor before it is reduced to a scalar, or use the scalar as it is",
      frame: "makes the scalar:"
    }
  end

  def call_error("access_out_of_bounds", detail, _operation) do
    %{
      title: "indexes past the end of an axis",
      detail:
        "An integer index, and each bound of a range, counts from the start of its axis, " <>
          "or from the end where it is negative, and must fall within the axis. Nx raises: #{detail}.",
      label: "out of bounds here",
      help:
        "index within the axis: from 0 to its size less 1, or from minus its size to -1 counting from the end",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_negative_step", detail, _operation) do
    %{
      title: "slices with a range that steps backwards",
      detail:
        "A tensor is sliced with ranges of positive step only, and Elixir gives `first..last` a " <>
          "step of -1 where `first` is above `last`, as in `1..-1`. Nx raises: #{detail}.",
      label: "steps backwards here",
      help:
        "write the step: `1..-1//1` takes from index 1 to the last; reverse a slice with Nx.reverse/2",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_empty_range", detail, _operation) do
    %{
      title: "slices a range that holds no index",
      detail:
        "Counted from the start of its axis, the range's first bound comes after its last, so it " <>
          "selects nothing. Nx raises: #{detail}.",
      label: "empty range here",
      help: "make the range's first bound come no later than its last",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_too_many_indices", detail, _operation) do
    %{
      title: "gets more indices than the tensor has axes",
      detail:
        "A list indexes the tensor's axes in order, one entry for each. Nx raises: #{detail}.",
      label: "too many indices here",
      help: "give at most one index for each axis; `..` keeps a whole axis",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_duplicate_name", detail, _operation) do
    %{
      title: "names an axis twice",
      detail: "A keyword list indexes each axis it names once. Nx raises: #{detail}.",
      label: "named twice here",
      help: "name each axis once",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_unknown_name", detail, _operation) do
    %{
      title: "names an axis the tensor does not have",
      detail: "A keyword list indexes the axes it names. Nx raises: #{detail}.",
      label: "unknown name here",
      help: "name one of the tensor's axes, or index by position",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_float_index", detail, _operation) do
    %{
      title: "gets a float as an index",
      detail: "An index is an integer, a scalar integer tensor or a range. Nx raises: #{detail}.",
      label: "float index here",
      help: "index with an integer, such as trunc(x) or round(x)",
      frame: "makes the float:"
    }
  end

  def call_error("access_float_index_tensor", detail, _operation) do
    %{
      title: "gets a float tensor as an index",
      detail: "Nx indexes with integer tensors only, and raises: #{detail}.",
      label: "float index here",
      help:
        "make the index an integer tensor, such as Nx.as_type(x, :s32), or compute it with integer operations",
      frame: "makes the index a float:"
    }
  end

  def call_error("access_tensor_in_list", detail, _operation) do
    %{
      title: "gets a tensor with axes among its indices",
      detail:
        "Each entry of a list indexes one axis: an integer, a scalar tensor or a range. " <>
          "Nx raises: #{detail}.",
      label: "tensor with axes here",
      help: "take along an axis with Nx.take/3, or index with a scalar tensor",
      frame: "makes the index:"
    }
  end

  def call_error("access_update", detail, operation) do
    %{
      title: "updates a tensor through Access",
      detail:
        "#{operation} updates the tensor at an index through Access.get_and_update/3 or " <>
          "Access.pop/2, which Nx.Tensor does not implement. Nx raises: #{detail}.",
      label: "updates the tensor here",
      help: "write the slice with Nx.put_slice/3, or the elements with Nx.indexed_put/3",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_index_clamped", detail, _operation) do
    %{
      title: "clamps a scalar tensor index into its axis",
      detail:
        "Nx slices at a tensor index clamped into its axis, where it raises for an integer " <>
          "index, and counts no tensor index from the end: #{detail}.",
      label: "clamped index here",
      help:
        "index with an integer, which Nx checks and counts from the end where negative, or keep the tensor index within the axis",
      frame: "makes the index:",
      severity: :warning
    }
  end

  def call_error("tuple_as_tensor", position, _operation) do
    %{
      title: "gets a tuple of tensors where it takes a tensor",
      detail:
        "Its #{ordinal(position)} argument is a tuple of tensors, which Nx cannot convert to a " <>
          "tensor, and Nx raises.",
      label: "gets a tuple here",
      help:
        "take the element meant, as in `{values, _indices} = Nx.top_k(t)` or `{sample, key} = Nx.Random.uniform(key)`; join tensors with Nx.stack/2 or Nx.concatenate/2",
      frame: "returns the tuple:"
    }
  end

  def call_error("tensors_in_tensor_data", detail, _operation) do
    %{
      title: "gets a list of tensors",
      detail: "Nx.tensor/2 builds a tensor from numbers, and raises for tensors: #{detail}.",
      label: "gets tensors here",
      help: "join the tensors with Nx.stack/2 or Nx.concatenate/2",
      frame: "makes the list:"
    }
  end

  def call_error("squeeze_input_size", detail, _operation) do
    %{
      title: "removes every axis of size 1, and an input decides which",
      detail:
        "Without :axes, Nx.squeeze removes each axis of size 1, and #{detail}, read from an " <>
          "input's shape: where the input has an axis of 1 (a batch or a sequence of one), the " <>
          "result has an axis fewer than the code expects, and later operations broadcast or " <>
          "contract the wrong axes.",
      label: "squeezes here",
      help: "name the axes to remove: Nx.squeeze(t, axes: [...])",
      frame: "makes the tensor:",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # An argument's position, counted from 0, as a reader counts it.
  defp ordinal("0"), do: "first"
  defp ordinal("1"), do: "second"
  defp ordinal("2"), do: "third"
  defp ordinal("3"), do: "fourth"
  defp ordinal("4"), do: "fifth"
  defp ordinal(position), do: "#{String.to_integer(position) + 1}th"
end
