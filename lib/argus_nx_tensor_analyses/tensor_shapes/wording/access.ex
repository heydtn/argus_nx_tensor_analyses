defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Access do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/access.dl` say. An access
  # finding's detail is the message Nx raises, and its label reads the
  # index, range, axis and shape from it.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  @impl true
  def call_error("access_scalar", detail, _operation) do
    %{
      title: "indexes a scalar tensor",
      detail: "A scalar tensor has no axis to index, and Nx raises: #{detail}.",
      label: "indexes a tensor of shape {}",
      help: "index the tensor before it is reduced to a scalar, or use the scalar as it is",
      frame: "makes the scalar:"
    }
  end

  def call_error("access_out_of_bounds", detail, _operation) do
    [_message, index, axis, shape] =
      Regex.run(~r/^index (-?\d+) is out of bounds for axis (\d+) in shape (\{.*\})$/, detail)

    %{
      title: "indexes past the end of an axis",
      detail:
        "An integer index, and each bound of a range, counts from the start of its axis, " <>
          "or from the end where it is negative, and must fall within the axis. Nx raises: #{detail}.",
      label: "index #{index} on axis #{axis}, of size #{axis_size(shape, axis)}",
      help:
        "index within the axis: from 0 to its size less 1, or from minus its size to -1 counting from the end",
      frame: "makes the #{shape} tensor:"
    }
  end

  def call_error("access_negative_step", detail, _operation) do
    [_message, range, step] = Regex.run(~r/got range: (-?\d+\.\.-?\d+\/\/(-?\d+))/, detail)

    %{
      title: "slices with a range that steps backwards",
      detail:
        "A tensor is sliced with ranges of positive step only, and Elixir gives `first..last` a " <>
          "step of -1 where `first` is above `last`, as in `1..-1`. Nx raises: #{sentence(detail)}",
      label: "range #{range} steps by #{step}",
      help:
        "write the step: `1..-1//1` takes from index 1 to the last; reverse a slice with Nx.reverse/2",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_empty_range", detail, _operation) do
    [_message, range] = Regex.run(~r/got: (\S+)$/, detail)

    %{
      title: "slices a range that holds no index",
      detail:
        "Counted from the start of its axis, the range's first bound comes after its last, so it " <>
          "selects nothing. Nx raises: #{detail}.",
      label: "range #{range}, counted from the start, holds no index",
      help: "make the range's first bound come no later than its last",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_too_many_indices", detail, _operation) do
    {axis, shape} = slicing_axis(detail)

    %{
      title: "gets more indices than the tensor has axes",
      detail:
        "A list indexes the tensor's axes in order, one entry for each. Nx raises: #{detail}.",
      label: "#{String.to_integer(axis) + 1} indices for #{shape}, of #{rank(shape)} axes",
      help: "give at most one index for each axis; `..` keeps a whole axis",
      frame: "makes the #{shape} tensor:"
    }
  end

  def call_error("access_duplicate_name", detail, _operation) do
    {axis, shape} = slicing_axis(detail)

    %{
      title: "names an axis twice",
      detail: "A keyword list indexes each axis it names once. Nx raises: #{detail}.",
      label: "names axis #{axis} of #{shape} twice",
      help: "name each axis once",
      frame: "makes the #{shape} tensor:"
    }
  end

  def call_error("access_unknown_name", detail, _operation) do
    [_message, name, names] =
      Regex.run(~r/^tensor does not have name (\S+)\. The tensor names are: (.*)$/, detail)

    %{
      title: "names an axis the tensor does not have",
      detail: "A keyword list indexes the axes it names. Nx raises: #{detail}.",
      label: "names #{name}, and the tensor's axes are #{names}",
      help: "name one of the tensor's axes, or index by position",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_float_index", detail, _operation) do
    [_message, float] = Regex.run(~r/[Gg]ot:? (\S+)$/, detail)

    %{
      title: "gets a float as an index",
      detail: "An index is an integer, a scalar integer tensor or a range. Nx raises: #{detail}.",
      label: "indexes with #{float}",
      help: "index with an integer, such as trunc(x) or round(x)",
      frame: "makes the float:"
    }
  end

  def call_error("access_float_index_tensor", detail, _operation) do
    %{
      title: "gets a float tensor as an index",
      detail: "Nx indexes with integer tensors only, and raises: #{detail}.",
      label: float_index_label(detail),
      help:
        "make the index an integer tensor, such as Nx.as_type(x, :s32), or compute it with integer operations",
      frame: "makes the index a float:"
    }
  end

  def call_error("access_tensor_in_list", detail, _operation) do
    [_detail, shape, axis] = Regex.run(~r/^a (.+) tensor for axis (\d+)$/, detail)

    %{
      title: "gets a tensor with axes among its indices",
      detail:
        "Each entry of a list indexes one axis: an integer, a scalar tensor or a range. " <>
          "Nx raises: tensor must be a scalar when accessing a list/keyword of dimensions, " <>
          "got: #Nx.Tensor<...>.",
      label: "indexes axis #{axis} with a #{shape} tensor",
      help: "take along an axis with Nx.take/3, or index with a scalar tensor",
      frame: "makes the index:"
    }
  end

  def call_error("access_update", detail, _operation) do
    [callback] = Regex.run(~r/^Access\.\w+\/\d/, detail)

    %{
      title: "updates a tensor through Access",
      detail:
        "put_in, update_in and get_and_update_in update a value at an index through " <>
          "Access.get_and_update/3, and pop_in through Access.pop/2, and Nx.Tensor implements " <>
          "neither. Nx raises: #{detail}.",
      label: "needs #{callback}, which Nx.Tensor does not implement",
      help: "write the slice with Nx.put_slice/3, or the elements with Nx.indexed_put/3",
      frame: "makes the tensor:"
    }
  end

  def call_error("access_index_clamped", detail, _operation) do
    [_message, index, axis, size, read] =
      Regex.run(~r/index of (-?\d+) into axis (\d+) of size (\d+) reads index (\d+)$/, detail)

    %{
      title: "clamps a scalar tensor index into its axis",
      detail:
        "Nx slices at a tensor index clamped into its axis, where it raises for an integer " <>
          "index, and counts no tensor index from the end: #{detail}.",
      label: "index #{index} on axis #{axis}, of size #{size}, reads index #{read}",
      help:
        "index with an integer, which Nx checks and counts from the end where negative, or keep the tensor index within the axis",
      frame: "makes the tensor:",
      severity: :warning
    }
  end

  def call_error("tuple_as_tensor", position, _operation) do
    %{
      title: "gets a tuple of tensors where it takes a tensor",
      detail:
        "Nx takes a number or a tensor there, and a tuple of tensors is neither: it raises " <>
          "that it cannot convert the tuple to tensor because it represents a collection of tensors.",
      label: "gets a tuple of tensors as its #{ordinal(position, 5)} argument",
      help:
        "take the element meant, as in `{values, _indices} = Nx.top_k(t)` or `{sample, key} = Nx.Random.uniform(key)`; join tensors with Nx.stack/2 or Nx.concatenate/2",
      frame: "returns the tuple:"
    }
  end

  def call_error("tensors_in_tensor_data", detail, _operation) do
    %{
      title: "gets a list of tensors",
      detail: "Nx.tensor/2 builds a tensor from numbers, and raises for tensors: #{detail}.",
      label: "gets a list that holds a tensor",
      help: "join the tensors with Nx.stack/2 or Nx.concatenate/2",
      frame: "makes the list:"
    }
  end

  def call_error("squeeze_input_size", detail, _operation) do
    [_detail, axis, size] = Regex.run(~r/^axis (\d+) has size (.+)$/, detail)
    size = input_size(size)

    %{
      title: "removes every axis of size 1, and an input decides which",
      detail:
        "Without :axes, Nx.squeeze removes each axis of size 1, and axis #{axis} has size " <>
          "#{size}, read from an input's shape: where the input has an axis of 1 (a batch or a " <>
          "sequence of one), the result has an axis fewer than the code expects, and later " <>
          "operations broadcast or contract the wrong axes.",
      label: "removes axis #{axis} wherever #{size} is 1",
      help: "name the axes to remove: Nx.squeeze(t, axes: [...])",
      frame: "makes the tensor:",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A message Nx raises, ended as a sentence: Nx ends a suggestion with a
  # question mark.
  defp sentence(message),
    do: if(String.ends_with?(message, "?"), do: message, else: message <> ".")

  # The axis and shape of Nx's "unknown or duplicate axis 2 found when
  # slicing shape {4, 5}".
  defp slicing_axis(detail) do
    [_message, axis, shape] =
      Regex.run(~r/^unknown or duplicate axis (\d+) found when slicing shape (\{.*\})$/, detail)

    {axis, shape}
  end

  # Nx's two messages for a float tensor key: a scalar Nx slices an axis at,
  # or indices it takes.
  defp float_index_label(detail) do
    case Regex.run(~r/got an? (\w+) type(?: for axis (\d+))?$/, detail) do
      [_message, class, axis] -> "indexes axis #{axis} with a #{class} tensor"
      [_message, class] -> "indexes with a #{class} tensor"
    end
  end

  # The sizes a shape spells, `{4, 5}` as `["4", "5"]`.
  defp sizes("{}"), do: []

  defp sizes(shape),
    do: shape |> String.trim_leading("{") |> String.trim_trailing("}") |> String.split(", ")

  defp axis_size(shape, axis), do: shape |> sizes() |> Enum.at(String.to_integer(axis))

  defp rank(shape), do: shape |> sizes() |> length()

  # A size the code reads from an input's shape, as the rules spell it
  # (`arg1#{0}`, the size of axis 0 of the second argument), as the code
  # asks it: `Nx.axis_size(arg1, 0)`.
  defp input_size(size) do
    case Regex.run(~r/^(.+)#(?:\{(\d+)\}|(:\w+))$/, size) do
      [_size, tensor, axis] -> "Nx.axis_size(#{tensor}, #{axis})"
      [_size, tensor, "", name] -> "Nx.axis_size(#{tensor}, #{name})"
      nil -> size
    end
  end
end
