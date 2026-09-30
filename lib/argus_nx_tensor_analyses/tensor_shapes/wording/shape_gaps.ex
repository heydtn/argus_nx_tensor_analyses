defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.ShapeGaps do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/shape_gaps.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @kinds %{
    "conv_options" => %{
      title: "gets options that do not fit its input and kernel",
      why:
        "A convolution takes one stride, one dilation and one {low, high} padding pair per spatial axis, strides and dilations of at least 1, permutations that name every axis once, and group sizes that divide the batch and the kernel's output channels, one of them 1.",
      help:
        "give one stride, dilation and padding pair per spatial axis, each stride and dilation at least 1, and group sizes that divide the batch and the filters"
    },
    "window_options" => %{
      title: "gets window options that do not fit its operand",
      why:
        "A window operation takes its window as a tuple of sizes of at least 1, one per axis of the operand, and a stride, a dilation and a {low, high} padding pair for every axis, strides and dilations of at least 1.",
      help:
        "write the window as a tuple, and give one stride, dilation and padding pair for every axis of the operand"
    },
    "nonpositive" => %{
      title: "gets a size, count or stride below 1",
      why:
        "Nx takes dimensions, repetitions, counts, lengths and strides of at least 1: a shape has no empty axes, a transform no empty length, and a slice steps forward.",
      help: "make the size, count or stride at least 1"
    },
    "no_tensors" => %{
      title: "joins no tensors",
      why:
        "Concatenating or stacking takes its shape from the tensors it joins, so it needs one.",
      help: "pass at least one tensor, or skip the call where the list is empty"
    },
    "tensor_as_shape" => %{
      title: "gets a tensor where it takes a shape",
      why:
        "Nx.iota/2 takes a shape tuple; handed any tensor, Nx 1.0 warns that this is deprecated and then raises.",
      help: "pass the tensor's shape: Nx.iota(Nx.shape(tensor))"
    },
    "ragged_data" => %{
      title: "builds a tensor from lists of different shapes",
      why:
        "Literal tensor data is rectangular: the lists at each depth have one length, and hold lists throughout or numbers throughout.",
      help: "give every list at a depth the same length, padding the short ones"
    },
    "cropped_axis" => %{
      title: "crops an axis to nothing",
      why:
        "Negative padding crops an axis from its edges, and cropping as much as the axis holds leaves no elements, which the backend cannot build.",
      help: "crop less than the axis holds, or slice out the part that should remain"
    },
    "irfft" => %{
      title: "rebuilds too short a signal",
      why:
        "Nx.irfft/2 rebuilds a signal of its :length (twice the axis's size less two without it) by mirroring the spectrum, and a length below 3 leaves it nothing to mirror.",
      help:
        "pass :length, the length of the signal the spectrum was taken of; it must be at least 3"
    }
  }

  @impl true
  def violation(kind), do: Map.get(@kinds, kind)

  @impl true
  def call_error("scalar_iota", number, _operation) do
    %{
      title: "makes a scalar, not a range",
      detail:
        "Nx.iota/2 takes a number as a tensor whose shape it takes, and a number's shape is {}: " <>
          "Nx.iota(#{number}) is the scalar 0, not a range of #{number} values.",
      label: "makes the scalar 0, not a range of #{number} values",
      help: "pass a shape tuple: Nx.iota({#{number}})",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("odd_irfft", length, _operation) do
    %{
      title: "assumes an even signal length",
      detail:
        "Without :length, Nx.irfft/2 rebuilds a signal of even length, twice the spectrum's " <>
          "size less two, but the spectrum is an Nx.rfft/2's of a signal of length #{length}: " <>
          "it gives a signal one shorter, of other values.",
      label: "rebuilds a signal of length #{String.to_integer(length) - 1}, not #{length}",
      help: "pass length: #{length}, the length of the signal the spectrum was taken of",
      frame: "the spectrum of the odd-length signal is taken by",
      severity: :warning
    }
  end

  def call_error("transposed_weights", weights, _operation) do
    %{
      title: "applies its weights to transposed axes",
      detail:
        "Nx.weighted_mean/3 lines weights of another shape up with the input by reshaping " <>
          "them to the input's rank, their axes last, and swapping the axis at the first " <>
          "reduced position with the last, so weights over two or more axes land transposed " <>
          "(here #{weights}): each element is weighed by another's weight.",
      label: "weights #{weights} land transposed",
      help:
        "broadcast the weights to the input's shape first, Nx.broadcast(weights, Nx.shape(input), axes: axes), so they are used as they are",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("scaling_factor_rank", shapes, _operation) do
    [factor, tensor] = String.split(shapes, " scaling ")

    %{
      title: "sums the scaled exponentials over other axes than the tensor's",
      detail:
        "Its :exp_scaling_factor, #{factor}, has more axes than the tensor, #{tensor}: Nx.logsumexp/2 " <>
          "broadcasts the factor with the exponentials and then sums over the :axes of that " <>
          "broadcast shape, which are not the tensor's axes.",
      label: "a #{factor} factor scales the #{tensor} tensor",
      help:
        "give the scaling factor at most the tensor's rank, or broadcast the tensor to the factor's shape first",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil
end
