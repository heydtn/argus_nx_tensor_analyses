defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Core do
  @moduledoc false
  # What the shape and type findings of `priv/tensor_shapes.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  # Each kind of shape finding: what the title says the call does, why it
  # fails (or why the code should not rely on it), and what to change.
  @kinds %{
    "axis" => %{
      title: "gets an axis its operand does not have",
      why:
        "An axis is an index below the operand's rank, counted from the end when negative, or one of its names, and a list of axes names each once.",
      help: "name an axis the operand has, or check the operand's rank where it is made"
    },
    "broadcast" => %{
      title: "gets shapes that do not broadcast",
      why:
        "Nx broadcasts by lining axes up from the last one: each pair of sizes must be equal, or one of them 1.",
      help:
        "make the sizes that meet equal, or 1 where one side should repeat; Nx.new_axis/3 or Nx.reshape/2 moves an axis to where it belongs"
    },
    "concatenate" => %{
      title: "joins tensors whose other axes differ",
      why: "Tensors joined along an axis must agree on every other axis.",
      help:
        "make the tensors' other axes the same size, or join them along the axis where they differ"
    },
    "conv" => %{
      title: "gets an input and kernel that do not fit",
      why:
        "A convolution needs an input and kernel of one rank, and windows that fit inside the padded input, which its strides step forward over.",
      help:
        "check the input's and kernel's ranks, the kernel's size against the input's spatial axes, and that the strides are positive"
    },
    "diagonal" => %{
      title: "gets a diagonal of the wrong length",
      why:
        "A diagonal written into a tensor must be exactly as long as the diagonal its offset picks.",
      help: "make the diagonal as long as the one the offset picks"
    },
    "diff" => %{
      title: "takes more differences than the axis holds",
      why:
        "Each order of a difference shortens the axis by one, so the order must be below the axis's size.",
      help: "lower the order, or take the differences along a longer axis"
    },
    "dot" => %{
      title: "contracts axes that do not match",
      why:
        "Nx.dot contracts axes in pairs, one from each side, and each pair must have one size; batch axes pair up the same way.",
      help:
        "contract axes of one size: transpose or reshape an operand, or give the contraction and batch axes explicitly"
    },
    "flatten" => %{
      title: "flattens axes that are not consecutive",
      why: "Flattening merges a run of neighboring axes into one.",
      help: "list axes that sit next to each other, or transpose them together first"
    },
    "gather" => %{
      title: "gets indices that do not fit the tensor",
      why:
        "Each entry along the indices' last axis addresses one axis of the tensor, so there can be no more of them than the tensor has axes.",
      help:
        "make the indices' last axis as long as the axes it addresses, and list those axes in order"
    },
    "indexed" => %{
      title: "gets indices or updates that do not fit the tensor",
      why:
        "An indexed update takes one update per index, and each index addresses the tensor's axes in order.",
      help:
        "make the updates' leading axis match the number of indices, and the indices' last axis match the axes they address"
    },
    "least_squares" => %{
      title: "gets a system and right-hand side that do not fit",
      why: "Least squares needs a right-hand side with as many rows as the system.",
      help: "make the right-hand side's rows match the system's"
    },
    "linspace" => %{
      title: "gets a start and stop that do not fit",
      why:
        "Nx.linspace/3 interpolates element by element, so start and stop need one shape, and n must be given.",
      help: "give start and stop the same shape, and give n"
    },
    "names" => %{
      title: "merges axes of different names",
      why:
        "Where axes meet, each keeps one name: a name merges with nil or with itself, and a tensor's names are its own.",
      help:
        "rename one side with Nx.rename/2 so the names agree, or drop the names that should not meet"
    },
    "pad" => %{
      title: "gets a padding configuration that does not fit",
      why:
        "Padding takes one {low, high, interior} per axis, with interior padding never negative.",
      help: "give one padding entry for each axis of the tensor"
    },
    "put_slice" => %{
      title: "puts a slice that does not fit the tensor",
      why:
        "A slice put into a tensor needs the tensor's rank, no axis longer than the tensor's, and one start index per axis.",
      help: "shrink the slice to fit the tensor, or give one start index per axis"
    },
    "rank" => %{
      title: "gets a tensor of the wrong rank",
      why:
        "The operation works on a certain number of axes: a vector, a matrix, or a batch of them.",
      help:
        "reshape the operand, or add an axis with Nx.new_axis/3, to the rank the operation takes"
    },
    "reshape" => %{
      title: "gets a shape it cannot reshape to",
      why:
        "A reshape keeps every element: the new sizes must multiply to as many elements as the old, with at most one :auto to infer.",
      help:
        "make the new sizes multiply to the old number of elements, or leave one of them :auto"
    },
    "scalar" => %{
      title: "gets a tensor where it takes a scalar",
      why: "This argument must be a single number: a tensor of rank 0 with no vectorized axes.",
      help: "pass a scalar, or reduce the tensor to one first"
    },
    "slice" => %{
      title: "slices outside the tensor",
      why:
        "A slice takes one start index, length and stride per axis, and each length must fit in its axis.",
      help: "give one start, length and stride for each axis, each within the axis"
    },
    "solve" => %{
      title: "gets a right-hand side that does not fit the system",
      why:
        "A system of n equations solves right-hand sides of n rows (n columns when solving from the right), batched as the system is.",
      help: "make the right-hand side's rows match the system's size"
    },
    "split" => %{
      title: "splits an axis at a point outside it",
      why: "A split cuts an axis in two, so the point must fall strictly inside it.",
      help: "split at a point between 0 and the axis's size"
    },
    "square" => %{
      title: "gets a matrix that is not square",
      why: "The operation takes square matrices: the last two axes must have one size.",
      help: "pass a square matrix, or a batch of them"
    },
    "squeeze" => %{
      title: "squeezes an axis whose size is not 1",
      why: "A squeeze only removes axes of size 1.",
      help: "squeeze only axes of size 1, or reshape to drop the axis"
    },
    "stack" => %{
      title: "stacks tensors of different shapes",
      why:
        "Stacking puts tensors side by side along a new axis, so they must all have one shape.",
      help: "give every tensor the same shape, or concatenate them along an axis they share"
    },
    "take_along_axis" => %{
      title: "gets indices that do not fit the tensor",
      why: "The indices must match the tensor on every axis but the one taken along.",
      help: "match the indices' shape to the tensor's on the other axes"
    },
    "top_k" => %{
      title: "takes more elements than the axis holds",
      why: "Nx.top_k/2 takes k elements from the last axis, which must hold at least k.",
      help: "lower k to at most the last axis's size"
    },
    "transpose" => %{
      title: "gets a permutation of the wrong length",
      why: "A transpose's permutation names every axis once.",
      help: "list every axis once in the permutation"
    },
    "vectorize" => %{
      title: "cannot vectorize these axes",
      why:
        "Vectorizing turns leading axes into vectorized ones, which need names of their own and the sizes they are given.",
      help: "vectorize axes the tensor has, under names it does not already use"
    },
    "vectorized_axes" => %{
      title: "gets vectorized axes of different sizes",
      why:
        "Vectorized axes of one name meet as broadcast axes do: their sizes must be equal, or one of them 1.",
      help: "make vectorized axes of one name the same size, or 1 on one side"
    },
    "weighted_mean" => %{
      title: "gets weights that do not fit the input",
      why:
        "A weighted mean needs weights of the input's shape, or axes that say where lower-rank weights go.",
      help: "give the weights the input's shape, or pass :axes for the axes they cover"
    },
    "window" => %{
      title: "gets a window that does not fit the tensor",
      why:
        "A window operation takes one window size and stride per axis, and windows that leave something of each axis, which its strides step forward over.",
      help: "give one window size and positive stride per axis, each within the padded axis"
    },
    "size_variables" => %{
      title: "lines up sizes the code names differently",
      why:
        "That holds only while the two happen to be equal, or one of them is 1 and broadcasts, and nothing in the code makes them so.",
      help: "derive both sizes from one variable, or check they agree where the variables are set"
    },
    "unnamed_axis" => %{
      title: "lines up an unnamed axis with a named one",
      why:
        "Nx merges an unnamed axis with any name, so it cannot check that these are the axes meant to meet.",
      help: "name the axis (Nx.rename/2, or names: where it is made) so the axes line up by name"
    },
    "contracted_names" => %{
      title: "contracts axes of different names",
      why:
        "Nx contracts axes by position and drops their names, so it cannot tell they were not meant to meet.",
      help: "rename one side so the contracted axes share a name"
    },
    "vectorize_name" => %{
      title: "vectorizes an axis under another name",
      why:
        "The axis keeps its size under the new name, and code that looks for it by its old name does not find it.",
      help: "vectorize the axis under its own name, or rename it first to say it is another"
    }
  }

  @impl true
  def violation(kind), do: Map.get(@kinds, kind)

  @impl true
  def type_error("non_integer_operand", class, operation, position, _certain) do
    {rejects, help} = integer_only(operation)

    %{
      title:
        "takes integers, and its #{ordinal(to_string(position), 4)} argument can be a #{class}",
      detail: "#{rejects} Nx raises for a #{class} tensor there.",
      label: "gets a #{class}, not an integer",
      help: help,
      frame: "makes it a #{class}:"
    }
  end

  def type_error("unsupported_type", name, _operation, _position, _certain) do
    %{
      title: "makes #{article(name)} #{name} tensor, which the backend does not support",
      detail:
        "The analysis is run with #{name} among the types the backend lacks (the " <>
          ":unsupported_types option). A backend without #{name} raises making it, or at the " <>
          "first operation over it.",
      label: "makes #{name}, which :unsupported_types lists",
      help:
        "make it a type the backend supports, such as f32, or run this on a backend that has #{name}",
      frame: "",
      severity: :error
    }
  end

  def type_error(_kind, _subject, _operation, _position, _certain), do: nil

  # What a function that takes only integers raises for anything else, and
  # what to change.
  defp integer_only(operation) do
    name = without_arity(operation)

    cond do
      name in ~w(Nx.take Nx.take_along_axis Nx.gather Nx.indexed_add Nx.indexed_put) ->
        {"Its indices must be an integer tensor.",
         "compute the indices as integers: round and then Nx.as_type(x, :s32), or use integer operations (Nx.quotient rather than Nx.divide)"}

      name in ~w(Nx.quotient Nx.Defn.Kernel.div) ->
        {"An integer quotient takes integer tensors only.",
         "make the operands integers, such as Nx.as_type(x, :s32), or divide and round (Nx.floor(Nx.divide(x, y))) to keep floats"}

      true ->
        {"Bitwise operations take integer tensors only.",
         "make the operand an integer tensor, such as Nx.as_type(x, :s32)"}
    end
  end
end
