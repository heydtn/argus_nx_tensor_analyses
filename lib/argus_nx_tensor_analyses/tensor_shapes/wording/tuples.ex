defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Tuples do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/tuples.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @violations %{
    "key" => %{
      title: "gets a tensor of another shape for its key",
      why:
        "A key is a single tensor of shape {2}. Nx.Random.split/2 returns its keys stacked, one per row, and each is taken out before it is used.",
      help:
        "pass one key: a row of the keys Nx.Random.split/2 returns (keys[0]), or what Nx.Random.key/1 returns"
    },
    "sampler_parameters" => %{
      title: "gets parameters that do not fit the shape it draws",
      why:
        "A sampler draws values of its :shape and combines them element by element with its parameters (the bounds of uniform and randint, the mean and standard deviation of normal), so each parameter must broadcast into that shape.",
      help:
        "give :shape the parameters' shape, such as shape: Nx.shape(mean), or give the parameters a shape that broadcasts into it"
    },
    "choice" => %{
      title: "gets a tensor, sample count or probabilities it cannot choose from",
      why:
        "Nx.Random.choice takes at least one sample from a tensor of rank 1 or more, no more samples than there are elements when it does not replace them, and a vector with one probability for each element it chooses among.",
      help:
        "check :samples and :replace against the elements sampled (the axis's, or all of them without :axis), and give one probability for each"
    },
    "multivariate_normal" => %{
      title: "gets a mean and covariance that do not fit",
      why:
        "A multivariate normal takes a mean vector of some size d and a square d by d covariance matrix.",
      help: "pass the mean as a vector, and the covariance as a square matrix of the mean's size"
    },
    "norm" => %{
      title: "gets an order it does not take for this rank",
      why:
        "The Frobenius and nuclear norms are of matrices, a matrix takes the integer orders 1, -1, 2 and -2 only, and an order is one Nx.LinAlg.norm/2 lists.",
      help:
        "pick an order the tensor's rank takes: :frobenius and :nuclear for a matrix, any integer for a vector"
    }
  }

  @impl true
  def violation(kind), do: Map.get(@violations, kind)

  @impl true
  def call_error("shared_draw", detail, operation) do
    %{
      title: "repeats one draw across its mean or standard deviation",
      detail:
        "#{operation} #{detail}: it draws values of its :shape only, and broadcasting them against a larger mean or standard deviation repeats each value along the axes those add, so the elements along them are not independent draws.",
      label: "draws fewer values than it returns",
      help:
        "give :shape the shape of the sample, such as shape: Nx.shape(mean), so that each element gets a draw of its own",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("norm_axes", order, operation) do
    %{
      title: "takes one norm of the whole matrix, whatever :axes says",
      detail:
        "With ord: #{order}, #{operation} first reduces the matrix to a vector (its column sums, row sums or singular values) and applies :axes to that vector, or ignores :axes for :nuclear: it returns one norm of the whole matrix, not one for each row or column.",
      label: "ord: #{order} gives one norm of the whole matrix",
      help:
        "for a norm of each row or column take a vector norm along the axis, such as Nx.LinAlg.norm(t, axes: [1]) or Nx.sum(Nx.abs(t), axes: [1])",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil
end
