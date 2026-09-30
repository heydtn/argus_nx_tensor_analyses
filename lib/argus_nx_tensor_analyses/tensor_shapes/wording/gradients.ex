defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Gradients do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/gradients.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  alias ArgusNxTensorAnalyses.TensorShapes.Wording.Causes

  @impl true
  def hazard("infinite_gradient", "spread", operation) do
    if without_arity(operation) == "Nx.standard_deviation" do
      %{
        title: "has a NaN gradient where every value is the same",
        detail:
          "A grad differentiates it, and a standard deviation is zero where the values it " <>
            "spreads are all the same. Its derivative there divides by it, 0/0, so the " <>
            "gradient is NaN, and an epsilon added after it, as in (x - mean) / (std + eps), " <>
            "does not keep the gradient finite.",
        label: "differentiated here",
        help:
          "take the root of the variance plus an epsilon instead: Nx.sqrt(Nx.add(Nx.variance(x), 1.0e-5))",
        frame: "the standard deviation can be zero because of this"
      }
    end
  end

  def hazard("gradient_at_origin", cause, _operation) do
    %{
      title: "has a NaN gradient where both its coordinates are zero",
      detail:
        "A grad differentiates it, and both of its coordinates can be zero: #{Causes.gradient(cause)}. " <>
          "The angle is 0 there, but its derivative, y/(x² + y²), is 0/0, so the gradient is " <>
          "NaN, and it carries back into everything before it.",
      label: "differentiated here",
      help:
        "keep the coordinates off the origin where they are differentiated: replace them there first (Nx.select(at_origin, 1.0, x)) and select the angle you want there after",
      frame: "a coordinate can be zero because of this",
      severity: if(cause == "input", do: :info, else: :warning)
    }
  end

  def hazard("gradient_at_edge", cause, operation) do
    {derivative, help} = edge(without_arity(operation))

    %{
      title: "has an infinite gradient at the edge of its domain",
      detail:
        "A grad differentiates it, and its operand can reach the edge of its domain: " <>
          "#{Causes.gradient(cause)}. The value is finite there, but the derivative, #{derivative}, is " <>
          "infinite, and the gradient carries it back into everything before it.",
      label: "differentiated here",
      help: help,
      frame: "the operand reaches the edge because of this"
    }
  end

  def hazard("masked_gradient", cause, _operation) do
    %{
      title: "has a NaN gradient where a select masks it out",
      detail:
        "A select keeps its result only where its operand is away from zero, and a grad " <>
          "differentiates it. The select's derivative sends zero into it where it is masked " <>
          "out, and there the derivative of the #{singular(cause)} is infinite: zero times " <>
          "infinity is NaN, and the gradient carries it back, though the value is finite.",
      label: "computed for every element here",
      help:
        "mask the operand too, so the branch is finite everywhere: take it of Nx.select(keep, x, 1.0) rather than of x (a double select)",
      frame: "masked out by"
    }
  end

  def hazard("exponent_gradient", cause, _operation) do
    base =
      if cause == "written",
        do: "is written negative",
        else: "can be negative as far as the code shows"

    %{
      title: "has a NaN gradient for its exponent where the base is negative",
      detail:
        "A grad differentiates its exponent, and its base #{base}. The exponent's derivative " <>
          "is the logarithm of the base times the power, NaN for a negative base, though the " <>
          "power itself can be finite.",
      label: "differentiated here",
      help:
        "differentiate the exponent only of a positive base: take the power of its absolute value and apply the sign apart, or stop_grad the exponent",
      frame: "because of this",
      severity: if(cause == "written", do: :warning, else: :info)
    }
  end

  def hazard("sigmoid_gradient_overflow", cause, _operation) do
    precision =
      if cause == "half_precision",
        do: " The number is past where exp overflows in f16, about -11, though not in f32.",
        else: ""

    %{
      title: "has a NaN gradient where an added mask takes its operand far below zero",
      detail:
        "A grad differentiates it, and its operand adds a number far below zero, as a mask " <>
          "added to logits does. Nx differentiates a sigmoid as exp(-x) times its value " <>
          "squared, and exp(-x) overflows to infinity there while the value is zero or " <>
          "nearly, so the gradient is NaN or infinite where it is 0.#{precision}",
      label: "differentiated here",
      help:
        "mask with a select rather than an addition: Nx.select(mask, x, -1.0e9) sends the masked elements' gradient to the constant",
      frame: "the mask is added here:",
      severity: if(cause == "half_precision", do: :info, else: :warning)
    }
  end

  def hazard("degenerate_gradient", cause, operation) do
    %{
      title: "has a NaN gradient for a degenerate matrix",
      detail:
        "A grad differentiates it, and its operand is #{matrix(cause)}. " <>
          "#{differentiated(operation)} with divisions by the gaps between its singular " <>
          "values or eigenvalues, or by the singular values, which are zero there, so the " <>
          "gradient is NaN.",
      label: "differentiated here",
      help: degenerate_help(cause),
      frame: "the matrix is made here:"
    }
  end

  def hazard(_kind, _cause, _operation), do: nil

  @impl true
  def call_error("singular_gradient", cause, _operation) do
    %{
      title: "raises computing the gradient of a singular matrix",
      detail:
        "A grad differentiates it, and its operand is #{matrix(cause)}, singular by " <>
          "construction. Nx's derivative solves with the matrix, and raises " <>
          "\"can't solve for singular matrix\".",
      label: "differentiated here",
      help: degenerate_help(cause),
      frame: "the matrix is made here:",
      severity: :warning
    }
  end

  def call_error("no_gradient", _detail, operation) do
    %{
      title: "has no gradient, and a grad differentiates it",
      detail:
        "Nx has no derivative for #{operation}, and a gradient flows back through it: " <>
          "computing the gradient raises \"cannot compute gradient for #{operation}\".",
      label: "differentiated here",
      help: no_gradient_help(without_arity(operation)),
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("complex_gradient", _detail, _operation) do
    %{
      title: "gives a complex gradient for a real input",
      detail:
        "A grad differentiates it, and Nx's derivative of a Fourier transform is complex: " <>
          "the gradient of the real values before it comes back as complex numbers, of which " <>
          "the real part is right. An update with it makes the parameters complex.",
      label: "differentiated here",
      help: "take the real part of the gradient before applying it: Nx.real(gradient)",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("custom_grad_not_list", _detail, _operation) do
    %{
      title: "has a gradient function that returns no list",
      detail:
        "A custom_grad's gradient function returns a list of gradients, one for each input " <>
          "it lists, and this one returns something else: Nx raises \"custom_grad/3 must " <>
          "return a list of tensors that map directly to the inputs\".",
      label: "defined here",
      help: "return a list with a gradient for each input: fn g -> [Nx.multiply(g, ...)] end",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("custom_grad_short", detail, _operation) do
    [returned, listed] = String.split(detail, " of ")

    %{
      title: "returns fewer gradients than it lists inputs",
      detail:
        "Its gradient function returns #{returned} gradients for #{listed} inputs. Nx pairs " <>
          "them with Enum.zip, which drops the inputs past the last gradient: they get no " <>
          "gradient from this expression, silently.",
      label: "defined here",
      help: "return a gradient for every input, in the order the inputs are listed",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("custom_grad_unlisted", _detail, _operation) do
    %{
      title: "is made from a differentiated value it does not list",
      detail:
        "Its expression is made from a value a grad differentiates that it does not list " <>
          "among its inputs. Nx follows the gradient back through the listed inputs only, so " <>
          "that value gets no gradient from the expression, silently.",
      label: "defined here",
      help:
        "list every differentiated value the expression is made from among the inputs, and return a gradient for each",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("gradient_of_container", shape, _operation) do
    %{
      title: "differentiates a function that returns a #{shape}",
      detail:
        "The function it differentiates returns a #{shape}, and Nx differentiates a tensor: " <>
          "it raises \"unable to build tensor expression, expected a tensor or a number\".",
      label: "differentiates here",
      help:
        "return the tensor to differentiate, or use value_and_grad/3 with a transform that picks it: Nx.Defn.value_and_grad(x, fun, &elem(&1, 0))",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A partial function's derivative, and how to keep its operand off the
  # edge where it is infinite.
  defp edge("Nx.acosh") do
    {"1/sqrt(x² - 1)",
     "keep the operand above 1 where it is differentiated, such as Nx.add(x, 1.0e-6) or a clip a little above 1"}
  end

  defp edge(name) do
    derivative = if name == "Nx.acos", do: "-1/sqrt(1 - x²)", else: "1/sqrt(1 - x²)"

    {derivative,
     "keep the operand strictly inside where it is differentiated: clip it a little short of the edge, such as Nx.clip(x, -1 + 1.0e-6, 1 - 1.0e-6)"}
  end

  defp singular("logarithm"), do: "logarithm"
  defp singular("root"), do: "root"
  defp singular(_division), do: "division"

  defp matrix("rank_one"), do: "an outer product, of rank one"

  defp matrix("low_rank"),
    do: "a product over an axis shorter than its sides, short of full rank"

  defp matrix(_scaled_identity),
    do: "the identity times a scalar, whose eigenvalues are all the same"

  # How Nx differentiates a decomposition, or a norm it computes from the
  # singular values.
  defp differentiated("Nx.LinAlg.norm" <> _arity),
    do: "Nx computes this norm from the matrix's singular values, and differentiates them"

  defp differentiated(_decomposition), do: "Nx differentiates it"

  defp degenerate_help("scaled_identity"),
    do:
      "differentiate the scale itself rather than a decomposition of the scaled identity, whose eigenvalues and singular values are the scale"

  defp degenerate_help(_singular),
    do:
      "keep the matrix full rank: add a small multiple of the identity before decomposing it, such as Nx.add(m, Nx.multiply(1.0e-6, Nx.eye(n)))"

  defp no_gradient_help(name) when name in ["Nx.quotient", "Nx.Defn.Kernel.div"],
    do:
      "divide and floor instead (Nx.floor(Nx.divide(x, y))), or take the quotient of stop_grad of its operands"

  defp no_gradient_help(_name),
    do:
      "use Nx.sum, Nx.product, Nx.reduce_max or a similar Nx function, or wrap it in stop_grad (no gradient) or custom_grad (a gradient of your own)"
end
