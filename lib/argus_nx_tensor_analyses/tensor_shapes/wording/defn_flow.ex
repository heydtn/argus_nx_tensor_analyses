defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.DefnFlow do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/defn_flow.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  # The call `defn` compiles a `while` to.
  @while "Nx.Defn.Kernel.__while__/6"

  @while_raise "the do-block in while must return tensors with the same shape, type, and " <>
                 "names as the initial arguments."

  @impl true
  def call_error("predicate_shape", shape, operation) do
    %{
      title: subject(operation, "decides on a tensor that is not a scalar"),
      detail:
        "An `if`, `cond` or `while` in a `defn` decides on a scalar, where 0 is false and anything " <>
          "else true, and this predicate has shape #{shape}. Nx raises while it builds the " <>
          "expression: condition must be a scalar tensor.",
      label: "decides on a #{shape} tensor, not a scalar",
      help:
        "reduce the predicate to a scalar with Nx.all/1 or Nx.any/1, or choose element by element with Nx.select/3",
      frame: "the predicate is computed by"
    }
  end

  def call_error("predicate_value", value, operation) do
    %{
      title: subject(operation, "decides on a value that is not a tensor"),
      detail:
        "An `if` or `cond` in a `defn` decides on true, false or a scalar tensor, and this " <>
          "predicate is #{value}#{missing_key(value)}. Nx raises while it builds the expression: " <>
          "cond in defn expects the predicate to be true, false, or a scalar tensor.",
      label: "decides on #{value}, not a tensor",
      help:
        "give the option a default with keyword!/2 (opts = keyword!(opts, training: false)), or match on an atom with `case`",
      frame: "#{value} comes from"
    }
  end

  def call_error("branch_shapes", detail, operation) do
    [_, later, earlier] =
      Regex.run(~r/^cannot broadcast tensor of dimensions (.+) to (.+)$/, detail)

    %{
      title: subject(operation, "has branches whose shapes do not broadcast"),
      detail:
        "Nx merges what the branches of an `if` or `cond` give into one tensor, broadcasting " <>
          "their shapes together, and raises where they do not broadcast: #{detail}.",
      label: "this branch gives #{later}",
      help:
        "make every branch give the same shape, reshaping or broadcasting the one that differs",
      frame: "an earlier branch gives #{earlier}"
    }
  end

  def call_error("branch_broadcast", detail, operation) do
    [_, earlier, later, merged] = Regex.run(~r/^(.+) and (.+) broadcast to (.+)$/, detail)

    %{
      title: subject(operation, "has branches Nx broadcasts to a shape neither gives"),
      detail:
        "Nx merges what the branches of an `if` or `cond` give by broadcasting their shapes " <>
          "together, without a word: #{detail}, a shape no branch computes.",
      label: "this branch gives #{later}, and Nx merges the two to #{merged}",
      help:
        "make every branch give the same shape; Nx.reshape/2 or Nx.squeeze/2 lines up the one that differs",
      frame: "an earlier branch gives #{earlier}",
      severity: :warning
    }
  end

  def call_error("branch_structure", detail, operation) do
    [earlier, later] = String.split(detail, " and ", parts: 2)

    %{
      title: subject(operation, "has branches that give different structures"),
      detail:
        "Every branch of an `if` or `cond` in a `defn` must give the same structure of " <>
          "tensors, and these give #{detail}. Nx raises while it builds the expression: " <>
          "cond/if expects all branches to return compatible tensor types.",
      label: "this branch gives #{later}",
      help: "give every branch the same structure: tuples of one size, or a tensor in each",
      frame: "an earlier branch gives #{earlier}"
    }
  end

  def call_error("cond_fallthrough", _detail, operation) do
    %{
      title: subject(operation, "has no clause that always holds"),
      detail:
        "A `cond` in a `defn` needs a last clause that always holds, and this one's last clause " <>
          "decides on a tensor the `defn` computes from its arguments, which may not hold. Nx " <>
          "raises while it builds the expression: cond/if expects at least one branch to always " <>
          "evaluate to true.",
      label: "the last clause decides on a tensor, which can be false",
      help: "end the `cond` with a `true ->` clause",
      frame: "the last predicate is computed by"
    }
  end

  def call_error("traced_raise", _detail, operation) do
    %{
      title: subject(operation, "raises in a branch while it decides on a tensor"),
      detail:
        "Nx builds every branch of an `if` or `cond` that decides on a tensor the `defn` " <>
          "computes from its arguments, whatever the data, so a `raise` in one of them runs " <>
          "every time the `defn` is called.",
      label: "this branch raises on every call, whatever the tensor holds",
      help:
        "raise at run time with runtime_raise/1 in the branch, or check the condition outside the defn",
      frame: "the tensor it decides on is computed by"
    }
  end

  def call_error("while_shape", detail, _operation) do
    %{
      title: "runs a `while` whose body changes the shape of its state",
      detail:
        "The body of a `while` must give tensors of the shapes, types and names the loop " <>
          "starts with, and #{detail}. Nx raises while it builds the expression: #{@while_raise}",
      label: labeled(detail),
      help:
        "start the state in the shape the body gives, such as Nx.broadcast(0.0, shape), or keep the body to the initial shape",
      frame: "the state is made here"
    }
  end

  def call_error("while_type", detail, _operation) do
    %{
      title: "runs a `while` whose body changes the type of its state",
      detail:
        "The body of a `while` must give tensors of the types the loop starts with, and " <>
          "#{detail}. Nx raises while it builds the expression: #{@while_raise}",
      label: labeled(detail),
      help:
        "start the state in the type the body gives, such as 0.0 for a float, or convert the body's result with Nx.as_type/2",
      frame: "the state is made here"
    }
  end

  def call_error("closure_captures_tensor", _detail, operation) do
    %{
      title: closure_title(operation),
      detail:
        "Nx builds the closure of a `while`, `Nx.reduce` or `Nx.window_reduce` apart from the " <>
          "rest of the `defn`, and this one computes with a tensor the `defn` makes of its " <>
          "arguments, captured from outside. Nx raises: cannot build defn because expressions " <>
          "come from different contexts.",
      label: "its closure computes with a tensor from outside it",
      help:
        "hand the tensor in through the loop's state, or the reduction's operands, rather than capturing it",
      frame: "the captured tensor is computed by"
    }
  end

  def call_error("atom_operand", value, _operation) do
    %{
      title: "gets an atom where it takes a tensor",
      detail:
        "Nx makes tensors of numbers, tensors, true and false, but of no other atom, and this " <>
          "operand is #{value}#{missing_key(value)}. In a `defn` an operator computes on tensors, " <>
          "so Nx raises: protocol Nx.LazyContainer not implemented for Atom.",
      label: "gets #{value}, which Nx makes no tensor of",
      help:
        "match on an atom with `case`, which a `defn` runs while it builds the expression, and give options defaults with keyword!/2",
      frame: "#{value} comes from"
    }
  end

  def call_error("tensor_as_integer", slot, _operation) do
    %{
      title: "gets a tensor where it takes an integer",
      detail:
        "A public `defn` makes every argument a tensor but a list or a function, and #{slot} " <>
          "here is computed from one. Nx needs it as an integer while it builds the expression, " <>
          "and raises for a tensor.",
      label: "gets a tensor for #{slot}, where Nx needs an integer",
      help:
        "pass the number in the keyword list of options (opts[:n]), which `defn` keeps as it is, or take it from a shape with Nx.axis_size/2",
      frame: "the tensor is computed by"
    }
  end

  def call_error("dropped_print", _detail, _operation) do
    %{
      title: "prints a value the defn never uses",
      detail:
        "Nx builds only what the output of a `defn` depends on, and nothing uses this call's " <>
          "result, so it never prints or calls back.",
      label: "nothing uses its result, so it never runs",
      help:
        "use the call's result in place of its argument (x = print_value(x)) so it reaches the output",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A finding about a `cond` or `if` is placed at a clause, no call, and
  # names its subject itself; one about a `while` is placed at its call.
  defp subject("", title), do: "An `if` or `cond` " <> title

  defp subject(@while, "decides on " <> _title),
    do: "runs a `while` whose condition is not a scalar"

  defp subject(_operation, title), do: title

  # A closure Nx builds apart is a `while`'s body, or a reduction's.
  defp closure_title(@while), do: "runs a `while` whose body uses a tensor from outside it"
  defp closure_title(_operation), do: "runs a closure that uses a tensor from outside it"

  # A `while`'s state as the rules place it, `element 1 of the state
  # starts as {} and the body makes it {3}`, as a label says it.
  defp labeled(detail),
    do: String.replace(detail, " and the body", ", and the body", global: false)

  # Where nil comes from, for a reader who never wrote it.
  defp missing_key("nil"), do: ", as a lookup of a key the options lack gives"
  defp missing_key(_value), do: ""
end
