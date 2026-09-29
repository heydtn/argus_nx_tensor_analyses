defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Traced do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/traced.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @impl true
  def call_error("data_read_in_trace", how, _operation) do
    %{
      title: "reads a tensor's data while Nx traces the function",
      detail:
        "#{traced(how)}: Nx runs it over expressions, tensors that stand for the values the " <>
          "compiled code computes and hold no data. Reading one's data raises " <>
          "(\"cannot invoke to_binary/2 on Nx.Defn.Expr\").",
      label: "reads the data here",
      help:
        "compute with Nx functions inside the traced code and read the value once the compiled function returns, or see it at run time with print_value/2",
      frame: "the expression it reads is made by"
    }
  end

  def call_error("jit_in_trace", how, operation) do
    %{
      title:
        titled(
          operation,
          "A jitted function runs while Nx traces another",
          "runs a jitted function while Nx traces another"
        ),
      detail:
        "#{traced(how)}. Nx compiles one function at a time, and a jitted or compiled " <>
          "function run during a compilation raises (\"cannot invoke JITed function when " <>
          "there is a JIT compilation happening\").",
      label: "runs it here",
      help:
        "call the function the jit wraps directly, which Nx traces into the computation around it, or give the jit `on_conflict: :reuse`",
      frame: "the jitted function is made by"
    }
  end

  def call_error("tensor_arithmetic", operator, operation) do
    %{
      title:
        titled(
          operation,
          "Elixir's #{spelled(operator)} gets a tensor",
          "applies Elixir's #{spelled(operator)} to a tensor"
        ),
      detail:
        "Elixir's #{spelled(operator)} takes numbers, and a tensor is a struct: it raises " <>
          "ArithmeticError. Nx overloads the operators only inside a `defn`, where " <>
          "`Nx.Defn.Kernel` replaces them.",
      label: "gets a tensor here",
      help: "use #{nx_arithmetic(operator)}, or move the computation into a `defn`",
      frame: "the tensor is made by"
    }
  end

  def call_error("tensor_boolean", operator, _operation) do
    %{
      title: "gets a tensor where Elixir's `#{operator}` takes a boolean",
      detail:
        "Elixir's `#{operator}` takes a boolean, and a tensor is a struct: it raises " <>
          "(BadBooleanError for `and` and `or`, ArgumentError for `not`). Nx's comparisons " <>
          "give tensors of 0s and 1s, not booleans.",
      label: "gets a tensor here",
      help:
        "combine tensors with Nx.logical_and/2, Nx.logical_or/2 or Nx.logical_not/1, or read a scalar's truth as Nx.to_number(x) == 1",
      frame: "the tensor is made by"
    }
  end

  def call_error("tensor_comparison", other, operation) do
    %{
      title:
        titled(
          operation,
          "A comparison compares a tensor as an Elixir term",
          "compares a tensor as an Elixir term"
        ),
      detail:
        "Elixir compares the tensor with #{other} as terms, not by the numbers it holds: a " <>
          "tensor is a struct, which sorts after every number and equals none, so the answer " <>
          "is the same whatever the tensor holds.",
      label: "compares it here",
      help:
        "compare with Nx (Nx.greater/2, Nx.equal/2, ...) and reduce with Nx.all/2 or Nx.any/2, then read the result with Nx.to_number/1 where Elixir needs a boolean",
      frame: "the tensor is made by",
      severity: :warning
    }
  end

  def call_error("tensor_truth", _detail, operation) do
    %{
      title:
        titled(
          operation,
          "A truth test takes a tensor as always true",
          "takes a tensor as always true"
        ),
      detail:
        "A tensor is a struct, which Elixir takes as true whatever it holds, so this `if`, " <>
          "`unless`, `&&`, `||` or `!` always goes the same way. Nx's comparisons and " <>
          "reductions give tensors of 0s and 1s, not booleans.",
      label: "tests the tensor here",
      help: "read the value and compare it, as Nx.to_number(Nx.all(x)) == 1",
      frame: "the tensor is made by",
      severity: :warning
    }
  end

  def call_error("compiled_template", detail, operation) do
    %{
      title:
        titled(
          operation,
          "A compiled function gets an argument its template does not fit",
          "gets an argument its compiled template does not fit"
        ),
      detail:
        "Nx.Defn.compile/3 compiles for the templates it is given, and the function it makes " <>
          "takes only tensors of their shapes, names and types. Nx raises: #{detail}.",
      label: "calls it here",
      help:
        "call it with tensors of the template's shape, or compile it for the shapes it is called with",
      frame: "the template is made by"
    }
  end

  def call_error("captured_tensor", how, _operation) do
    %{
      title: "traces a closure that captures a tensor",
      detail:
        "Nx traces the closure with the captured tensor as a constant, and the expressions it " <>
          "traces mix only with tensors on Nx.BinaryBackend: on EMLX, EXLA or any other " <>
          "backend #{captured_raise(how)}. On Nx.BinaryBackend it inlines the tensor into the " <>
          "computation.",
      label: "captures a tensor here",
      help:
        "pass the tensor to the traced function as an argument, or run the whole computation in a `defn` or Nx.Defn.jit/2 so the tensor comes in as a parameter",
      frame: "the captured tensor is made by",
      severity: :warning
    }
  end

  def call_error("template_computed", position, operation) do
    %{
      title:
        titled(
          operation,
          "A jitted function computes with a template",
          "computes with a template"
        ),
      detail:
        "Its #{ordinal(position)} argument is a template, which has a shape and a type and no " <>
          "data: Nx raises computing with it (\"cannot perform operations on a " <>
          "Nx.TemplateBackend tensor\"). Templates are for Nx.Defn.compile/3, which compiles " <>
          "for them.",
      label: "gets a template here",
      help:
        "compute with a tensor that holds data, or hand the template to Nx.Defn.compile/3 and call the function it makes with tensors",
      frame: "the template is made by"
    }
  end

  def call_error("batch_size", size, _operation) do
    %{
      title: "gets a batch size it has no clause for",
      detail:
        "Nx.to_batched/3 takes a positive integer batch size, and #{size} is not one: it " <>
          "raises FunctionClauseError.",
      label: "gets #{size} here",
      help: "pass a positive integer batch size",
      frame: "because of this"
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  @impl true
  def violation("needs_axes") do
    %{
      title: "gets a scalar where it takes a tensor with axes",
      why:
        "Nx.to_list/1 makes a list of each axis, Nx.to_heatmap/2 draws the axes and " <>
          "Nx.to_batched/3 cuts the leading one, so each takes a tensor of at least one axis " <>
          "(a vectorized axis counts for Nx.to_list/1 and Nx.to_batched/3).",
      help:
        "read a scalar with Nx.to_number/1, or keep an axis where the tensor is made (keep_axes: true, Nx.new_axis/3)"
    }
  end

  def violation("batch") do
    %{
      title: "takes batches larger than the tensor's leading axis",
      why:
        "Nx.to_batched/3 cuts the leading axis (the first vectorized axis, where there is " <>
          "one) into batches of the size it is given, which must fit in it at least once.",
      help: "use a batch size at most the leading axis's size, or pad the tensor to it first"
    }
  end

  def violation(_kind), do: nil

  # Where the function Nx traces comes from, as the rules name it.
  defp traced("defn"),
    do: "This code runs inside a `defn`, which reaches it through a transform or a helper"

  defp traced("grad"), do: "This code runs in a function a grad differentiates, which Nx traces"

  defp traced(_jit),
    do:
      "This code runs in a function handed to Nx.Defn.jit/2 (or jit_apply/3 or compile/3), which Nx traces to compile it"

  defp captured_raise("grad"),
    do:
      "Nx.Defn.grad/2 raises that \"a tensor captured as a closure ... is on a non-default backend\""

  defp captured_raise(_jit), do: "Nx raises Nx.Defn.IncompatibleBackendsError"

  # A title, whole where the finding is at an instruction that calls
  # nothing, else completing the call it names.
  defp titled("", sentence, _predicate), do: sentence
  defp titled(_operation, _sentence, predicate), do: predicate

  # The Nx function that does to tensors what an Elixir operator does to
  # numbers.
  @nx_arithmetic %{
    "+" => "Nx.add/2",
    "-" => "Nx.subtract/2 or Nx.negate/1",
    "*" => "Nx.multiply/2",
    "/" => "Nx.divide/2",
    "div" => "Nx.quotient/2",
    "rem" => "Nx.remainder/2",
    "abs" => "Nx.abs/1",
    "float" => "Nx.as_type/2",
    "trunc" => "Nx.as_type/2",
    "round" => "Nx.round/1",
    "ceil" => "Nx.ceil/1",
    "floor" => "Nx.floor/1",
    "bnot" => "Nx.bitwise_not/1",
    "band" => "Nx.bitwise_and/2",
    "bor" => "Nx.bitwise_or/2",
    "bxor" => "Nx.bitwise_xor/2",
    "bsl" => "Nx.left_shift/2",
    "bsr" => "Nx.right_shift/2",
    "fconv" => "the Nx function for the operator, such as Nx.divide/2"
  }

  defp nx_arithmetic(operator), do: Map.get(@nx_arithmetic, operator, "the Nx function")

  # An operator as the reader knows it: `fconv` takes a value into float
  # arithmetic, as `/` does.
  defp spelled("fconv"), do: "float arithmetic (such as `/`)"
  defp spelled(operator), do: "`#{operator}`"

  @ordinals ~w(first second third fourth fifth sixth seventh eighth ninth tenth)

  defp ordinal(position) do
    index = String.to_integer(position)
    Enum.at(@ordinals, index, "#{index + 1}th")
  end
end
