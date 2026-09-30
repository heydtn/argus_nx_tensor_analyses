defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Containers do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/containers.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @impl true
  def call_error("dropped_field_read", detail, _operation) do
    %{struct: struct, field: field, default: default} = dropped(detail)

    %{
      title: "reads .#{field}, which #{struct}'s Nx.Container resets to #{default} inside defn",
      detail:
        "Nx rebuilds a struct handed to defn from what its Nx.Container traverses and keeps, " <>
          "and #{struct}'s container keeps no #{field}: inside the defn the field holds " <>
          "#{default}, whatever the caller set. A derived Nx.Container gives every field in " <>
          "neither containers: nor keep: its default.",
      label: "reads .#{field} as #{default}, its default, whatever the caller set",
      help:
        "list :#{field} in keep: (a value fixed when the defn compiles) or containers: " <>
          "(tensors) where #{struct} derives Nx.Container, or pass the value as an argument of its own",
      frame: "builds the struct, whose .#{field} Nx resets on the way in",
      severity: :warning
    }
  end

  def call_error("dropped_field_returned", detail, _operation) do
    %{struct: struct, field: field, default: default} = dropped(detail)

    %{
      title:
        "reads .#{field} of a struct a defn returns, which #{struct}'s Nx.Container resets to #{default}",
      detail:
        "Nx rebuilds a struct a defn or jit returns from what its Nx.Container traverses and " <>
          "keeps, and #{struct}'s container keeps no #{field}: the struct that comes back " <>
          "holds #{default} there, whatever the defn or its caller set. A derived " <>
          "Nx.Container gives every field in neither containers: nor keep: its default.",
      label: "reads .#{field} as #{default}, whatever the defn set",
      help:
        "list :#{field} in keep: or containers: where #{struct} derives Nx.Container, " <>
          "or read the field from the struct handed in",
      frame: "returns the struct, .#{field} reset to #{default}:",
      severity: :warning
    }
  end

  def call_error("container_leaf", detail, operation) do
    {severity, holds, where} = certainty(detail)
    [leaf, place] = String.split(where, " at ", parts: 2)
    leaf = struct_leaf(leaf)

    %{
      title: subject(operation, "gets a container holding #{leaf}, which Nx cannot trace"),
      detail:
        "defn, jit and Nx.Batch take only tensors and numbers in a container: they traverse " <>
          "every element of a tuple, every value of a map and every field a struct's " <>
          "Nx.Container traverses. #{capitalized(place)} #{holds} #{leaf}, and Nx " <>
          "raises Protocol.UndefinedError there.",
      label: "hands in a container that #{holds} #{leaf} at #{place}",
      help:
        "pass a tensor or a number there (Nx.tensor/1 for a boolean), a tuple for a list, and " <>
          "move a value fixed at compile time out of the container: into an option, or a " <>
          "derived struct's `keep:` list",
      frame: "",
      severity: severity
    }
  end

  def call_error("captured_gradient", detail, _operation) do
    captured =
      case detail do
        "" -> "the value it differentiates"
        path -> "the part at #{path} of the value it differentiates"
      end

    %{
      title: "differentiates a value its function captures",
      detail:
        "The function reads #{captured} from its closure rather than from its argument. " <>
          "Nx treats every value a gradient's function captures as a constant, so the " <>
          "gradient through that read is zero and the result silently wrong.",
      label: "its function reads #{captured} from its closure",
      help:
        "read the value from the function's argument, as in fn {w, b} -> ... b ... end, " <>
          "not from the variable outside it",
      frame: "the captured value is made by",
      severity: :warning
    }
  end

  def call_error("container_order", detail, _operation) do
    [name, order, other] = String.split(detail, " ")
    {struct, field} = split_field(name)

    %{
      title: "traverse/3 visits .#{field} at another place than reduce/3 does",
      detail:
        "Nx flattens a container with reduce/3 and rebuilds it with traverse/3, matching " <>
          "tensors by their place in that order. #{struct}'s implementation visits .#{field} " <>
          "#{place(order)} in traverse/3 and #{place(other)} in reduce/3, so tensors land in " <>
          "each other's fields, as a while loop over the struct shows.",
      label: "visits .#{field} #{place(order)}",
      help: "visit the fields in one order in traverse/3 and reduce/3, or derive Nx.Container",
      frame: "reduce/3 visits .#{field} #{place(other)}",
      severity: :warning
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A dropped field's detail, `Struct.field=default`.
  defp dropped(detail) do
    [name, default] = String.split(detail, "=", parts: 2)
    {struct, field} = split_field(name)
    %{struct: struct, field: field, default: default}
  end

  # `Struct.field` as the struct and the field.
  defp split_field(name) do
    [field | modules] = name |> String.split(".") |> Enum.reverse()
    {modules |> Enum.reverse() |> Enum.join("."), field}
  end

  # A finding at a call of a fun, no named call, names its subject itself.
  defp subject("", title), do: "A jitted function " <> title
  defp subject(_operation, title), do: title

  # A field's place in the order a container's functions visit fields,
  # counted from 0 by the rules.
  defp place(order), do: "at place #{String.to_integer(order) + 1}"

  defp capitalized(text) do
    {first, rest} = String.split_at(text, 1)
    String.upcase(first) <> rest
  end

  # A struct with no container implementation, which the rules spell `a
  # Module struct`, as the code writes one.
  defp struct_leaf(leaf) do
    case Regex.run(~r/^a ([A-Z][\w.]*) struct$/, leaf) do
      [_, module] -> "%#{module}{}"
      nil -> leaf
    end
  end

  # A leaf's detail, `leaf at argument 2.b`, with `possibly ` in front where
  # the container holds it only on some path.
  defp certainty("possibly " <> where), do: {:warning, "can hold", where}
  defp certainty(where), do: {:error, "holds", where}
end
