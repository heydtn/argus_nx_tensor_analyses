defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Options do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/options.dl` say.
  #
  # A finding lists the keys a function takes and the atoms an option takes
  # as the rules' tables have them: this module reads those tables from the
  # rules when it compiles.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  import ArgusNxTensorAnalyses.Text

  @rules Path.expand("../../../../priv/tensor_shapes/options.dl", __DIR__)
  @external_resource @rules
  @source File.read!(@rules)

  # The keys each function takes: `option_keys("Nx.sum", " axes keep_axes ").`
  @option_keys ~r/^option_keys\("([^"]+)", " ([^"]*) "\)\./m
               |> Regex.scan(@source)
               |> Map.new(fn [_fact, operation, keys] -> {operation, String.split(keys)} end)

  # The atoms each option takes, by function and key, and `:padding`'s for
  # the functions `padded_window/1` lists.
  @padded for [_fact, operation] <- Regex.scan(~r/padded_window\("([^"]+)"\)/, @source),
              into: %{},
              do: {{operation, "padding"}, [":valid", ":same"]}
  @option_atoms ~r/^option_atoms\("([^"]+)", "([^"]+)", " ([^"]*) "\)\./m
                |> Regex.scan(@source)
                |> Map.new(fn [_fact, operation, key, atoms] ->
                  {{operation, key}, String.split(atoms)}
                end)
                |> Map.merge(@padded)

  # What an option takes besides the atoms the rules check.
  @other_values %{
    "length" => ["a positive integer"],
    "padding" => ["a list of {low, high} per spatial axis"]
  }

  @padding_types for [_fact, atom] <- Regex.scan(~r/padding_type\("([^"]+)"\)/, @source),
                     do: atom

  # Keys other libraries or a slip of the hand use for the ones Nx takes,
  # most likely first.
  @key_aliases %{
    "axis" => ["axes"],
    "axes" => ["axis"],
    "dim" => ["axis", "axes"],
    "dims" => ["axes", "axis"],
    "dimension" => ["axis", "axes"],
    "dimensions" => ["axes", "axis"],
    "keepdims" => ["keep_axes", "keep_axis"],
    "keepdim" => ["keep_axis", "keep_axes"],
    "keep_dims" => ["keep_axes", "keep_axis"],
    "keep_axes" => ["keep_axis"],
    "keep_axis" => ["keep_axes"],
    "stride" => ["strides"],
    "num" => ["n"],
    "steps" => ["n"],
    "tol" => ["atol", "rtol"],
    "full_matrices" => ["full_matrices?"],
    "k" => ["offset"],
    "offset" => ["k"],
    "diagonal" => ["offset", "k"],
    "dtype" => ["type"],
    "n" => ["order", "length"],
    "descending" => ["direction"],
    "dilation" => ["kernel_dilation", "window_dilations"],
    "dilations" => ["kernel_dilation", "window_dilations"],
    "window_dilation" => ["window_dilations"],
    "groups" => ["feature_group_size"],
    "size" => ["shape", "length"],
    "correction" => ["ddof"],
    "unbiased" => ["ddof"]
  }

  # Atoms other libraries use for the ones Nx takes, or a hint where Nx
  # has no atom for what they mean.
  @atom_aliases %{
    ":descending" => ":desc",
    ":ascending" => ":asc",
    ":last" => ":high",
    ":first" => ":low",
    ":full" => ":complete",
    ":r" => ":reduced",
    ":economic" => ":reduced",
    ":adjoint" => ":conjugate",
    ":transposed" => ":transpose",
    ":chol" => ":cholesky",
    ":edge" => ":replicate",
    ":wrap" => ":cyclic",
    ":circular" => ":cyclic",
    ":next_power_of_two" => ":power_of_two",
    ":pow2" => ":power_of_two"
  }

  @impl true
  # Options a caller hands down, found at the caller's call: the detail
  # names the Nx function that raises for them before what breaks them
  # (`Nx.sum axis`), and the finding says what the call hands it.
  def call_error(kind, "Nx." <> _handed = detail, _operation)
      when kind in ["unknown_option", "options_not_keyword", "option_form", "option_value"] do
    [name, rejected] = String.split(detail, " ", parts: 2)
    shown = if kind == "unknown_option", do: ":#{rejected}", else: rejected
    wording = call_error(kind, rejected, name)

    %{
      wording
      | title: String.replace_prefix(wording.title, "gets ", "hands #{name} "),
        label: String.replace_prefix(wording.label, "gets ", "hands #{name} "),
        frame: "raises for #{shown} in"
    }
  end

  def call_error("unknown_option", key, operation) do
    name = without_arity(operation)
    keys = Map.get(@option_keys, name, [])

    %{
      title: "gets :#{key}, an option it does not take",
      detail:
        "#{name} checks every key of its options against the ones it takes " <>
          "(#{spell_keys(keys)}), and raises for any other: unknown key :#{key}.",
      label: "gets :#{key} here",
      help: key_help(key, keys, name),
      frame: "the options come from here"
    }
  end

  def call_error("options_not_keyword", list, operation) do
    name = without_arity(operation)
    keys = Map.get(@option_keys, name, [])

    %{
      title: "gets options that are not a keyword list",
      detail:
        "#{name} takes its options as a keyword list of #{spell_keys(keys)}, and raises " <>
          "for #{list}: expected a keyword list.",
      label: "gets #{list} as its options",
      help: positional_help(list, keys, name),
      frame: "the options come from here"
    }
  end

  def call_error("option_form", detail, operation) do
    {key, value} = split_detail(detail)
    form_wording(key, value, without_arity(operation))
  end

  def call_error("option_value", detail, operation) do
    name = without_arity(operation)

    case split_detail(detail) do
      {"", atom} -> value_wording(name, "padding type", @padding_types, atom)
      {key, atom} -> value_wording(name, key, Map.get(@option_atoms, {name, key}), atom)
    end
  end

  def call_error(_kind, _detail, _operation), do: nil

  defp form_wording("axes", value, name) do
    %{
      title: "gets :axes as one axis rather than a list",
      detail:
        "#{name} takes :axes as a list of axes (or nil for all of them), and raises for " <>
          "anything else, here axes: #{value} (in Nx.Shape.normalize_axes/4).",
      label: "gets axes: #{value} here",
      help: "wrap the axis in a list: axes: #{as_list(value)}",
      frame: "the value comes from here"
    }
  end

  defp form_wording(key, value, name) do
    %{
      title: "gets :#{key} as a list rather than one axis",
      detail:
        "#{name} takes :#{key} as one axis, an index or a name, and raises for a list or " <>
          "tuple of them (given axis ... invalid), here #{key}: #{value}.",
      label: "gets #{key}: #{value} here",
      help:
        "give one axis, such as #{key}: 0; for several axes, call it once per axis or use a " <>
          "function that takes :axes",
      frame: "the value comes from here"
    }
  end

  defp value_wording(name, key, nil, atom) do
    %{
      title: "gets #{atom} for :#{key}, which it requires",
      detail: "#{name} requires :#{key}, and raises for nil or false: missing option :#{key}.",
      label: "gets #{key}: #{atom} here",
      help: "give :#{key} a number, or leave it out for the default",
      frame: "the value comes from here"
    }
  end

  defp value_wording(name, key, atoms, atom) do
    shown = if key == "padding type", do: "as its #{key}", else: "for :#{key}"
    label = if key == "padding type", do: "gets #{atom} here", else: "gets #{key}: #{atom} here"
    taken = Enum.concat(atoms, Map.get(@other_values, key, []))

    %{
      title: "gets #{atom} #{shown}, which it does not take",
      detail:
        "#{name} takes #{spell_atoms(taken)} #{shown}, and raises for any other atom, " <>
          "here #{atom}.",
      label: label,
      help: atom_help(key, atoms, atom),
      frame: "the value comes from here"
    }
  end

  # What to write instead of a key the function does not take: a key it
  # takes that the one written is likely meant for, or the keys it takes.
  defp key_help(key, keys, name) do
    case {suggested_key(key, keys), key, name} do
      {"axes", _key, _name} ->
        "use :axes, which takes a list of axes: axes: [...] rather than #{key}: ..."

      {"axis", _key, _name} ->
        "use :axis, which takes one axis: axis: ... rather than #{key}: [...]"

      {nil, "type", "Nx.Constants." <> _constant} ->
        "drop :type: the type is #{name}'s first argument"

      {nil, _key, _name} ->
        "drop :#{key}: #{name} takes #{spell_keys(keys)}"

      {suggestion, _key, _name} ->
        "use :#{suggestion}, the option #{name} takes for this"
    end
  end

  defp suggested_key(key, keys) do
    aliased = @key_aliases |> Map.get(key, []) |> Enum.find(&(&1 in keys))

    aliased ||
      keys
      |> Enum.map(&{String.jaro_distance(&1, key), &1})
      |> Enum.filter(fn {distance, _key} -> distance >= 0.85 end)
      |> Enum.max(fn -> {0, nil} end)
      |> elem(1)
  end

  defp positional_help(list, keys, name) do
    cond do
      name == "Nx.Random.choice" ->
        "give the probabilities as a tensor, Nx.tensor(#{list}): a list is taken for options"

      name == "Nx.covariance" ->
        "give the mean as a tensor, Nx.tensor(#{list}): a list is taken for options"

      "axes" in keys ->
        "name the axes with the option: axes: #{list}"

      true ->
        "write the options as key: value pairs of #{spell_keys(keys)}"
    end
  end

  defp atom_help(key, atoms, atom) do
    aliased = Map.get(@atom_aliases, atom)

    cond do
      aliased in atoms ->
        "use #{aliased}, the atom Nx has for this"

      key == "padding type" ->
        "use one of #{spell_atoms(atoms)}, or give a number to pad with that constant"

      key == "padding" ->
        "use :valid or :same, or give the padding of each spatial axis as a list of {low, high}"

      atoms == ["nil"] ->
        "give :#{key} as #{@other_values |> Map.get(key, ["a number"]) |> hd()}, or leave it out"

      true ->
        "use one of #{spell_atoms(atoms)}"
    end
  end

  # The value of `axes: 1` as a list: `[1]`, `{0, 1}` as `[0, 1]`, and a
  # value the code computes as `[axis]`.
  defp as_list("{" <> rest), do: "[" <> String.replace_suffix(rest, "}", "]")
  defp as_list("a" <> _computed), do: "[axis]"
  defp as_list(value), do: "[#{value}]"

  defp split_detail(detail) do
    case String.split(detail, ": ", parts: 2) do
      [key, value] -> {key, value}
      [value] -> {"", value}
    end
  end

  defp spell_keys([]), do: "no options"
  defp spell_keys(keys), do: keys |> Enum.map(&":#{&1}") |> join("and")

  defp spell_atoms(atoms), do: join(atoms, "or")
end
