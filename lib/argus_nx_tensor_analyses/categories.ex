defmodule ArgusNxTensorAnalyses.Categories do
  @moduledoc false
  # The categories the findings are reported under, each an analysis to
  # Argus (`error[argus.nx_shapes]`), which a run selects by name: what it
  # finds, whether a run that names none runs it, and the kinds of finding
  # it holds. A finding is in its kind's category; a report relation whose
  # rows carry no kind is a kind of its own.

  # Each category, in the order a run lists them, with the kinds it holds.
  @categories [
    nx_shapes: %{
      description:
        "operands whose shapes Nx rejects, tensor access it rejects, and tuples used as tensors",
      default: true,
      kinds: ~w(broadcast axis dot reshape concatenate stack squeeze transpose flatten
        slice put_slice pad split rank square solve least_squares diagonal diff gather indexed
        take_along_axis top_k conv window linspace weighted_mean vectorize vectorized_axes scalar
        needs_axes batch batch_size nonpositive no_tensors tensor_as_shape ragged_data
        cropped_axis irfft key sampler_parameters choice multivariate_normal norm scalar_iota
        odd_irfft transposed_weights scaling_factor_rank squeeze_input_size access_scalar
        access_out_of_bounds access_negative_step access_empty_range access_too_many_indices
        access_float_index access_float_index_tensor
        access_tensor_in_list access_update access_index_clamped tuple_as_tensor
        tensors_in_tensor_data)
    },
    nx_names: %{
      description:
        "axis names and sizes that do not line up: names Nx rejects, names and sizes the code mixes, reshape orders",
      default: true,
      kinds: ~w(names access_unknown_name access_duplicate_name size_variables unnamed_axis
        contracted_names vectorize_name reshape_order)
    },
    nx_math: %{
      description: "math that can give an infinity or a NaN, and operands nothing checks",
      default: true,
      kinds: ~w(divide_by_zero log_of_zero root_of_negative log_of_negative exp_overflow
        outside_domain infinite_at_edge log_base_one nan_comparison unchecked_divisor
        unchecked_logarithm unchecked_root unchecked_domain)
    },
    nx_types: %{
      description:
        "types Nx rejects, values a type cannot hold, lossy casts, silent type changes",
      default: true,
      kinds: ~w(non_integer_operand unsupported_type complex_operand complex_spread
        integer_past_s32 atom_type_rejected atom_type_misread invalid_type literal_wraps
        literal_overflows literal_flushes float_as_integer unsigned_wraparound count_wraparound
        narrow_wraparound index_wraparound sequence_precision cast_wraparound complex_to_real
        unchecked_cast_wraparound float_truncation literal_overflow literal_underflow cast_overflow float_sum_overflow
        unsigned_logsumexp upcast narrowing_merge pad_type_mismatch integer_determinant
        f32_precision constant_type integer_negative_power rounded_logarithm)
    },
    nx_options: %{
      description:
        "option keys a function does not take, and options of a form or value Nx rejects",
      default: true,
      kinds: ~w(unknown_option options_not_keyword option_form option_value conv_options
        window_options norm_axes)
    },
    nx_indices: %{
      description: "indices that can go negative, slices Nx clamps, and empty ranges",
      default: true,
      kinds: ~w(negative_index negative_slice_start slice_past_end non_integer_start
        ddof_not_below_count negative_ddof random_range_empty random_range_reversed
        random_range_outside_type random_type_not_integer random_bound_truncated)
    },
    nx_traced: %{
      description:
        "tensor data read while Nx traces, Elixir operators on tensors, and defn control flow",
      default: true,
      kinds: ~w(data_read_in_trace jit_in_trace compiled_template template_computed
        captured_tensor tensor_arithmetic tensor_boolean tensor_comparison tensor_truth
        predicate_shape predicate_value branch_shapes branch_broadcast branch_structure
        cond_fallthrough traced_raise while_shape while_type closure_captures_tensor
        atom_operand tensor_as_integer dropped_print)
    },
    nx_gradients: %{
      description:
        "gradients that come out NaN, infinite or not at all, and custom_grad mistakes",
      default: true,
      kinds: ~w(infinite_gradient gradient_at_origin gradient_at_edge masked_gradient
        exponent_gradient sigmoid_gradient_overflow degenerate_gradient singular_gradient
        no_gradient complex_gradient custom_grad_not_list custom_grad_short
        custom_grad_unlisted gradient_of_container)
    },
    nx_containers: %{
      description: "container fields defn drops, leaves it rejects, and grads of captured values",
      default: true,
      kinds: ~w(dropped_field_read dropped_field_returned container_leaf captured_gradient
        container_order)
    },
    nx_random: %{
      description:
        "random keys and seeds drawn from twice, captured or returned spent, and repeated draws",
      default: true,
      kinds: ~w(reused_key reused_seed captured_loop_key passed_back_key spent_key_returned
        shared_draw)
    },
    nx_freed: %{
      description:
        "tensors read or handed on after a transfer, deallocation or donation frees them",
      default: true,
      kinds: ~w(used_after_transfer used_after_deallocation used_after_donation)
    },
    nx_serving: %{
      description:
        "servings that mix or drop the batch, templates that do not fit it, and API misuse",
      default: true,
      kinds: ~w(serving_scalar_output serving_output_batch_axis serving_mixes_batch
        serving_template_batch_size serving_template_type batch_incompatible_entries
        batch_scalar_entry batch_empty serving_entry_shape_varies serving_run_input
        serving_run_name serving_batched_run_struct serving_batch_size_conflict
        serving_preprocessing_result serving_builder_result serving_computation_arity
        serving_postprocessing_input)
    },
    emlx: %{
      description:
        "what EMLX computes differently from BinaryBackend and EXLA, and mixed backends",
      default: false,
      kinds: ~w(narrowed_type narrowed_transfer negative_remainder negative_integer_power
        round_half wrapped_shift tensor_emlx_mixed_backends)
    }
  ]

  @by_kind for {category, %{kinds: kinds}} <- @categories,
               kind <- kinds,
               into: %{},
               do: {kind, category}

  # The categories, in the order a run lists them.
  @spec names() :: [atom()]
  def names, do: Keyword.keys(@categories)

  # The categories a run that names none runs.
  @spec default() :: [atom()]
  def default, do: for({category, %{default: true}} <- @categories, do: category)

  # What a category finds, in a phrase.
  @spec description(atom()) :: String.t()
  def description(category), do: Keyword.fetch!(@categories, category).description

  # The kinds a category holds.
  @spec kinds(atom()) :: [String.t()]
  def kinds(category), do: Keyword.fetch!(@categories, category).kinds

  # The rows of the engine's output relations (`rows`, by relation name),
  # split by the category each is in: the categories that hold any, in
  # order, each with its rows by relation, in their order. An evidence
  # relation's rows carry the kind of the finding they join, so each stays
  # with it. Rows of relations the engine does not declare are left out.
  @spec split(module(), %{String.t() => [list()]}) :: [{atom(), %{String.t() => [list()]}}]
  def split(engine, rows) do
    placed =
      for %{name: name, fields: fields} <- engine.output_relations(),
          relation = Atom.to_string(name),
          kind_at <- [Enum.find_index(fields, &(elem(&1, 0) == :kind))],
          row <- Map.get(rows, relation, []),
          do: {category(relation, kind_at, row), {relation, row}}

    by_category = Enum.group_by(placed, &elem(&1, 0), &elem(&1, 1))

    for category <- names(),
        {:ok, category_rows} <- [Map.fetch(by_category, category)],
        do: {category, Enum.group_by(category_rows, &elem(&1, 0), &elem(&1, 1))}
  end

  # The category of a row of `relation`: its kind's, where the relation's
  # fields have a kind at `kind_at`, else the relation's own.
  defp category(relation, nil, _row), do: Map.fetch!(@by_kind, relation)
  defp category(_relation, kind_at, row), do: Map.fetch!(@by_kind, Enum.at(row, kind_at))
end
