defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Serving do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/serving.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @slices "A serving slices every leaf of its computation's output back along the first " <>
            "axis, the batch axis, a slice of the batch's size for each caller"

  @template "Nx raises \"argument at position 1 is not compatible with compiled function " <>
              "template\" on such a batch."

  @impl true
  def call_error("serving_scalar_output", detail, _operation) do
    %{
      title: "compiles a serving computation whose output is a scalar",
      detail:
        "#{@slices}, and #{detail}. Nx raises \"given axis (0) invalid for shape with rank 0\" " <>
          "on every batch.",
      label: "the serving's computation",
      help:
        "reduce each entry along its own axes and keep the batch axis first, such as " <>
          "Nx.sum(x, axes: [1]) rather than Nx.sum(x), or reduce in the client postprocessing",
      frame: "makes the output:",
      severity: :error
    }
  end

  def call_error("serving_output_batch_axis", detail, _operation) do
    %{
      title: "compiles a serving computation whose output drops the batch axis",
      detail:
        "#{@slices}, and #{detail}. Nx slices along whatever axis comes first, so each caller " <>
          "gets part of something else (the batch's total, another axis), or Nx raises where " <>
          "that axis is shorter than the batch.",
      label: "the serving's computation",
      help:
        "keep the batch axis first in every output: reduce, transpose and reshape only the " <>
          "entry's own axes (axes: [1], or Nx.transpose(x, axes: [0, 2, 1]))",
      frame: "makes the output:",
      severity: :warning
    }
  end

  def call_error("serving_mixes_batch", detail, _operation) do
    %{
      title: "runs along the batch axis of a serving's computation",
      detail:
        "This runs inside a serving's computation, whose first axis is the batch, and " <>
          "#{detail}. Each entry's result then depends on the rest of the batch: the zero rows " <>
          "Nx.Batch.pad adds, and under batched_run other callers' requests.",
      label: "mixes the batch's entries here",
      help:
        "reduce, sort or contract along the entry's own axes (axes: [1], axis: 1), never axis 0",
      frame: "because of this",
      severity: :warning
    }
  end

  def call_error("serving_template_batch_size", detail, _operation) do
    %{
      title: "compiles a serving computation for a batch size its batches do not have",
      detail:
        "An ahead-of-time compiled computation takes only batches of its template's shape, and " <>
          "#{detail}. #{@template}",
      label: "compiles for a fixed batch size",
      help:
        "pad every batch to the template's size before the compiled function runs, such as " <>
          "Nx.Batch.pad(batch, size - batch.size) in a function the builder returns, or jit " <>
          "instead of compiling ahead of time",
      frame: "the batch:",
      severity: severity(detail)
    }
  end

  def call_error("serving_template_type", detail, _operation) do
    [template, entries] = String.split(detail, " ")
    entries = if entries in ~w(integer float complex), do: "#{entries}s", else: entries

    %{
      title: "compiles a serving computation for another type than its batch's",
      detail:
        "An ahead-of-time compiled computation takes only batches of its template's type, and " <>
          "it is compiled for #{template} and its batch's entries are #{entries}. #{@template}",
      label: "compiles for #{template}, and the entries are #{entries}",
      help:
        "make the template's type the entries' (Nx.template(shape, Nx.type(entry))), or convert " <>
          "the entries with Nx.as_type in the client preprocessing",
      frame: "the batch:",
      severity: :error
    }
  end

  def call_error("batch_incompatible_entries", detail, _operation) do
    %{
      title: "makes a batch of entries Nx cannot join",
      detail:
        "A batch's entries must all have one template, the same shape (past the first axis for " <>
          "a concatenation) and axis names, and #{detail}. Nx raises \"cannot add to batch due " <>
          "to incompatible tensors/containers\".",
      label: "joins the entries here",
      help:
        "pad or reshape the entries to one shape first, or give entries of different shapes " <>
          "different batch keys (Nx.Batch.key/2)",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("batch_scalar_entry", detail, _operation) do
    %{
      title: "concatenates a scalar into a batch",
      detail:
        "Nx.Batch.concatenate/2 joins its entries along their first axis, and #{detail}. " <>
          "Nx raises \"cannot concatenate scalar tensor\".",
      label: "concatenates here",
      help: "stack scalars with Nx.Batch.stack/2, which gives each entry an axis of its own",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("batch_empty", detail, _operation) do
    %{
      title: "runs a serving on an empty batch",
      detail:
        "A serving runs its computation over a batch's entries, and #{detail}. Nx raises " <>
          "\"cannot run with empty Nx.Batch\".",
      label: "runs the batch here",
      help: "add entries to the batch first, with Nx.Batch.stack/2 or Nx.Batch.concatenate/2",
      frame: "makes the empty batch:",
      severity: :error
    }
  end

  def call_error("serving_entry_shape_varies", detail, _operation) do
    %{
      title: "sets a preprocessing whose batches a serving process cannot merge",
      detail:
        "A serving process merges the batches of concurrent requests into one, and #{detail}. " <>
          "Nx raises \"cannot merge batches due to incompatible templates\" there, every caller " <>
          "in the batch exits, and the serving's supervisor, which does not restart, dies with " <>
          "them. This holds only where requests' shapes differ.",
      label: "sets the preprocessing here",
      help:
        "pad or reshape each request to one shape in the preprocessing, or give each shape its " <>
          "own batch key (Nx.Batch.key/2, and :batch_keys when starting the serving)",
      frame: "makes the batch:",
      severity: :warning
    }
  end

  def call_error("serving_run_input", detail, _operation) do
    %{
      title: "runs a serving on input its default preprocessing cannot take",
      detail:
        "A serving without a client preprocessing takes an Nx.Batch, or a stream of them, and " <>
          "#{detail}. Nx raises before the computation runs.",
      label: "runs it here",
      help:
        "make a batch of the input (Nx.Batch.stack([tensor])), or set a client preprocessing " <>
          "that returns {batch, info}",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("serving_run_name", name, _operation) do
    %{
      title: "is handed a name where it takes the serving itself",
      detail:
        "Nx.Serving.run/2 runs a serving struct inline, in the calling process, and is handed " <>
          "#{name}, a serving process's name. Nx raises FunctionClauseError.",
      label: "handed #{name} here",
      help:
        "run the serving process by its name with Nx.Serving.batched_run(#{name}, input), or " <>
          "hand Nx.Serving.run/2 the serving struct",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("serving_batched_run_struct", _detail, _operation) do
    %{
      title: "is handed a serving where it takes a serving process's name",
      detail:
        "Nx.Serving.batched_run/2 sends its input to a serving process started under a name, " <>
          "and is handed a serving struct. Nx raises FunctionClauseError.",
      label: "handed the serving struct here",
      help:
        "start the serving as a process ({Nx.Serving, serving: serving, name: MyServing}) and " <>
          "call Nx.Serving.batched_run(MyServing, input), or run the struct inline with " <>
          "Nx.Serving.run(serving, input)",
      frame: "the serving is made by",
      severity: :error
    }
  end

  def call_error("serving_batch_size_conflict", detail, _operation) do
    [size, given] = String.split(detail, " ", parts: 2)

    %{
      title: "sets a batch size its serving process is started with otherwise",
      detail:
        "It sets the serving's batch size to #{size}, and the serving process is started " <>
          "with batch_size: #{given}. Nx raises \"the batch size set via " <>
          "Nx.Serving.batch_size/2 (#{size}) does not match the batch size given to the " <>
          "serving process (#{given})\" when the process starts.",
      label: "sets batch size #{size} here",
      help:
        "give the batch size once: drop batch_size: from the start options, or this " <>
          "Nx.Serving.batch_size/2 call",
      frame: "the serving process is started with batch_size: #{given} by",
      severity: :error
    }
  end

  def call_error("serving_preprocessing_result", detail, _operation) do
    %{
      title: "sets a client preprocessing that returns no {batch, info} pair",
      detail:
        "A serving's client preprocessing must return a batch, or a stream of batches, and any " <>
          "information for the postprocessing, as a pair, and #{detail}. Nx raises " <>
          "\"client_preprocessing function ... must return a two element tuple\".",
      label: "sets the preprocessing here",
      help: "return {Nx.Batch.stack([input]), info} from the preprocessing",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("serving_builder_result", detail, _operation) do
    %{
      title: "is handed a builder that returns no compiled function",
      detail:
        "Nx.Serving.new/2 takes a function of the compiler options that returns the " <>
          "computation, jitted or compiled, and #{detail}. It is called with a keyword list " <>
          "when the serving starts, and Nx raises there.",
      label: "the builder is handed here",
      help:
        "wrap the computation: Nx.Serving.new(fn options -> Nx.Defn.jit(&computation/1, options) end), " <>
          "or use Nx.Serving.jit(&computation/1)",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("serving_computation_arity", detail, _operation) do
    %{
      title: "compiles a serving computation that takes more than the batch",
      detail:
        "A serving calls its computation with the batch alone, and #{detail}. Nx raises " <>
          "\"should return an AOT or JIT compiled function that expects one argument\".",
      label: "compiles it here",
      help:
        "capture the other arguments in a closure of one argument, such as " <>
          "fn batch -> predict(params, batch) end",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error("serving_postprocessing_input", detail, _operation) do
    %{
      title: "is handed the pair a serving's postprocessing gets",
      detail:
        "A two-argument client postprocessing is handed {output, metadata} and the " <>
          "preprocessing's information, and #{detail}. Nx raises on the tuple.",
      label: "takes the pair as a tensor here",
      help:
        "match the pair in the postprocessing's head: fn {output, _metadata}, info -> ... end",
      frame: "because of this",
      severity: :error
    }
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A batch of a known size raises at once; a process serving's smaller
  # batches only when its batch times out.
  defp severity(detail),
    do: if(detail =~ "times out", do: :warning, else: :error)
end
