defmodule ArgusNxTensorAnalyses.TensorShapes.Wording.Consumption do
  @moduledoc false
  # What the findings of `priv/tensor_shapes/consumption.dl` say.

  use ArgusNxTensorAnalyses.TensorShapes.Wording

  @same_bits "Every Nx.Random sampler draws its bits from split(key)[1], whichever it is, so draws from one key get the same bits: equal samples, or samples tied element by element (Nx.Random.normal is the inverse normal CDF of Nx.Random.uniform over the same bits). Nothing raises."

  @freed "EMLX raises \"Tensor has been deallocated\" and EXLA \"Buffer has been deleted or donated\" reading it. The BinaryBackend frees nothing, so the code runs there and its tests pass."

  @donated "EXLA raises \"Buffer has been deleted or donated\" reading it. EMLX and the BinaryBackend do not donate, so the code runs there."

  @impl true
  def call_error("reused_key", detail, _operation) do
    {drawn, earlier} = split_draws(detail)
    {earlier, key} = named(earlier, "the key")

    %{
      title: "draws from a random key that was already drawn from",
      detail:
        "#{drawn} draws from #{key}, which #{earlier} drew from earlier in this function. " <>
          @same_bits,
      label: "draws again from #{key}",
      help:
        "draw from the new key each sampler returns ({sample, key} = Nx.Random.uniform(key)), or split the key once (Nx.Random.split(key, parts: n)) and draw from each part",
      frame: "the key was first drawn from by",
      severity: :warning
    }
  end

  def call_error("reused_seed", detail, _operation) do
    [seed, draws] = String.split(detail, " ", parts: 2)
    {drawn, earlier} = split_draws(draws)

    %{
      title: "draws from a key made of the same seed as one drawn from earlier",
      detail:
        "#{drawn} draws from Nx.Random.key(#{seed}), and #{earlier} drew from another Nx.Random.key(#{seed}) earlier in this function. Keys made of one seed are equal. #{@same_bits}",
      label: "draws from Nx.Random.key(#{seed}) again",
      help:
        "make one key and split it (Nx.Random.split(key, parts: 2)), or draw from the new key each sampler returns; give keys meant to differ different seeds",
      frame: "another Nx.Random.key(#{seed}) was first drawn from by",
      severity: :warning
    }
  end

  def call_error("captured_loop_key", drawn, _operation) do
    %{
      title: "draws from the same key on every pass of a loop",
      detail:
        "The fun a loop runs captured this key, so #{drawn} draws from the same key on every pass. #{@same_bits}",
      label: "draws from a key its fun captured, the same on every pass",
      help:
        "give each pass a key of its own: Nx.Random.fold_in(key, index) of the pass's index, a part of Nx.Random.split(key, parts: n), or a key threaded through the loop's accumulator",
      frame: "the loop runs in",
      severity: :warning
    }
  end

  def call_error("passed_back_key", drawn, _operation) do
    %{
      title: "draws from a key the loop hands back unchanged",
      detail:
        "The loop's fun draws #{drawn} from its key and returns that same key for the next pass, not the new key the draw returns, so every pass draws from one key. #{@same_bits}",
      label: "draws from its key, then returns that key for the next pass",
      help:
        "return the key the draw returns ({sample, key} = Nx.Random.uniform(key)) in the loop's state or accumulator",
      frame: "the loop runs in",
      severity: :warning
    }
  end

  def call_error("spent_key_returned", detail, _operation) do
    {drawn, key} = named(detail, "the key")

    %{
      title: "returns a random key it has already drawn from",
      detail:
        "This function draws #{drawn} from #{key} and returns the same key, not the new key the draw returns, so a caller that draws from it draws again. #{@same_bits}",
      label: "returns #{key}, a key it drew from",
      help:
        "return the key the draw returns ({sample, key} = Nx.Random.uniform(key)), or split the key and return a part the function does not draw from",
      frame: "the key was drawn from by",
      severity: :warning
    }
  end

  def call_error("used_after_transfer", detail, _operation) do
    freed_use(detail, %{
      after: "Nx.backend_transfer freed it",
      done: "freed",
      why:
        "Nx.backend_transfer copies a tensor to the other backend and frees it where it was. #{@freed}",
      help:
        "use the tensor the transfer returns (tensor = Nx.backend_transfer(tensor)), or Nx.backend_copy/2 where the original must stay",
      frame: "freed by",
      read: :error
    })
  end

  def call_error("used_after_deallocation", detail, _operation) do
    freed_use(detail, %{
      after: "Nx.backend_deallocate freed it",
      done: "freed",
      why: "Nx.backend_deallocate frees a tensor's memory. #{@freed}",
      help: "deallocate the tensor after its last use",
      frame: "freed by",
      read: :error
    })
  end

  def call_error("used_after_donation", detail, _operation) do
    freed_use(detail, %{
      after: "it was donated",
      done: "donated",
      why:
        "The tensor was marked with Nx.donatable/1 and handed to a JIT call before this, and EXLA gives a donated tensor's memory to the call's result; the mark shares its memory with the tensor it marks. #{@donated}",
      help:
        "use what the JIT call returns, or mark only a tensor the code does not read after the call",
      frame: "marked by",
      read: :warning
    })
  end

  def call_error(_kind, _detail, _operation), do: nil

  # A detail's value and the name the rules give it, `Nx.Random.uniform
  # from argument 1`, or the value alone where it has none.
  defp named(detail, unnamed) do
    case String.split(detail, " from ", parts: 2) do
      [value, name] -> {value, name}
      [value] -> {value, unnamed}
    end
  end

  # "Nx.Random.normal after Nx.Random.uniform": the draw, and the earlier one.
  defp split_draws(detail) do
    case String.split(detail, " after ", parts: 2) do
      [drawn, earlier] -> {drawn, earlier}
      [drawn] -> {drawn, "another call"}
    end
  end

  # A use of a freed tensor, `read argument 2`: a call that reads it raises
  # on the backends that free, and one that returns it or hands it on
  # raises where the caller or callee reads it.
  defp freed_use(detail, wording) do
    {how, tensor} =
      case String.split(detail, " ", parts: 2) do
        [how, tensor] -> {how, tensor}
        [how] -> {how, "the tensor"}
      end

    freed(how, Map.put(wording, :tensor, tensor))
  end

  defp freed("read", wording) do
    %{
      title: "reads a tensor after #{wording.after}",
      detail: wording.why,
      label: "reads #{wording.tensor} after it was #{wording.done}",
      help: wording.help,
      frame: wording.frame,
      severity: wording.read
    }
  end

  defp freed("handed_on", wording) do
    %{
      title: "hands on a tensor after #{wording.after}",
      detail: "The function it is handed to reads it. #{wording.why}",
      label: "hands on #{wording.tensor} after it was #{wording.done}",
      help: wording.help,
      frame: wording.frame,
      severity: :warning
    }
  end

  defp freed(_returned, wording) do
    %{
      title: "returns a tensor after #{wording.after}",
      detail: "A caller that reads what this returns reads freed memory. #{wording.why}",
      label: "returns #{wording.tensor} after it was #{wording.done}",
      help: wording.help,
      frame: wording.frame,
      severity: :warning
    }
  end
end
