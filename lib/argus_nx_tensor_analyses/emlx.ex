defmodule ArgusNxTensorAnalyses.EMLX do
  @moduledoc """
  What [EMLX](https://github.com/elixir-nx/emlx) computes differently from
  Nx's BinaryBackend and EXLA, and tensors of two backends that meet,
  found in compiled code by the Datalog program in `priv/emlx.dl`. That
  program includes the tensor shapes analysis's (`priv/tensor_shapes.dl`)
  and reads its words: the values it follows and the contexts it follows
  them in, the signs a value can have, and the types tensors are made in.

  The checks are EMLX's, so they run in their own analysis, `tensor_emlx`:
  a project on EMLX runs it (`mix argus tensor_emlx`, or `mix argus
  --all`), and a project on EXLA leaves it out.

    * A tensor made in f64 or c128, which EMLX keeps as f32 and c64 on
      either device while the tensor still says f64 or c128. Reported
      without the tensor shapes analysis's `unsupported_types` option;
      where the project lists the type there, the tensor shapes analysis
      reports it and this one does not, so no call is reported twice.
    * A remainder whose divisor or dividend can be negative, which EMLX
      computes wrongly: `Nx.remainder([-6, 7], [2, -2])` is `[-2, -1]`
      on EMLX and `[0, 1]` on BinaryBackend and EXLA.
    * An integer power whose exponent can be negative, which runs forever
      on EMLX's CPU device when run eagerly and gives 0 elsewhere on EMLX.
    * A round of a value the code's math can put exactly on a half, which
      EMLX rounds to even and BinaryBackend and EXLA away from zero.
    * An Nx call that gets tensors of two backends that cannot meet, such
      as the result of a function compiled with EXLA and a tensor made on
      EMLX: Nx raises `Nx.Defn.IncompatibleBackendsError`.

  Which way it errs: as the tensor shapes analysis, quiet where it cannot
  follow a value. An operand negative only as an input may be is reported
  as info rather than a warning. A function a compiler or
  `Nx.with_default_backend/2` runs on another backend is taken to run
  there wherever it is called, so an f64 tensor it makes is not reported
  even where it is also called on EMLX. The calls in a `defn` are not
  checked for mixed backends: whether they mix them is up to the compiler
  that runs it (EMLX's moves its arguments to EMLX, the evaluator does
  not), which is configured outside the code.

  It extracts, solves, caches and places findings as
  `ArgusNxTensorAnalyses.TensorShapes` does, through its `solve/3` and
  `run/2`.
  """

  @behaviour Argus.Analysis

  alias Argus.Findings
  alias ArgusNxTensorAnalyses.TensorShapes

  @external_resource Path.expand("../../priv/emlx.dl", __DIR__)

  @impl true
  def name, do: :tensor_emlx

  @impl true
  def description,
    do: "Nx calls EMLX computes differently from BinaryBackend and EXLA, and mixed backends"

  @impl true
  def rules_file, do: Application.app_dir(:argus_nx_tensor_analyses, "priv/emlx.dl")

  @impl true
  def extractors, do: TensorShapes.extractors()

  @impl true
  def output_relations do
    [
      %{
        name: :tensor_emlx_divergence,
        fields: [
          {:id, :instr_id, "the Nx call"},
          {:func, :func_id, "the function making it"},
          {:operation, :symbol, "the Nx function, as Nx.remainder/2"},
          {:kind, :symbol,
           "what EMLX does differently: narrowed_type, negative_remainder, negative_integer_power or round_half"},
          {:detail, :symbol,
           "the type (f64, c128), the operand that can be negative (divisor, dividend, exponent), or how the operand lands on a half (halved, half_added, mean)"},
          {:certain, :number,
           "1 where the code's own math causes it, 0 where only an input the analysis does not follow may"},
          {:origin, :symbol, "the Nx call whose math causes it, or empty"},
          {:origin_operation, :symbol, "that call's function, as Nx.subtract/2"}
        ],
        key: [:id, :kind, :detail],
        doc: "An Nx call EMLX computes differently from BinaryBackend and EXLA."
      },
      %{
        name: :tensor_emlx_mixed_backends,
        fields: [
          {:id, :instr_id, "the Nx call"},
          {:func, :func_id, "the function making it"},
          {:operation, :symbol, "the Nx function, as Nx.add/2"},
          {:backend, :symbol, "the backend of one tensor it gets, as EXLA.Backend"},
          {:origin, :symbol, "the call that puts that tensor there"},
          {:origin_operation, :symbol, "that call's function, as Nx.Defn.jit/2"},
          {:other, :symbol, "the backend of another tensor it gets, as EMLX.Backend"},
          {:other_origin, :symbol, "the call that puts that tensor there"},
          {:other_origin_operation, :symbol, "that call's function, as Nx.iota/2"}
        ],
        key: [:id, :backend, :other],
        doc: "An Nx call that gets tensors of two backends that cannot meet."
      }
    ]
  end

  @impl true
  def finding(:tensor_emlx_divergence, [id, _func, operation, kind, detail, certain | rest]) do
    [origin, shown] = rest
    wording = divergence(kind, detail)

    Findings.new(severity(kind, certain), "#{operation} #{wording.title}", wording.detail,
      at: Findings.at_instr(id),
      at_label: wording.label,
      help: [wording.help],
      related: origin_frame(wording.frame, origin, shown)
    )
  end

  def finding(:tensor_emlx_mixed_backends, [id, _func, operation, backend | rest]) do
    [origin, shown, other, other_origin, other_shown] = rest

    Findings.new(
      :warning,
      "#{operation} gets tensors of #{backend} and #{other}, which cannot meet",
      "Nx raises Nx.Defn.IncompatibleBackendsError for a call over tensors of two backends, " <>
        "unless one of them is Nx.BinaryBackend.#{compiled(shown, backend)}",
      at: Findings.at_instr(id),
      at_label: "gets #{backend} and #{other} here",
      help: [
        "move one tensor to the other's backend first, such as Nx.backend_transfer(tensor, EMLX.Backend)"
      ],
      related:
        origin_frame(placed_by(shown, backend), origin, shown) ++
          origin_frame(placed_by(other_shown, other), other_origin, other_shown)
    )
  end

  # An input the analysis cannot follow may never be negative, and a half
  # is where rounding differs, not an error, so those report as info.
  defp severity("round_half", _certain), do: :info
  defp severity(_kind, certain) when certain in [0, "0"], do: :info
  defp severity(_kind, _certain), do: :warning

  # What a compiled function's results are on, where one puts the tensor
  # on its backend.
  defp compiled(shown, backend) do
    if String.starts_with?(shown, ["Nx.Defn.", "EXLA."]),
      do: " A function compiled for #{backend} returns its tensors there, whatever it is handed.",
      else: ""
  end

  # What a frame at the call that makes a tensor on its backend says.
  defp placed_by(shown, backend) do
    if String.starts_with?(shown, ["Nx.backend_", "Nx.Defn.", "EXLA.", "Nx.with_default"]),
      do: "puts a tensor on #{backend}:",
      else: "makes a tensor on #{backend}:"
  end

  # The frame at the call a finding comes from, none for a written value.
  defp origin_frame(_frame, "", _shown), do: []

  defp origin_frame(frame, origin, shown),
    do: [Findings.related(String.trim("#{frame} #{shown}"), Findings.at_instr(origin))]

  # What each kind of finding says: its title, why, the label at the call,
  # what to change, and the frame at its origin.
  defp divergence("narrowed_type", "f64") do
    %{
      title: "makes an f64 tensor, which EMLX keeps as f32",
      detail:
        "EMLX has no 64-bit float: on either device it stores an f64 tensor as f32, while the " <>
          "tensor still says f64, so what is computed from it is computed at f32 precision " <>
          "(0.1 reads back as 0.10000000149011612, and 1 + 1.0e-10 as 1.0). BinaryBackend and " <>
          "EXLA compute in f64.",
      label: "makes f64 here",
      help:
        "compute it where f64 exists (Nx.Defn.jit(fun, compiler: EXLA), or backend: EXLA.Backend) and move the result to EMLX as f32, or make it f32",
      frame: ""
    }
  end

  defp divergence("narrowed_type", type) do
    %{
      title: "makes #{article(type)} #{type} tensor, which EMLX keeps as c64",
      detail:
        "EMLX has no 128-bit complex type: it stores #{article(type)} #{type} tensor as c64, " <>
          "while the tensor still says #{type}, computes at c64 precision, and raises reading " <>
          "it back (Nx.to_number/1, Nx.to_list/1: no function clause matching in " <>
          "EMLX.Backend.maybe_modify_binary/3).",
      label: "makes #{type} here",
      help: "make it c64, or compute it on a backend that has #{type}",
      frame: ""
    }
  end

  defp divergence("negative_remainder", "divisor") do
    %{
      title: "takes a remainder by a divisor that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, and subtracts the divisor " <>
          "where the dividend is negative: with a negative divisor that is wrong wherever the " <>
          "remainder is not zero, and wherever the dividend is negative. remainder(7, -2) is -1 " <>
          "and remainder(-7, -3) is 2 on EMLX, 1 and -1 on BinaryBackend and EXLA.",
      label: "the divisor can be negative here",
      help:
        "divide by a positive number (the absolute value, flipping the result's sign where it should), or run this remainder off EMLX",
      frame: "the divisor can be negative because of this:"
    }
  end

  defp divergence("negative_remainder", _dividend) do
    %{
      title: "takes a remainder of a dividend that can be negative, which EMLX gets wrong",
      detail:
        "EMLX takes MLX's remainder, which has the divisor's sign, and subtracts the divisor " <>
          "where the dividend is negative: where a negative dividend divides exactly, EMLX " <>
          "gives minus the divisor instead of 0. remainder(-6, 2) is -2 on EMLX, 0 on " <>
          "BinaryBackend and EXLA.",
      label: "the dividend can be negative here",
      help:
        "keep the dividend from going negative (add a multiple of the divisor first), or select 0 where the result equals minus the divisor",
      frame: "the dividend can be negative because of this:"
    }
  end

  defp divergence("negative_integer_power", _exponent) do
    %{
      title: "raises an integer to an exponent that can be negative, which hangs EMLX",
      detail:
        "An integer power with a negative exponent runs forever on EMLX's CPU device when run " <>
          "eagerly, and gives 0 on its GPU and under its compiler, for a base of 1 too. EXLA " <>
          "gives 0 (1 for a base of 1), and BinaryBackend raises.",
      label: "the exponent can be negative here",
      help:
        "make the base a float (Nx.as_type(base, :f32), or 2.0 rather than 2) for a fractional result, or keep the exponent from going negative",
      frame: "the exponent can be negative because of this:"
    }
  end

  defp divergence("round_half", cause) do
    %{
      title: "rounds values that can lie on a half, which EMLX rounds to even",
      detail:
        "EMLX rounds a value halfway between two integers to the even one (0.5 to 0, 2.5 to 2), " <>
          "BinaryBackend and EXLA away from zero (to 1 and 3), and #{half_cause(cause)}.",
      label: "rounds here",
      help:
        "round the halves the way you mean explicitly, such as Nx.floor(Nx.add(x, 0.5)) to round them up",
      frame: "puts it on a half:"
    }
  end

  defp divergence(kind, detail) do
    %{
      title: "computes something else on EMLX (#{kind})",
      detail: "The analysis reports #{kind}: #{detail}.",
      label: "here",
      help: "see what EMLX computes for this call",
      frame: "because of this:"
    }
  end

  # How the rounded value lands on a half, by the cause the program names.
  defp half_cause("halved"), do: "the value rounded is an integer halved, which lies on halves"
  defp half_cause("half_added"), do: "the value rounded is an integer plus a half, always a half"

  defp half_cause("mean"),
    do: "the value rounded is a mean of integers, which lies on a half for an even count"

  defp half_cause(_cause), do: "the value rounded can lie on a half"

  defp article(name), do: if(String.starts_with?(name, ["f", "s"]), do: "an", else: "a")

  @doc """
  Extracts the modules (atoms or `.beam` paths) and solves this analysis's
  program over them, returning its output relations' rows by name, as
  `ArgusNxTensorAnalyses.TensorShapes.solve/3` does, with its options.
  """
  @spec solve([module() | Path.t()], keyword()) :: {:ok, map()} | {:error, term()}
  def solve(modules, options \\ []), do: TensorShapes.solve(modules, rules_file(), options)

  @doc """
  Solves this analysis's program over the modules (atoms or `.beam`
  paths) and returns each finding placed at its call in the module's
  source. Takes `ArgusNxTensorAnalyses.TensorShapes.solve/3`'s options:
  give it a `:cache` directory of its own.
  """
  @spec run([module() | Path.t()], keyword()) :: {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(modules, options \\ []),
    do: TensorShapes.run(modules, Keyword.put(options, :analysis, __MODULE__))
end
