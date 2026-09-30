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
    * A tensor of either type made off EMLX that `Nx.backend_transfer/2`
      or `Nx.backend_copy/2` moves onto it: EMLX keeps an f64 as f32 while
      it still says f64, and raises taking a c128 in.
    * A remainder whose divisor or dividend can be negative, which EMLX
      computes wrongly: `Nx.remainder([-6, 7], [2, -2])` is `[-2, -1]`
      on EMLX and `[0, 1]` on BinaryBackend and EXLA.
    * An integer power whose exponent can be negative, which runs forever
      on EMLX's CPU device when run eagerly and gives 0 elsewhere on EMLX.
    * A round of a value the code's math can put exactly on a half, which
      EMLX rounds to even and BinaryBackend and EXLA away from zero.
    * A shift by a written amount at or past the width EMLX shifts the
      type in, which it takes modulo that width: `Nx.left_shift(1, 33)`
      of an s32 is 2 on EMLX and 0 on BinaryBackend and EXLA.
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

  alias ArgusNxTensorAnalyses.EMLX.Wording
  alias ArgusNxTensorAnalyses.Finding
  alias ArgusNxTensorAnalyses.Solve
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
           "what EMLX does differently: narrowed_type, narrowed_transfer, negative_remainder, negative_integer_power, round_half or wrapped_shift"},
          {:detail, :symbol,
           "the type (f64, c128), the operand that can be negative (divisor, dividend, exponent), how the operand lands on a half (halved, half_added, mean), or the type and the amount shifted (s32 33)"},
          {:certain, :number,
           "1 where the code's own math causes it, 0 where only an input the analysis does not follow may"},
          {:cause, :symbol,
           "how, where a label shows it: the number written for an operand that can be negative (-1), or how it can be (subtract, input), or the class and number that put a round's operand on a half (divide 2, add 0.5); else empty"},
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
  def finding(
        :tensor_emlx_divergence = relation,
        [_id, _func, operation, kind, detail, _certain, cause | _rest] = row
      ) do
    wording =
      Wording.divergence(kind, detail, cause, operation) || unknown_divergence(kind, detail)

    Finding.build(relation, row, wording)
  end

  def finding(
        :tensor_emlx_mixed_backends = relation,
        [id, _func, operation, backend | rest] = row
      ) do
    [origin, shown, other, other_origin, other_shown] = rest
    wording = Wording.mixed_backends(backend, other, shown)

    related =
      Finding.origin_frame(Wording.placed_by(shown, backend), origin, shown) ++
        Finding.origin_frame(Wording.placed_by(other_shown, other), other_origin, other_shown)

    Finding.build(wording, id, operation, Finding.severity(relation, row, wording), related)
  end

  # A kind the program has and the wording does not describe still reads.
  defp unknown_divergence(kind, detail) do
    %{
      title: "computes something else on EMLX (#{kind})",
      detail: "The analysis reports #{kind}: #{detail}.",
      label: "computes #{kind} (#{detail}) on EMLX",
      help: "see what EMLX computes for this call",
      frame: "because of this:"
    }
  end

  @doc """
  Extracts the modules (atoms or `.beam` paths) and solves this analysis's
  program over them, returning its output relations' rows by name, as
  `ArgusNxTensorAnalyses.TensorShapes.solve/3` does, with its options.
  """
  @spec solve([module() | Path.t()], keyword()) :: {:ok, map()} | {:error, term()}
  def solve(modules, options \\ []), do: Solve.solve(__MODULE__, modules, rules_file(), options)

  @doc """
  Solves this analysis's program over the modules (atoms or `.beam`
  paths) and returns each finding placed at its call in the module's
  source. Takes `ArgusNxTensorAnalyses.TensorShapes.solve/3`'s options:
  give it a `:cache` directory of its own.
  """
  @spec run([module() | Path.t()], keyword()) :: {:ok, [Argus.Located.t()]} | {:error, term()}
  def run(modules, options \\ []), do: Solve.run(__MODULE__, modules, options)
end
