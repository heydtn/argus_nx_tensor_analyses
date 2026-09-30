# How it works

`ArgusNxTensorAnalyses.TensorShapes.ShapeFlow`, an Argus extractor,
summarizes where each function's values come from: its parameters, the
calls it makes, the terms it builds and the fields it reads, plus the
Elixir operators it applies and the order of its calls. A Soufflé program,
`priv/tensor_shapes.dl`, runs over those summaries and Argus's own facts.
It includes one file per area under `priv/tensor_shapes/`. It applies the
rules `Nx.Shape` and `Nx.Type` apply, operation by operation, and follows
values through the program to where they meet. The `emlx` checks'
program, `priv/emlx.dl`, includes this one and outputs its findings too,
so a run that reports `emlx` solves that program alone.

- **Sizes are symbolic.** A size is known where the code writes it
  (`Nx.iota({2, 3})`). It is a variable where the code reads it from a
  parameter (`config.heads`), and otherwise not known. Sizes multiply and
  divide symbolically, so a reshape's `:auto` is inferred where the
  variables cancel.
- **Analysis crosses functions.** Values cross calls and returns, `defn`
  calls between modules, closures (where `cond` and `if` put a `defn`'s
  branches, and `while` its body), protocol dispatch to the project's own
  implementations, and funs the code captures and calls. A function is
  analyzed once for each set of arguments it is handed. A call that hands
  values naming size variables, or terms the caller built, runs its callee
  in a context of its own, up to five calls deep.
- **Helpers take what the program hands them.** A function the program
  enters only through its own calls takes, in each context, what those
  calls hand it. A function the program is entered at from outside (one
  nothing in the project calls, or one handed out as a fun) takes inputs.
- **Signs follow the math.** For what a call divides by, takes the
  logarithm or root of, or uses as an index, the rules work out whether
  the value can be negative, zero or positive from how it is computed: a
  square is never negative, an exponential never zero, an iota starts at
  zero, an epsilon the tensor's type rounds to zero (1.0e-12 in f16) keeps
  nothing from zero, and an input can be anything. A finding says why its operand can
  be zero or negative, and points at the call that makes it so. A test on
  the way to the call (`if n == 0`, `n > 0`) or a select on a comparison
  with zero (`Nx.select(Nx.equal(d, 0), 1, d)`) checks the operand it
  tests.
- **Ranges follow the math.** For asin, acos, atanh, erf_inv, log1p and
  acosh, the rules work out where the operand can lie against ±1: a tanh,
  erf or sigmoid rounds to exactly ±1 for large inputs, a sine or cosine
  reaches it, a clip reaches its bounds, and a vector over its norm or a
  cosine similarity is within ±1 only before rounding. A clip or a test on
  the way to the call keeps the operand where it is.
- **Types follow `Nx.Type`.** Each value can be an integer, a float or a
  complex number, and, where a check needs it, an exact type (`u8`,
  `bf16`, `f32`). Types merge as `Nx.Type.merge/2` merges them, and a
  number meets a tensor as `merge_number/2` has it. A type the code reads
  from configuration is checked as each float type the project lists
  (`float_types:`), and a call that breaks a check in several of them is
  reported once, naming each: once for each operand, for a type error, and
  once for each type, for a pad value of such a type. A tensor whose type
  the code does not show has none, and nothing is reported of it.
- **Traced code is known.** The rules follow which code Nx traces rather
  than runs: `defn` bodies and what they reach, and funs handed to
  `Nx.Defn.jit`, `jit_apply`, `compile`, `grad` and `value_and_grad`, or
  to EXLA's `jit`, `jit_apply` and `compile`. Reading a tensor's data
  there, or calling Elixir operators on a tensor outside it, is
  reported.
- **Gradients.** A function handed to a grad, and whatever it calls, is
  differentiated. Values made from the variable it differentiates are
  followed up to `stop_grad`, and a `custom_grad` replaces the gradient of
  what it wraps.
- **Literal tests pick branches.** A `case` on an option the caller writes
  takes the branch the option names, and two dispatches on one value in a
  function run one implementation. Other branches are not told apart: a
  value that reaches a call along several paths has every shape it can
  arrive with.

A finding is an error where Nx raises, or the result is wrong, on every
path the analysis sees; a warning where it can happen, since branches the
analysis cannot tell apart may never combine; and info where it depends
on an input nothing checks, or on values the code does not show.
[checks.md](checks.md) gives each finding's severity, and what "certain"
and "on some path" mean.

The rules model Nx 1.0. The test suite runs each of its cases through Nx
itself and checks that the analysis agrees with what Nx computes or
raises.

## Limits

- A value that depends on runtime data (a tensor read from a file, a size
  computed from a value the code never writes) is not known. Operations
  over it give no shape and no findings.
- The analysis follows the project's own code. A call into a dependency
  other than Nx gives a value that is not known, apart from the lookups it
  models: `Map.get`, `Map.fetch!`, `Keyword.get`, `Keyword.fetch!` and
  `Access.get`.
- Configuration is not in the compiled code. The `emlx` checks take EMLX
  as the default backend unless the code calls `Nx.default_backend/1`, and
  which compiler runs a `defn` is not known.
- Branches that no literal test separates are merged, so a finding on
  such a path is reported as a warning rather than an error.
- A check has to test the operand itself: `if n > 0` checks
  `Nx.divide(t, n)`, not `Nx.divide(t, Nx.multiply(t, n))`.
- A function the project calls is judged by what the project hands it,
  even where code outside the project calls it too.
- Only the `grad` calls in the project are seen: a training library that
  differentiates a function the project hands it does not make that
  function differentiated here.
- A clip's bounds keep a value in range only where they are written
  numbers.
- A padding configuration the code builds counts where each amount is a
  number in a context. An amount of a size variable leaves the axis it
  pads a size not known, and one the analysis has no value for, such as a
  difference of two variables (`target - n`), leaves the call no shape.
- In a `defn`, a Kernel operator on two numbers gives a number. Its value
  is known for `+`, `-`, `*`, `div`, `rem`, `max` and `min` of integers
  the analysis knows; any other is taken as a scalar.
