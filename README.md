# ArgusNxTensorAnalyses

[Argus](https://github.com/QuinnWilton/argus) analyses for code that uses
[Nx](https://github.com/elixir-nx/nx). They read a project's compiled
modules and report tensor shapes that do not fit where they meet, math
that can give an infinity or a NaN, and tensor types Nx or the backend
rejects, before the code runs.

- **Shape mismatches:** Nx calls whose operand shapes Nx rejects, such as
  shapes that do not broadcast, a `dot` over axes of different sizes, a
  reshape that changes the number of elements, or an axis the tensor does
  not have.
- **Axis misalignments:** calls Nx accepts but where the code does not line
  its axes up. For example, sizes the code names differently
  (`config.heads` against `config.kv_heads`), an unnamed axis meeting a
  named one, or contracted axes with different names.
- **Results that can be infinite or NaN:** Nx calls whose operand the
  code's own math lets reach a value the call is not defined at. For
  example, a divisor that is a sum of squares or a norm, which is zero for
  a zero vector; the logarithm of a count; the square root of a variance
  written as E[x²] − E[x]², which rounding can take below zero; a
  softmax that does not subtract the maximum first; atanh of a tanh,
  which rounds to 1 for large inputs; the arc cosine of a cosine
  similarity, which rounding can take just past 1; or a square root or
  norm that can be zero where a `grad` differentiates it.
- **Unchecked operands:** a division, logarithm, square root, or asin,
  acos, atanh, erf_inv, log1p or acosh of a value the analysis cannot see
  into, such as an input or a config field, that no test on the way to
  the call, no `Nx.select` and no clip keeps where the call is defined.
  Such a value may never get there, so these are noisier and reported as
  info.
- **Tensor types:** calls that take only integers handed a float, such as
  a bitwise operation, an integer quotient, or `Nx.take` with indices
  computed by division; and, for a project that names the types its
  backend lacks, calls that make a tensor of one, such as f64 on EMLX.

```
error[argus.tensor_shapes]: Nx.dot/2 contracts axes that do not match
   ╭─[lib/my_app/model.ex:13:5]
   │
12 │   def project(input, weight) do
13 │     Nx.dot(input, weight)
   •     ──────────┬──────────
   •               ╰── gets {4, 8} and {6, 16}
14 │   end
15 │
16 │   def model do
17 │     input = Nx.iota({4, 8})
   •     ───────────┬───────────
   •                ╰── makes {4, 8}, the first argument of Nx.dot/2
18 │     good = project(input, Nx.iota({8, 16}))
19 │     bad = project(input, Nx.iota({6, 16}))
   •     ──────────────────┬───────────────────
   •                       ╰── makes {6, 16}, the second argument of Nx.dot/2; calls MyApp.Model.project/2 with these shapes
20 │     {good, bad}
21 │   end
   │
   ╰─────
     note: Nx.dot contracts axes in pairs, one from each side, and each pair must have one size; batch axes pair up the same way. Nx raises: dot/zip expects shapes to be compatible, dimension 1 of left-side (8) does not equal dimension 0 of right-side (6).
     help: contract axes of one size: transpose or reshape an operand, or give the contraction and batch axes explicitly
```

Each finding's title says what the call does wrong, and its label shows the
shapes the call gets. Related frames point at the calls that make each
operand's shape and the calls that bring the operands into the function.
The note says which rule Nx applies, then gives the exact error Nx raises.

## Installation

Add the package to your project's dependencies. It is a development tool,
so it does not need to be part of your release:

```elixir
def deps do
  [
    {:argus_nx_tensor_analyses,
     github: "heydtn/argus_nx_tensor_analyses", only: [:dev, :test], runtime: false}
  ]
end
```

Argus runs only the analyses it ships with, so this package has its own task
that runs Argus's analyses and these together. Alias it as `argus` to use it
in place of `mix argus`:

```elixir
def project do
  [
    # ...
    aliases: [argus: "argus_nx_tensor_analyses"]
  ]
end
```

You also need [Soufflé](https://souffle-lang.github.io/install) 2.5 on
your `PATH`, as Argus needs Soufflé. The rules do not compile under 2.4,
which is what Soufflé's Ubuntu PPA installs: take 2.5's package from its
[release](https://github.com/souffle-lang/souffle/releases/tag/2.5).

## Usage

```
mix argus --all                 # every analysis, Argus's and these
mix argus tensor_shapes         # the tensor shape analysis alone
mix argus tensor_shapes ets     # it and Argus's `ets`
mix argus                       # Argus's configured analyses only
mix argus --list                # what's available
```

The task takes `mix argus`'s command line: `--format`, `--fail-above` and
`--color` apply to every finding. Without the alias, run
`mix argus_nx_tensor_analyses` with the same arguments.

The analyses read your project's own modules, not its dependencies. Results
are kept under `_build/<env>/argus_nx_tensor_analyses` and reused until the
compiled code changes, so an edit to a comment or a doc solves nothing again.

A project whose backend lacks some tensor types names them in its
`mix.exs`, and every call that makes a tensor of one is reported:

```elixir
def project do
  [
    # ...
    argus_nx_tensor_analyses: [unsupported_types: [:f64]]
  ]
end
```

## How it works

`ArgusNxTensorAnalyses.TensorShapes.ShapeFlow`, an Argus extractor,
summarizes where each function's values come from: its parameters, the
calls it makes, the terms it builds and the fields it reads. A Soufflé
program, `priv/tensor_shapes.dl`, runs over those summaries and Argus's own
facts. It applies the rules `Nx.Shape` applies, operation by operation, and
follows shapes through the program to where tensors meet.

- **Sizes are symbolic.** A size is known where the code writes it
  (`Nx.iota({2, 3})`). It is a variable where the code reads it from a
  parameter (`config.heads`), and otherwise not known. Sizes multiply and
  divide symbolically, so a reshape's `:auto` is inferred where the
  variables cancel.
- **Analysis crosses functions.** Shapes cross calls and returns, `defn`
  calls between modules, closures (where `cond` and `if` put a `defn`'s
  branches), protocol dispatch to the project's own implementations, and
  funs the code captures and calls. A function is analyzed once for each
  set of arguments it is handed. A call that hands values naming size
  variables, or terms the caller built, runs its callee in a context of
  its own, up to five calls deep.
- **Signs follow the math.** For what a call divides by, or takes the
  logarithm or root of, the rules work out whether the value can be
  negative, zero or positive from how it is computed: a square is never
  negative, an exponential never zero, an iota starts at zero, and an
  input can be anything. A finding says why its operand can be zero and
  points at the call that makes it so. A test on the way to the call
  (`if n == 0`, `n > 0`) or a select on a comparison with zero
  (`Nx.select(Nx.equal(d, 0), 1, d)`) checks the operand it tests.
- **Ranges follow the math.** For asin, acos, atanh, erf_inv, log1p and
  acosh, the rules work out where the operand can lie against ±1: a tanh,
  erf or sigmoid rounds to exactly ±1 for large inputs, a sine or cosine
  reaches it, a clip reaches its bounds, and a vector over its norm or a
  cosine similarity is within ±1 only before rounding. A clip or a test on
  the way to the call keeps the operand where it is.
- **Gradients.** A function handed to `Nx.Defn.grad` or `value_and_grad`,
  and whatever it calls, is differentiated. A square root, root power or
  norm there whose result can be zero has an infinite or NaN derivative.
- **Types follow the math.** For the calls that take only integers, the
  rules work out whether each operand can be an integer, a float or a
  complex number, as `Nx.Type` has it: a float operand makes a float,
  division and the transcendental functions make one, comparisons and
  indices give integers, and a `type:` gives its own. A tensor whose type
  the code does not show has none, and nothing is reported of it.
- **Helpers take what the program hands them.** A function the program
  enters only through its own calls takes, in each context, what those
  calls hand it. A function the program is entered at from outside (one
  nothing in the project calls, or one handed out as a fun) takes inputs.
- **Literal tests pick branches.** A `case` on an option the caller writes
  takes the branch the option names, and two dispatches on one value in a
  function run one implementation. Other branches are not told apart: a
  value that reaches a call along several paths has every shape it can
  arrive with.

A mismatch is an error when some chain of calls reaching the call brings it
no other operands. It is a warning when the operands also arrive otherwise,
since not every combination may occur. Misalignments and results that can
be infinite or NaN are warnings, and unchecked operands are info. An
operand that takes only integers is an error where some context hands it
nothing else, and a warning otherwise; a type the backend lacks is an
error.

The rules model Nx 1.0. The test suite runs each of its cases through Nx
itself and checks that the analysis agrees with what Nx computes or raises.

## Limits

- A shape that depends on runtime data (a tensor read from a file, a size
  computed from a value the code never writes) is not known. Operations
  over it give no shape and no findings.
- The analysis follows the project's own code. A call into a dependency
  other than Nx gives a value that is not known, apart from the lookups it
  models: `Map.get`, `Map.fetch!`, `Keyword.get`, `Keyword.fetch!` and
  `Access.get`.
- Branches that no literal test separates are merged, so a finding on
  such a path is reported as a warning rather than an error.
- A check has to test the operand itself: `if n > 0` checks
  `Nx.divide(t, n)`, not `Nx.divide(t, Nx.multiply(t, n))`.
- A function the project calls is judged by what the project hands it,
  even where code outside the project calls it too.
- Only the `grad` calls in the project are seen: a training library that
  differentiates a function the project hands it does not make that
  function differentiated here. A `custom_grad` is not modeled.
- A clip's bounds keep a value in range only where they are written
  numbers.

## Development

```
mix deps.get
mix test
```

The tests need Soufflé 2.5 on `PATH`.
`test/argus_nx_tensor_analyses/tensor_shapes_test.exs` compiles its
fixtures into a temporary directory, runs each case through Nx and through
the analysis, and checks that they agree.

## License

MIT. See [LICENSE](LICENSE).
