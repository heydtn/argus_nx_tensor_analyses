# ArgusNxTensorAnalyses

[Argus](https://github.com/QuinnWilton/argus) analyses for code that uses
[Nx](https://github.com/elixir-nx/nx). They read a project's compiled
modules and report tensor shapes that do not fit where they meet, before
the code runs.

- **Shape mismatches:** Nx calls whose operand shapes Nx rejects, such as
  shapes that do not broadcast, a `dot` over axes of different sizes, a
  reshape that changes the number of elements, or an axis the tensor does
  not have.
- **Axis misalignments:** calls Nx accepts but where the code does not line
  its axes up. For example, sizes the code names differently
  (`config.heads` against `config.kv_heads`), an unnamed axis meeting a
  named one, or contracted axes with different names.

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

You also need [Soufflé](https://souffle-lang.github.io/install) on your
`PATH`, as Argus does.

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
- **Literal tests pick branches.** A `case` on an option the caller writes
  takes the branch the option names, and two dispatches on one value in a
  function run one implementation. Other branches are not told apart: a
  value that reaches a call along several paths has every shape it can
  arrive with.

A mismatch is an error when some chain of calls reaching the call brings it
no other operands. It is a warning when the operands also arrive otherwise,
since not every combination may occur. Misalignments are warnings.

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

## Development

```
mix deps.get
mix test
```

The tests need Soufflé on `PATH`.
`test/argus_nx_tensor_analyses/tensor_shapes_test.exs` compiles its
fixtures into a temporary directory, runs each case through Nx and through
the analysis, and checks that they agree.

## License

MIT. See [LICENSE](LICENSE).
