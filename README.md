# ArgusNxTensorAnalyses

[Argus](https://github.com/QuinnWilton/argus) analyses for code that uses
[Nx](https://github.com/elixir-nx/nx). They read a project's compiled
modules and report, before the code runs, calls Nx would reject or compute
wrongly: shapes that do not fit where tensors meet, math that can give an
infinity or a NaN, types, literals and options Nx rejects, and misuse of
traced code, gradients, random keys, containers and servings.
[docs/checks.md](docs/checks.md) lists every finding.

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
mix argus                       # Argus's configured analyses and the default ones here
mix argus --all                 # every analysis, Argus's and these
mix argus tensor_emlx           # the EMLX analysis alone
mix argus tensor_shapes ets     # the tensor shape analysis and Argus's `ets`
mix argus --list                # what's available
```

The task takes `mix argus`'s command line: `--format`, `--fail-above` and
`--color` apply to every finding. Without the alias, run
`mix argus_nx_tensor_analyses` with the same arguments.

The analyses read your project's own modules, not its dependencies. Results
are kept under `_build/<env>/argus_nx_tensor_analyses` and reused until the
compiled code changes, so an edit to a comment or a doc solves nothing again.

## Choose analyses

The analyses marked ✓ run by default.

| Analysis | Finds | Default |
|---|---|:---:|
| `tensor_shapes` | shapes, types, literals and options Nx rejects; misaligned axes; math that can turn infinite or NaN; misuse of traced code, gradients, random keys, containers and servings | ✓ |
| `tensor_emlx` | calls EMLX computes differently from BinaryBackend and EXLA (f64 kept as f32, remainders of negatives, hanging integer powers, halves rounded to even), and tensors of two backends meeting | |

Use analysis names or these sets in `analyses:`:

- `:default`: the analyses marked ✓;
- `:all`: every analysis in the table.

```elixir
def project do
  [
    # ...
    argus_nx_tensor_analyses: [analyses: [:default, :tensor_emlx]]
  ]
end
```

Analyses named on the command line run instead, and `--all` runs them all.

Two more options there change what `tensor_shapes` reports:

- `unsupported_types:` the tensor types your backend lacks. A call that
  makes one is reported (`[:f64]` for a backend without f64).
- `float_types:` the float types the code may run at. A type the code
  reads from configuration is checked as each of them
  (`[:f16, :bf16, :f32]`).

[docs/checks.md](docs/checks.md) gives every finding and its severity.
[docs/how-it-works.md](docs/how-it-works.md) explains how the analyses
follow values through a program, and what they cannot see.

## Development

```
mix deps.get
mix test
```

The tests need Soufflé 2.5 on `PATH`. They compile their fixtures under
`_build/test`, run each case through Nx and through the analysis, and
check that the two agree. Their solves are cached there as well, so a run
that changes neither the rules nor the fixtures solves nothing again.
[dev/identity](dev/identity/README.md) checks that a change to the rules
that should change no finding changes no row.

## License

MIT. See [LICENSE](LICENSE).
