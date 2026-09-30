# ArgusNxTensorAnalyses

[Argus](https://github.com/QuinnWilton/argus) analyses for code that uses
[Nx](https://github.com/elixir-nx/nx). They read a project's compiled
modules and report, before the code runs, calls Nx would reject or compute
wrongly: shapes that do not fit where tensors meet, math that can give an
infinity or a NaN, types, literals and options Nx rejects, and misuse of
traced code, gradients, random keys, containers and servings.
[docs/checks.md](docs/checks.md) lists every finding.

```
error[argus.nx_shapes]: Nx.dot/2 contracts axes that do not match
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
mix argus emlx                  # the EMLX analysis alone
mix argus nx_shapes ets         # the Nx shape analysis and Argus's `ets`
mix argus --list                # what's available
```

The task takes `mix argus`'s command line: `--format`, `--fail-above` and
`--color` apply to every finding. Without the alias, run
`mix argus_nx_tensor_analyses` with the same arguments.

The analyses read your project's own modules, not its dependencies. Results
are kept under `_build/<env>/argus_nx_tensor_analyses` and reused until the
compiled code changes, so an edit to a comment or a doc solves nothing again.
A run solves one program for all the analyses it runs: the `nx_` analyses
share one, and a run with `emlx` solves a larger one that finds them too.

## Choose analyses

Each analysis is a category of finding, reported under its name
(`error[argus.nx_shapes]`). The analyses marked ✓ run by default.

| Analysis | Finds | Default |
|---|---|:---:|
| `nx_shapes` | operands that do not broadcast; `dot`, `conv` and window axes that do not fit; reshapes that change the element count; axes a tensor lacks; out-of-range slices and `tensor[key]` access; tuples used as tensors | ✓ |
| `nx_names` | sizes the code names differently (`config.heads` and `config.kv_heads`); named axes meeting unnamed ones; contracted axes of different names; named axes vectorized under another name; reshapes that scramble axes | ✓ |
| `nx_math` | division by zero; logarithms of zero or negatives; square roots of negatives; `exp` overflow in an unshifted softmax or a written-out softplus or logistic; asin, acos, atanh and acosh past their domain; NaN comparisons; divisors and logarithms of unchecked inputs | ✓ |
| `nx_types` | floats where only integers go; unsigned wraparound; literals a type cannot hold; lossy casts; silent upcasts and narrowing merges; types the backend lacks | ✓ |
| `nx_options` | option keys a function does not take; options in the wrong form or with values Nx rejects | ✓ |
| `nx_indices` | indices that can go negative; slice starts Nx clamps; `ddof` at or past the count; empty or reversed random ranges | ✓ |
| `nx_traced` | tensor data read while Nx traces; Elixir operators and `if` on tensors; non-scalar `if`, `cond` and `while` predicates; branches of different shapes; `while` state that changes shape or type; `defn` arguments used as integers | ✓ |
| `nx_gradients` | NaN or infinite gradients (a root or norm at zero, the standard deviation of equal values, `atan2` at the origin, `select`-masked logarithms, degenerate decompositions); calls with no gradient; `custom_grad` mistakes; grads of tuples and maps | ✓ |
| `nx_containers` | struct fields a derived `Nx.Container` resets inside `defn`; containers holding nil, atoms or lists; grads capturing the value they differentiate; `traverse` and `reduce` visiting fields in different orders | ✓ |
| `nx_random` | keys drawn from twice; keys a loop captures or passes back; spent keys returned; two keys of one written seed; one draw repeated across a mean or standard deviation | ✓ |
| `nx_freed` | tensors read, handed on or returned after `backend_transfer`, `backend_deallocate` or donation | ✓ |
| `nx_serving` | outputs without the batch axis; operations across the batch; ahead-of-time templates whose size or type do not fit; per-request shapes that crash the serving; incompatible `Nx.Batch` entries; serving API misuse | ✓ |
| `emlx` | f64 and c128 kept as f32 and c64; remainders of negatives; integer powers of negative exponents, which hang; halves rounded to even; shifts EMLX wraps; tensors of two backends meeting, such as EMLX's and EXLA's | |

Use analysis names or these sets in `analyses:`:

- `:default`: the analyses marked ✓;
- `:all`: every analysis in the table.

```elixir
def project do
  [
    # ...
    argus_nx_tensor_analyses: [analyses: [:default, :emlx]]
  ]
end
```

Analyses named on the command line run instead, and `--all` runs them all.

Two more options there change what the `nx_` analyses report:

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
