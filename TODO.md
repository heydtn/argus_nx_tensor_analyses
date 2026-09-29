# To consider

## A field tested to be at least 1 is a size

`Nx.subtract(config.seq, 1)` used as an index or a slice start is
reported as possibly negative, because the sign rules treat every field
as possibly 0. Code usually rules that out somewhere: an `if`, a `cond`
or a guard testing `config.seq >= 1` (or `> 0`), or a validation that
raises otherwise. Recognize such a test and treat the field as at least
1 wherever it holds, rather than treating every field as a size.

## Integer arithmetic outside Nx, used as an index

`t[[.., pos - 1]]` and slice starts computed with plain integers can go
negative just as `Nx.subtract` can. A rule for them was built and taken
out: in practice it mostly flagged `len - 1` idioms that are never
negative. Reconsider it once tests like the one above are recognized, so
a guarded `len - 1` stays quiet.

## Which backend a project computes on

The EMLX analysis takes EMLX as the default backend unless the code calls
`Nx.default_backend/1`, because `config :nx, default_backend:` is not in
the compiled code. A project that builds some tensors on EXLA and moves
them to EMLX through configuration gets false findings, such as f64
tables that are really computed on EXLA and cast to f32 before they move.
Options: a `default_backend:` project option, reading the project's
config files, or moving the EMLX checks into a package of their own.
