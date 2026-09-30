# Identity checks

A change to the rules that should change no finding must change no row.
These scripts check that over whole programs, not just the test suite's
assertions. They snapshot every output relation plus the internal
lattice, demand and detection relations, then compare two snapshots.

Run them from the checkout under test. The data lives in the main
checkout's `_build/identity`, which its worktrees share
(`data_dir.sh` prints it; `ARGUS_NX_IDENTITY_DIR` overrides it). None of
it is committed; all of it can be made again.

## Filling the data directory

- **The test fixtures:** run the suite with the fixtures kept:
  `ARGUS_NX_FIXTURE_BEAMS="$(dev/identity/data_dir.sh)/beams" mix test`.
  `identity.exs` solves them with the options their tests use.
- **A project's own code:** copy its compiled beams
  (`_build/dev/lib/<app>/ebin/*.beam`) into `beams/<name>/`. Keep that
  copy frozen for as long as you compare against it.

Beams compiled again differ from the ones they replace (a `defn` keeps
its source file's path), so after refilling `beams/`, take a new
baseline before comparing.

## Checking a change

1. At the commit before the change: `mix run dev/identity/identity.exs baseline`.
2. After it: `mix run dev/identity/identity.exs after`.
3. `dev/identity/compare.sh baseline after` prints each relation that
   differs, with the rows only one side has. It prints nothing when the
   two snapshots are the same.

If the change renames or merges one of the internal relations, keep it
comparable with a view in `_build/probe/identity_views.dl`. That file is
untracked, and `identity.exs` includes it after the rules.

For a change to how findings are built or worded, compare the findings
themselves: `mix run dev/identity/findings.exs NAME` keeps every finding
both analyses place, with every field, and
`dev/identity/compare_findings.sh BEFORE AFTER` compares two runs, with
the same exit codes as `compare.sh`.

`dev/identity/compile_cost.sh [program]` measures what Soufflé takes to
compile the rules over empty facts, in instructions retired, which barely
move with load.
