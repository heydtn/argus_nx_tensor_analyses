#!/bin/sh
# compile_cost.sh [PROGRAM]: Soufflé's instructions retired and user time to
# compile PROGRAM (default priv/tensor_shapes.dl) of the checkout it runs
# in, over empty facts, after the Argus files the runner includes first.
# Instructions retired barely move with load; compare those.
program="${1:-priv/tensor_shapes.dl}"
facts="$("$(dirname "$0")/data_dir.sh")/facts_empty"
# An empty file for every relation Argus's library or these rules read.
mkdir -p "$facts"
grep -rhoE '^[[:space:]]*\.input[[:space:]]+[A-Za-z_0-9]+' deps/argus_beam/priv/dl priv |
  awk '{print $2}' | sort -u | while read -r relation; do
    [ -f "$facts/$relation.facts" ] || : > "$facts/$relation.facts"
  done
# The Argus files the runner solves a program after, as it names them
# (only the paths: compiling first prints to the same output).
wrapper="$(mktemp -d)/wrapper.dl"
mix run --no-start -e 'Enum.each(ArgusNxTensorAnalyses.Solve.argus_includes(), &IO.puts("include " <> &1))' |
  sed -n 's/^include //p' | while read -r include; do printf '.include "%s"\n' "$include"; done > "$wrapper"
if [ "$(wc -l < "$wrapper")" -lt 1 ]; then
  echo "compile_cost.sh: no Argus includes from Solve.argus_includes/0" >&2
  exit 2
fi
printf '.include "%s"\n' "$PWD/$program" >> "$wrapper"
out="$(mktemp -d)"
log="$(mktemp)"
if ! /usr/bin/time -l souffle -F "$facts" -D "$out" "$wrapper" > /dev/null 2> "$log"; then
  echo "compile_cost.sh: souffle failed:" >&2
  grep -v '^ *[0-9]' "$log" | head -20 >&2
  exit 1
fi
awk '/instructions retired/ {instructions = $1} / user / {user = $3} END {printf "%.1fG instructions, %ss user\n", instructions / 1e9, user}' "$log"
