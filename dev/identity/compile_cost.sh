#!/bin/sh
# compile_cost.sh [PROGRAM]: Soufflé's instructions retired and user time to
# compile PROGRAM (default priv/tensor_shapes.dl) of the checkout it runs
# in, over empty facts, with Argus's library included first as the runner
# does. Instructions retired barely move with load; compare those.
program="${1:-priv/tensor_shapes.dl}"
facts="$("$(dirname "$0")/data_dir.sh")/facts_empty"
# An empty file for every relation Argus's library or these rules read.
mkdir -p "$facts"
grep -rhoE '^[[:space:]]*\.input[[:space:]]+[A-Za-z_0-9]+' deps/argus_beam/priv/dl priv |
  awk '{print $2}' | sort -u | while read -r relation; do
    [ -f "$facts/$relation.facts" ] || : > "$facts/$relation.facts"
  done
wrapper="$(mktemp -d)/wrapper.dl"
printf '.include "%s"\n.include "%s"\n' "$PWD/deps/argus_beam/priv/dl/clientlib/imports.dl" "$PWD/$program" > "$wrapper"
out="$(mktemp -d)"
/usr/bin/time -l souffle -F "$facts" -D "$out" "$wrapper" 2>&1 >/dev/null |
  awk '/instructions retired/ {instructions = $1} / user / {user = $3} END {printf "%.1fG instructions, %ss user\n", instructions / 1e9, user}'
