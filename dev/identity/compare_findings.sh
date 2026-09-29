#!/bin/sh
# compare_findings.sh BEFORE AFTER: every findings file of two runs of
# findings.exs that differs, with the findings only one side has. Prints
# nothing and exits 0 when they are the same; exits 1 when they differ,
# and 2 when a run is missing.
export LC_ALL=C
root="$("$(dirname "$0")/data_dir.sh")/findings"
for name in "$1" "$2"; do
  if [ -z "$name" ] || [ ! -d "$root/$name" ]; then
    echo "no findings named '$name' in $root" >&2
    exit 2
  fi
done
status=0
files="$( (cd "$root/$1" && find . -name '*.txt'); (cd "$root/$2" && find . -name '*.txt') )"
for file in $(echo "$files" | sort -u); do
  before="$root/$1/$file"
  after="$root/$2/$file"
  if [ ! -f "$before" ] || [ ! -f "$after" ]; then
    echo "only one side: $file"
    status=1
  elif ! cmp -s "$before" "$after"; then
    echo "differs: $file (-$(comm -23 "$before" "$after" | wc -l | tr -d ' ') +$(comm -13 "$before" "$after" | wc -l | tr -d ' '))"
    comm -23 "$before" "$after" | sed 's/^/  - /'
    comm -13 "$before" "$after" | sed 's/^/  + /'
    status=1
  fi
done
exit $status
