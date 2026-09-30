#!/bin/sh
# compare.sh BEFORE AFTER: every relation file of two identity snapshots
# that differs, with how many rows only one side has. Prints nothing and
# exits 0 when they are the same; exits 1 when they differ, and 2 when a
# snapshot is missing.
export LC_ALL=C
root="$("$(dirname "$0")/data_dir.sh")/snapshots"
for name in "$1" "$2"; do
  if [ -z "$name" ] || [ ! -d "$root/$name" ]; then
    echo "no snapshot named '$name' in $root" >&2
    exit 2
  fi
done
differences="$(
  for file in $(cd "$root/$1" && find . -name '*.tsv') $(cd "$root/$2" && find . -name '*.tsv'); do
    echo "$file"
  done | sort -u | while read -r file; do
    before="$root/$1/$file"
    after="$root/$2/$file"
    if [ ! -f "$before" ] || [ ! -f "$after" ]; then
      echo "only one side: $file"
    elif ! cmp -s "$before" "$after"; then
      echo "differs: $file (-$(comm -23 "$before" "$after" | wc -l | tr -d ' ') +$(comm -13 "$before" "$after" | wc -l | tr -d ' '))"
    fi
  done
)"
[ -z "$differences" ] && exit 0
echo "$differences"
exit 1
