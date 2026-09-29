#!/bin/sh
# The identity data directory: `_build/identity` of the main checkout of
# the repository this script is in, shared by its worktrees, or
# ARGUS_NX_IDENTITY_DIR.
if [ -n "$ARGUS_NX_IDENTITY_DIR" ]; then
  echo "$ARGUS_NX_IDENTITY_DIR"
else
  here="$(dirname "$(readlink -f "$0")")"
  echo "$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")/_build/identity"
fi
