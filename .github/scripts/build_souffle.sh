#!/usr/bin/env bash
# Builds Souffle 2.5 from its tag into the directory given, for a runner
# the release has no package for, with Souffle's default options: numbers
# of 32 bits, where the release's packages and Homebrew's build take 64.
set -euo pipefail

prefix="$1"
tag=2.5
commit=5682a9f12e2668ecdd26348fe63cc508bc0fcf47
checkout="$RUNNER_TEMP/souffle-$tag"

case "$(uname -s)" in
  Linux)
    sudo apt-get update -qq
    sudo apt-get install -y -qq bison flex libffi-dev libncurses-dev libsqlite3-dev zlib1g-dev
    ;;
  Darwin)
    # macOS's own bison is older than the 3.2 Souffle needs.
    brew install bison
    PATH="$(brew --prefix bison)/bin:$PATH"
    ;;
esac

git clone --quiet --depth 1 --branch "$tag" https://github.com/souffle-lang/souffle.git "$checkout"

if [ "$(git -C "$checkout" rev-parse HEAD)" != "$commit" ]; then
  echo "Souffle's $tag tag is not at $commit" >&2
  exit 1
fi

cmake -S "$checkout" -B "$checkout/build" -DSOUFFLE_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build "$checkout/build" --parallel "$(getconf _NPROCESSORS_ONLN)"
cmake --install "$checkout/build"
