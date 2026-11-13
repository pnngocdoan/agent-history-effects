#!/usr/bin/env bash
# Apply the numbered patches in tools/patches/ to the pinned benchmark checkout.
# Run from the repo root after `git submodule update --init`. Running it twice gives the same result.
set -euo pipefail
BENCH=benchmark/Agent-SafetyBench
git -C "$BENCH" checkout -q -- evaluation score
git -C "$BENCH" clean -fdq evaluation score
for p in tools/patches/*.patch; do
  git -C "$BENCH" apply "../../$p"
  echo "applied: $p"
done
