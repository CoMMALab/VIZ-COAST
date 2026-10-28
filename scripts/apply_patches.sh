#!/usr/bin/env bash
# Apply the VIZ-COAST patches in patches/ to the pinned submodules in external/.
# Safe to re-run: a patch that is already applied is skipped.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# patch file -> directory it applies to (relative to repo root)
targets=(
  "pybullet-planning.patch:external/pybullet-planning"
  "symbolic.patch:external/symbolic"
  "symbolic-ctrl-utils.patch:external/symbolic/external/ctrl_utils/ctrl-utils.git"
  "spatial-dyn.patch:external/spatial-dyn"
  "spatial-dyn-ctrl-utils.patch:external/spatial-dyn/external/ctrl_utils/ctrl-utils.git"
)

for entry in "${targets[@]}"; do
  patch="$root/patches/${entry%%:*}"
  dir="$root/${entry#*:}"
  if [[ ! -e "$dir/.git" ]]; then
    echo "error: $dir is not checked out; run 'git submodule update --init --recursive' first" >&2
    exit 1
  fi
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: ${entry%%:*}"
  elif git -C "$dir" apply --check "$patch"; then
    git -C "$dir" apply "$patch"
    echo "applied: ${entry%%:*}"
  else
    echo "error: ${entry%%:*} does not apply cleanly to ${entry#*:}" >&2
    exit 1
  fi
done
