#!/bin/bash
# Start the swash full fine-tune for one model: parameters.yaml defaults + models/<model>.params.
#   recipes/granite4-defense-lora/slurm/sft-swash/train.sh <model> [KEY=VALUE ...]
# <model> is a file name under models/ without .params (swash-3b, swash-2b, g51-20b). Extra
# KEY=VALUE arguments override both; add --dry-run to only render the build.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
M="${1:?usage: train.sh <model> [KEY=VALUE ...] [--dry-run]}"; shift
F="$HERE/models/$M.params"
[ -f "$F" ] || { echo "no $F; models: $(cd "$HERE/models" && ls *.params | sed 's/\.params$//' | xargs)" >&2; exit 1; }
args=()
while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  args+=(--param "$line")
done < "$F"
dry=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then dry=(--dry-run --save-build-file "/tmp/sft-swash-$M.yaml"); else args+=(--param "$a"); fi
done
cd "$HERE/../../../.."
exec env GB_ENVIRONMENT=STANDALONE PATH="$HOME/.local/bin:$PATH" .venv/bin/gb build start \
  -f "$HERE/build.yaml" --parameters-path "$HERE/parameters.yaml" --skip-version-check \
  "${args[@]}" "${dry[@]}"
