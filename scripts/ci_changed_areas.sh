#!/usr/bin/env bash
# Write frontend=true/false to $GITHUB_OUTPUT: whether a change touches the frontend sources, the
# quantem.gpu ref they bundle, or their build. EVENT, BASE and HEAD come from widget-ci. Manual
# runs, a change to the workflow or to this script, and a push whose base is unknown (a new branch,
# a force push) count as a frontend change.
set -euo pipefail
out=${GITHUB_OUTPUT:-/dev/stdout}
case "${EVENT:-}" in
  pull_request) range="$BASE...$HEAD" ;;
  push) range="$BASE..$HEAD" ;;
  *) echo "frontend=true" >> "$out"; exit 0 ;;
esac
if [[ -z "${BASE:-}" || "$BASE" =~ ^0+$ ]] || ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  echo "frontend=true" >> "$out"; exit 0
fi
files=$(git diff --name-only "$range")
printf '%s\n' "$files"
frontend='^(js/|package(-lock)?\.json$|tsconfig[^/]*\.json$|vite[^/]*$|vitest[^/]*$|gpu-ref\.txt$|scripts/(build|sync)[^/]*$|scripts/ci_changed_areas\.sh$|\.github/workflows/widget-ci\.yml$)'
if grep -qE "$frontend" <<< "$files"; then echo "frontend=true"; else echo "frontend=false"; fi >> "$out"
