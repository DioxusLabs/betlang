#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
total=0

while IFS= read -r -d '' file; do
  bytes=$(wc -c < "$file")
  printf '%8d  %s\n' "$bytes" "$file"
  total=$((total + bytes))
done < <(find "$root" -type f -name '*.rs' -print0)

printf 'total=%d\n' "$total"
