#!/usr/bin/env bash
# Run the ANJANEYA index-history backfill (POSIX shell wrapper).
#
#   ./scripts/run_backfill.sh --dry-run
#   ./scripts/run_backfill.sh --force-refresh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${ANJANEYA_PYTHON:-}" ]]; then
  exe="${ANJANEYA_PYTHON}"
  pre=()
elif [[ -x ".venv/bin/python" ]]; then
  exe=".venv/bin/python"
  pre=()
elif command -v uv >/dev/null 2>&1; then
  exe="uv"
  pre=(run python)
else
  exe="python3"
  pre=()
fi

echo "==> ANJANEYA-CAPTURE index backfill (mode=backfill)"
exec "$exe" "${pre[@]}" -m apps.anjaneeya_capture.run --mode backfill "$@"
