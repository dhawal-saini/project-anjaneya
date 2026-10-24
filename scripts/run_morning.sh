#!/usr/bin/env bash
# ANJANEYA morning routine (bash wrapper).
#
# The one command for the start of a trading day: guard the calendar, verify the
# option universe resolves, mint a fresh Kite token and start the capture session.
# Extra arguments are forwarded to the Python entry point, e.g.:
#
#     ./scripts/run_morning.sh
#     ./scripts/run_morning.sh --refresh-instruments
#     ./scripts/run_morning.sh --dry-run
#     ./scripts/run_morning.sh -- --smoke --until 15:30
#
# Exit codes: 0 ok (a clean NO_SESSION on a holiday is a 0), 1 login or capture
# failed, 2 configuration, 4 the universe resolved to zero subscriptions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${ANJANEYA_PYTHON:-}" ]]; then
  exe="${ANJANEYA_PYTHON}"
  pre=()
elif [[ -x ".venv/bin/python" ]]; then
  exe=".venv/bin/python"
  pre=()
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  exe=".venv/Scripts/python.exe"
  pre=()
elif command -v uv >/dev/null 2>&1; then
  exe="uv"
  pre=(run python)
else
  exe="python3"
  pre=()
fi

echo "==> ANJANEYA morning routine (calendar -> universe -> login -> capture)"
exec "$exe" "${pre[@]}" scripts/run_morning.py "$@"
