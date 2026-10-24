#!/usr/bin/env bash
# Run the ANJANEYA daily post-close pipeline (POSIX shell wrapper).
#
# One command for every trading day after the 15:30 close:
#   candles -> greeks -> features -> labels -> validate -> dataset -> train -> backtest
#
#   ./scripts/run_daily_pipeline.sh
#   ./scripts/run_daily_pipeline.sh --dry-run
#   ./scripts/run_daily_pipeline.sh --date 2026-10-01 --force
#   ./scripts/run_daily_pipeline.sh --skip validate,backtest
#
# Exit codes: 0 ok, 1 a step failed, 2 configuration, 4 below the training gate.
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

echo "==> ANJANEYA daily pipeline (candles -> backtest)"
exec "$exe" "${pre[@]}" scripts/run_daily_pipeline.py "$@"
