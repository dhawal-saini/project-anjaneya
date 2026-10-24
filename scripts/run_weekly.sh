#!/usr/bin/env bash
# Run the ANJANEYA weekly cadence pipeline (POSIX shell wrapper).
#
# The daily chain plus the cadence steps that read the artefacts it leaves behind:
#   candles -> ... -> backtest -> report -> explain -> importance -> charts -> accumulation
#
#   ./scripts/run_weekly.sh
#   ./scripts/run_weekly.sh --dry-run
#   ./scripts/run_weekly.sh --date 2026-10-05 --force
#   ./scripts/run_weekly.sh --skip charts,explain
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

echo "==> ANJANEYA weekly pipeline (candles -> backtest -> report -> accumulation)"
exec "$exe" "${pre[@]}" scripts/run_daily_pipeline.py --weekly "$@"