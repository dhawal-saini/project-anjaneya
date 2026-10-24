#!/usr/bin/env bash
# Start an ANJANEYA-CAPTURE live session (POSIX shell wrapper).
#
#   ./scripts/start_capture.sh --smoke
#   ./scripts/start_capture.sh --until 15:30
#   SKIP_LOGIN=1 ./scripts/start_capture.sh --dry-run
#
# The environment can select the interpreter:
#   ANJANEYA_PYTHON=/path/to/python   (default: .venv/bin/python, then `uv run python`)
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

if [[ -z "${SKIP_LOGIN:-}" ]]; then
  echo "==> ANJANEYA daily login"
  "$exe" "${pre[@]}" scripts/daily_login.py || \
    echo "warning: login exited with $?; capture will report any auth failure" >&2
fi

echo "==> ANJANEYA-CAPTURE session (mode=capture)"
exec "$exe" "${pre[@]}" -m apps.anjaneeya_capture.run --mode capture "$@"
