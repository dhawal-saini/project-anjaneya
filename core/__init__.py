"""ANJANEYA shared core library.

This package contains **pure logic only** — indicators, engines, schemas,
calendar maths, validation and ML/backtest helpers — shared by the three
ANJANEYA applications:

* ``apps.anjaneeya_capture`` — fetch/store raw market data.
* ``apps.anjaneeya_engine``  — ticks -> candles -> features.
* ``apps.anjaneeya_ai``      — datasets -> model -> backtest -> reports.

Design rules enforced by the project (master spec sections 1 and 30):

1. Importing anything in ``core`` must have **no I/O side effects** — no
   network calls, no database connections, no file writes at import time.
2. No look-ahead bias: at timestamp ``T`` only information available at or
   before ``T`` may be used.
3. Every parameter comes from YAML configuration, never from a literal in
   code.
4. All timestamps are timezone-aware (Asia/Kolkata by default).
"""

from __future__ import annotations

__all__ = ["__version__", "__project_name__"]

__project_name__ = "ANJANEYA"
__version__ = "0.1.0"
