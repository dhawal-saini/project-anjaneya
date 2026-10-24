"""ANJANEYA-CAPTURE: raw market data acquisition (Phase 2).

Modules
-------
``instruments``          Kite instrument dump cache + attribute-based universe
                         resolution (never hand-built symbol strings).
``nifty_index_fetcher``  REST backfill/top-up of NIFTY index OHLCV, with
                         self-planned windows and adaptive window halving.
``storage_writer``       batched DuckDB writes, the daily Parquet mirror and the
                         coverage ledger.
``heartbeat``            the JSON file that answers "is capture alive right now?"
                         (the only live view while DuckDB is held by the writer).
``websocket_listener``   ``KiteTicker`` adapter, tick -> row conversion (depth ->
                         bid/ask, machine-local -> IST timestamps) and the
                         reconnect/stall watchdog.
``option_chain_poller``  periodic REST quotes for OI/two-sided quotes, the
                         independent second source the spec asks for.
``session``              one session, start to finish: plan, subscribe, monitor,
                         poll, store, report.
``run``                  the CLI entry point (``python -m apps.anjaneeya_capture.run``).

Everything here is import-safe (no I/O, no network, no ``kiteconnect`` import at
module scope) so the whole pipeline can be tested offline with fakes.
"""

from __future__ import annotations

__all__: list[str] = []
