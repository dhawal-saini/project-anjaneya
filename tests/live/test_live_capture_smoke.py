"""Live (opt-in) smoke test: prove the real capture path works end to end.

Nothing here runs unless an operator asks for it *and* provides real credentials:

    # Windows PowerShell
    $env:ANJANEYA_RUN_LIVE = "1"
    uv run pytest tests/live -m live -q

Three tests, because the live path needs three different things from the world and
an operator has to be told *which* one failed:

1. ``test_the_broker_path_authenticates_and_stores_the_master`` - the access token
   resolves against ``profile()`` and the ~10 MB instrument dump is downloaded,
   parsed and written to DuckDB. **This holds at any hour**, needs nothing but a
   live token, and is therefore the test that can prove the broker integration on
   an evening, a weekend or a holiday.
2. ``test_the_universe_resolves_from_a_real_price`` - the ATM is anchored on a
   *real* NIFTY price and the configured strikes resolve to listed contracts.
   Normally also any hour, because Kite quotes the index after the close too; when
   it genuinely cannot answer, it says so instead of pretending the wiring broke.
3. ``test_a_one_minute_session_is_stored_end_to_end`` - subscribe, poll once,
   store, and read the heartbeat back; a real tick in DuckDB is demanded only
   **while the exchange is open**. Outside the session this skips, because a
   closed market is not a defect.

The split is the point. As one test, a run outside market hours reported
``skipped`` *even though steps 1 and 2 had already passed*, so an operator
verifying the broker wiring after the close got no signal at all.

It is excluded from the default suite because it needs the network, a paid data
subscription, market hours for ticks to exist, and it consumes API rate limit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from apps.anjaneeya_capture.heartbeat import read_heartbeat
from apps.anjaneeya_capture.instruments import InstrumentDump
from apps.anjaneeya_capture.session import CaptureSession, SessionError
from core.config_loader import AnjaneyaConfig, load_config
from core.env import SKIP_ENV_FILE_VAR, load_env_file
from core.storage import connect
from tests.support.capture import make_app_context

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: The opt-in switch is read *before* ``.env`` on purpose: enabling live tests must
#: be an explicit shell action, never a side effect of a credentials file.
LIVE = os.environ.get("ANJANEYA_RUN_LIVE") == "1"

#: Where the credentials may come from: ``.env`` (how a developer runs this) or the
#: environment. The probe reads a *copy* of the environment - importing this module
#: must not mutate anything, or it would silently disable ``.env`` loading for the
#: whole offline suite (which relies on ANJANEYA_SKIP_DOTENV=1, see conftest.py).
_ENV_PROBE: dict[str, str] = {
    name: value for name, value in os.environ.items() if name != SKIP_ENV_FILE_VAR
}
ENV_FILE = load_env_file(config_dir=REPO_ROOT / "config", environ=_ENV_PROBE)
CREDENTIALS = bool(_ENV_PROBE.get("ZERODHA_API_KEY"))

if not (LIVE and CREDENTIALS):  # pragma: no cover - conditional skip
    pytest.skip(
        "live capture test requires ANJANEYA_RUN_LIVE=1 and real ZERODHA_* credentials "
        f"(environment or {ENV_FILE.path})",
        allow_module_level=True,
    )


@pytest.fixture(scope="module", autouse=True)
def real_credentials() -> Iterator[None]:
    """Apply ``.env``/the environment for the live run, then put the process back.

    This is the only place that deliberately lifts ``ANJANEYA_SKIP_DOTENV`` - and
    only for this module, so the offline suite stays hermetic.
    """
    skip_flag = os.environ.pop(SKIP_ENV_FILE_VAR, None)
    applied = load_env_file(config_dir=REPO_ROOT / "config")
    try:
        yield
    finally:
        for name in applied.names:
            os.environ.pop(name, None)
        if skip_flag is not None:
            os.environ[SKIP_ENV_FILE_VAR] = skip_flag


@pytest.fixture(scope="module")
def live_config(tmp_path_factory):
    """The shipped configuration, pointed at a scratch data root and database."""
    root = tmp_path_factory.mktemp("live")
    os.environ["ANJANEYA_DATA_ROOT"] = str(root / "data")
    os.environ["ANJANEYA_DB_PATH"] = str(root / "db" / "anjaneeya.duckdb")
    config = load_config(REPO_ROOT / "config")
    config.ensure_directories()
    return config


# ---------------------------------------------------------------------------
# Shared live setup
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Connected:
    """A live session that has authenticated and stored the instrument master."""

    config: AnjaneyaConfig
    context: Any
    session: CaptureSession
    client: Any
    dump: InstrumentDump
    stored: int


@pytest.fixture(scope="module")
def connected(live_config: AnjaneyaConfig) -> Connected:
    """Authenticate against the real broker once and download the master once.

    Module-scoped on purpose: the dump is ~10 MB from a rate-limited endpoint, and
    all three tests read the same one. A failure here *is* the answer to "does the
    broker path work?" - the traceback names the cause (an expired token says so),
    and every test that needs the connection reports it.
    """
    context = make_app_context(live_config, mode="capture")  # creates and bootstraps the DB
    session = CaptureSession(context, poller_thread=False)
    client = session.client()
    dump = session.load_master(client=client, force_refresh=True)
    stored = session.persist_instruments(dump)
    return Connected(
        config=live_config, context=context, session=session, client=client, dump=dump, stored=stored
    )


def scalar(config: AnjaneyaConfig, sql: str) -> int:
    """Run a one-number query against the live store."""
    connection = connect(config.db_path, read_only=True)
    try:
        return int(connection.execute(sql).fetchone()[0])
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 1. The broker path - any hour of any day
# ---------------------------------------------------------------------------
def test_the_broker_path_authenticates_and_stores_the_master(connected: Connected) -> None:
    """A live token resolves, the real dump arrives, and it reaches DuckDB.

    Deliberately needs no market hours and no price: this is the one that can
    prove the broker integration on an evening, a weekend or a holiday.
    """
    # The token is not merely present - the broker accepts it.
    whoami = connected.client.profile()
    assert whoami.get("user_id"), "profile() did not return a user id"

    assert len(connected.dump.master) > 100, "the instrument dump looks far too small"
    assert connected.stored > 0, "the instrument dump produced no DuckDB rows"

    # What the writer reported is exactly what the table holds.
    assert scalar(connected.config, "SELECT count(*) FROM instruments") == connected.stored


# ---------------------------------------------------------------------------
# 2. The universe - needs a real price
# ---------------------------------------------------------------------------
def test_the_universe_resolves_from_a_real_price(connected: Connected) -> None:
    """The ATM is anchored on a *real* NIFTY price and the strikes are listed.

    Kite quotes the index after the close as well as during the session, so this
    normally runs at any hour. When it genuinely cannot answer - a closed market
    against a scratch database that has no ``index_history`` yet - that is an
    environment limitation rather than broken wiring, and it says so.
    """
    try:
        price, source = connected.session.reference_price(
            connected.dump.master, client=connected.client
        )
    except SessionError as exc:  # pragma: no cover - depends on the hour it is run
        pytest.skip(f"no reference price outside the session: {exc}")

    assert price > 0
    assert source, "the reference price must report where it came from"

    is_open = connected.context.calendar.is_market_open(datetime.now(connected.context.calendar.timezone))
    if is_open:
        # While the exchange trades, the price must come from the broker, never
        # from a stale dump row or a stored bar.
        assert source.startswith("rest-ltp"), f"an open market must quote live, got {source}"

    plan = connected.session.plan(client=connected.client)
    assert plan.universe is not None
    assert not plan.universe.is_empty, f"no option contracts resolved (missing={plan.universe.missing})"


# ---------------------------------------------------------------------------
# 3. The session - needs market hours
# ---------------------------------------------------------------------------
def test_a_one_minute_session_is_stored_end_to_end(connected: Connected) -> None:
    """Subscribe, poll once, store, and read the heartbeat back.

    The only part of the live proof that needs the exchange to be open. Outside
    the session it skips with the reason: a closed market is not a defect.
    """
    summary = connected.session.run(smoke_minutes=1, force_refresh=False)
    assert summary.status in {"SUCCESS", "PARTIAL", "NO_SESSION"}, summary.describe()
    if summary.status == "NO_SESSION":
        pytest.skip(f"outside the trading session: {summary.reason}")

    assert summary.poller["rows"] >= 1, "the option-chain poll returned nothing"
    assert scalar(connected.config, "SELECT count(*) FROM option_chain_snapshots") >= 1

    heartbeat = read_heartbeat(connected.config.heartbeat_path())
    assert heartbeat is not None
    assert heartbeat["status"] == "STOPPED"

    ticks = scalar(connected.config, "SELECT count(*) FROM ticks")
    is_open = connected.context.calendar.is_market_open(datetime.now(connected.context.calendar.timezone))
    if is_open:
        assert ticks > 0, "the market is open but no tick was captured"
    else:
        assert ticks == 0 or summary.stream["out_of_session"] == ticks
