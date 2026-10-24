"""Tests for :class:`core.auth.zerodha_client.ZerodhaClient`.

The SDK client is faked, so these tests exercise the real rate limiting, retry,
session-expiry handling and error mapping without a single network call.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.auth.zerodha_client import (
    SessionExpiredError,
    ZerodhaClient,
    ZerodhaClientError,
)
from core.rate_limiter import ENDPOINT_HISTORICAL, ENDPOINT_OTHER, ENDPOINT_QUOTE, RateLimiter, RetryPolicy


class KiteError(Exception):
    """Stand-in for a kiteconnect exception (it exposes ``.code``)."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class FakeKite:
    """A scriptable fake of ``kiteconnect.KiteConnect``."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.access_token: str | None = None
        self.expiry_hook = None
        self.script: dict[str, list] = {}

    # -- SDK surface --------------------------------------------------------
    def set_access_token(self, token: str) -> None:
        self.access_token = token

    def set_session_expiry_hook(self, method) -> None:
        self.expiry_hook = method

    def _scripted(self, name: str, default):
        queue = self.script.get(name)
        if queue:
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return default

    def profile(self):
        self.calls.append("profile")
        return self._scripted("profile", {"user_id": "AB1234", "user_name": "Test User"})

    def instruments(self, exchange=None):
        self.calls.append(f"instruments:{exchange}")
        return self._scripted("instruments", [{"instrument_token": 1, "tradingsymbol": "X"}])

    def ltp(self, *instruments):
        self.calls.append(f"ltp:{len(instruments)}")
        return self._scripted("ltp", {"NSE:NIFTY 50": {"last_price": 25000.0}})

    def quote(self, *instruments):
        self.calls.append(f"quote:{len(instruments)}")
        return self._scripted("quote", {"NSE:NIFTY 50": {"last_price": 25000.0}})

    def historical_data(self, token, from_date, to_date, interval, continuous=False, oi=False):
        self.calls.append(f"historical:{token}:{interval}")
        return self._scripted("historical", [{"date": to_date, "close": 25000.0}])


def make_client(kite: FakeKite | None = None, *, build_kite: bool = False, **overrides) -> ZerodhaClient:
    """Create a client around a fake SDK, unless ``build_kite`` is set."""
    limiter = overrides.pop(
        "rate_limiter",
        RateLimiter({ENDPOINT_QUOTE: 1000.0, ENDPOINT_HISTORICAL: 1000.0, ENDPOINT_OTHER: 1000.0}),
    )
    if not build_kite and kite is None:
        kite = FakeKite()
    return ZerodhaClient(
        "key1234567890",
        "access-abc",
        rate_limiter=limiter,
        retry_policy=overrides.pop("retry_policy", RetryPolicy(max_attempts=3, jitter=False)),
        kite=kite,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def test_client_installs_the_session_expiry_hook() -> None:
    kite = FakeKite()
    client = make_client(kite)
    assert kite.expiry_hook is not None
    assert client.session_expired is False
    kite.expiry_hook()  # the SDK calls this on a 403 TokenException
    assert client.session_expired is True


def test_client_from_config_uses_the_configured_limits(config) -> None:
    client = ZerodhaClient.from_config(
        config,
        "access-abc",
        kite=FakeKite(),
        environ={"ZERODHA_API_KEY": "key-from-env"},
    )
    assert client.api_key == "key-from-env"
    stats = client.rate_limiter_stats()
    assert set(stats["buckets"]) == {ENDPOINT_QUOTE, ENDPOINT_HISTORICAL, ENDPOINT_OTHER}
    assert client.timeout == config.data_sources.zerodha.timeouts.request_seconds


def test_client_from_config_requires_an_api_key(config) -> None:
    with pytest.raises(ZerodhaClientError, match="ZERODHA_API_KEY"):
        ZerodhaClient.from_config(config, "access-abc", kite=FakeKite(), environ={})


def test_client_uses_the_sdk_client_factory_when_not_injected(monkeypatch) -> None:
    """The real SDK constructor must receive the API key, token and timeout."""
    created: dict = {}

    class RecordingKite:
        def __init__(self, api_key, timeout=None):
            created["api_key"] = api_key
            created["timeout"] = timeout

        def set_access_token(self, token):
            created["token"] = token

        def set_session_expiry_hook(self, method):
            created["hook"] = method

    import kiteconnect

    monkeypatch.setattr(kiteconnect, "KiteConnect", RecordingKite)
    make_client(build_kite=True, timeout=3.5)
    assert created["api_key"] == "key1234567890"
    assert created["token"] == "access-abc"
    assert created["timeout"] == 3.5
    assert "hook" in created


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def test_profile_and_instruments_are_forwarded() -> None:
    kite = FakeKite()
    client = make_client(kite)

    profile = client.profile()
    assert profile["user_id"] == "AB1234"

    instruments = client.instruments("NFO")
    assert instruments[0]["instrument_token"] == 1
    assert kite.calls == ["profile", "instruments:NFO"]


def test_quotes_are_forwarded_and_validated() -> None:
    kite = FakeKite()
    client = make_client(kite)

    assert client.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"] == 25000.0
    assert client.quote(["NSE:NIFTY 50", "NFO:NIFTY26OCT25000CE"])
    assert kite.calls == ["ltp:1", "quote:2"]

    with pytest.raises(ZerodhaClientError, match="no instruments"):
        client.ltp([])
    with pytest.raises(ZerodhaClientError, match="no instruments"):
        client.quote([])


def test_historical_data_validates_the_interval() -> None:
    kite = FakeKite()
    client = make_client(kite)
    candles = client.historical_data(256265, date(2026, 9, 1), date(2026, 9, 29), "minute")
    assert candles[0]["close"] == 25000.0
    assert kite.calls == ["historical:256265:minute"]

    with pytest.raises(ZerodhaClientError, match="Invalid interval"):
        client.historical_data(256265, date(2026, 9, 1), date(2026, 9, 29), "2minute")


def test_historical_data_forwards_continuous_and_oi() -> None:
    recorded: dict = {}

    class RecordingKite(FakeKite):
        def historical_data(self, token, from_date, to_date, interval, continuous=False, oi=False):
            recorded.update(
                token=token, interval=interval, continuous=continuous, oi=oi
            )
            return []

    make_client(RecordingKite()).historical_data(
        1, date(2026, 9, 1), date(2026, 9, 2), "5minute", continuous=True, oi=True
    )
    assert recorded == {"token": 1, "interval": "5minute", "continuous": True, "oi": True}


# ---------------------------------------------------------------------------
# Rate limiting, retry and error mapping
# ---------------------------------------------------------------------------
def test_requests_pass_through_the_rate_limiter() -> None:
    class CountingLimiter(RateLimiter):
        def __init__(self) -> None:
            super().__init__({ENDPOINT_QUOTE: 1000.0, ENDPOINT_OTHER: 1000.0})
            self.acquired: list[str] = []

        def acquire(self, endpoint=ENDPOINT_OTHER, *, tokens=1.0, timeout=None):  # type: ignore[override]
            self.acquired.append(endpoint)
            return super().acquire(endpoint, tokens=tokens, timeout=timeout)

    limiter = CountingLimiter()
    client = make_client(rate_limiter=limiter)
    client.profile()
    client.ltp(["NSE:NIFTY 50"])
    assert limiter.acquired == [ENDPOINT_OTHER, ENDPOINT_QUOTE]


def test_transient_failures_are_retried_then_succeed() -> None:
    kite = FakeKite()
    kite.script["profile"] = [KiteError("too many requests", 429), KiteError("bad gateway", 502)]
    client = make_client(kite, retry_policy=RetryPolicy(max_attempts=4, jitter=False))
    client.retry_policy.initial_backoff_seconds = 0.0  # retries are real: keep them instant
    assert client.profile()["user_id"] == "AB1234"
    # Three attempts: two failures and the successful third.
    assert kite.calls == ["profile", "profile", "profile"]


def test_non_transient_errors_are_not_retried() -> None:
    kite = FakeKite()
    kite.script["profile"] = [KiteError("bad request", 400) for _ in range(5)]
    client = make_client(kite, retry_policy=RetryPolicy(max_attempts=4, jitter=False))

    with pytest.raises(ZerodhaClientError, match="bad request"):
        client.profile()
    assert kite.calls == ["profile"]  # exactly one attempt


def test_a_403_becomes_a_session_expired_error() -> None:
    """An expired token must be actionable, not a generic failure."""
    kite = FakeKite()
    kite.script["profile"] = [KiteError("Token is invalid", 403)]
    client = make_client(kite)

    with pytest.raises(SessionExpiredError, match="re-authenticate"):
        client.profile()


def test_session_expiry_hook_makes_later_failures_session_errors() -> None:
    kite = FakeKite()
    kite.script["ltp"] = [KiteError("session expired", 500) for _ in range(3)]
    client = make_client(kite)
    client.retry_policy.max_attempts = 3
    kite.expiry_hook()  # the SDK observed a 403 earlier in the session

    with pytest.raises(SessionExpiredError):
        client.ltp(["NSE:NIFTY 50"])


def test_exhausted_retries_surface_as_a_client_error() -> None:
    kite = FakeKite()
    kite.script["ltp"] = [KiteError("throttled", 429) for _ in range(5)]
    client = make_client(kite, retry_policy=RetryPolicy(max_attempts=2, jitter=False))
    client.retry_policy.initial_backoff_seconds = 0.0

    with pytest.raises(ZerodhaClientError, match="ltp"):
        client.ltp(["NSE:NIFTY 50"])


def test_retry_logs_a_warning_when_a_logger_is_supplied() -> None:
    messages: list[str] = []

    class RecordingLogger:
        def warning(self, message: str) -> None:
            messages.append(message)

        def error(self, message: str) -> None:  # pragma: no cover - not used here
            messages.append(message)

    kite = FakeKite()
    kite.script["ltp"] = [KiteError("throttled", 429)]
    client = make_client(
        kite,
        logger=RecordingLogger(),
        retry_policy=RetryPolicy(max_attempts=2, jitter=False, initial_backoff_seconds=0.0),
    )
    client.ltp(["NSE:NIFTY 50"])
    assert any("retrying" in message for message in messages)
