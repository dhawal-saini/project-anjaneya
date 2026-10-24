"""Rate-limited, retrying wrapper around the Zerodha Kite Connect REST client.

The apps never talk to ``kiteconnect`` directly: this client is the single door
to the broker API, which is what makes the rate limits, retries, timeouts and
error mapping impossible to bypass by accident.

Design notes
------------
* Every call goes through :func:`core.rate_limiter.retry_call`, so each request
  first takes a token from the endpoint's bucket, and a 429 backs the whole
  bucket off (not just the failing call).
* The underlying SDK client is injectable (``kite=...``), so the tests exercise
  the full wrapper with a fake object and never touch the network.
* ``set_session_expiry_hook`` is wired to the SDK's 403/TokenException path so
  an expired token surfaces as an explicit, actionable error instead of a
  confusing empty response.
* Returned data passes through unchanged (the SDK returns plain dicts/lists);
  ANJANEYA converts to its own row schemas at the storage layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

from core.constants import HISTORICAL_INTERVALS
from core.rate_limiter import (
    ENDPOINT_HISTORICAL,
    ENDPOINT_OTHER,
    ENDPOINT_QUOTE,
    RateLimiter,
    RetryPolicy,
    retry_call,
)

__all__ = ["SessionExpiredError", "ZerodhaClient", "ZerodhaClientError", "VALID_INTERVALS"]

#: Minute/hour/day interval strings accepted by the historical-candle endpoint.
#: Re-exported from :data:`core.constants.HISTORICAL_INTERVALS` so the SDK wrapper
#: and the configuration loader validate against one identical list.
VALID_INTERVALS = HISTORICAL_INTERVALS


class ZerodhaClientError(RuntimeError):
    """Base class for client-side failures raised by :class:`ZerodhaClient`."""


class SessionExpiredError(ZerodhaClientError):
    """Raised when Kite rejects the access token (HTTP 403 / TokenException).

    This is the one error the apps must *not* retry blindly: the remedy is to
    re-authenticate, so it is surfaced immediately.
    """


class ZerodhaClient:
    """A thin, well-behaved wrapper over ``kiteconnect.KiteConnect``."""

    def __init__(
        self,
        api_key: str,
        access_token: str | None = None,
        *,
        rate_limiter: RateLimiter,
        retry_policy: RetryPolicy,
        timeout: float = 7.0,
        kite: Any = None,
        logger: Any = None,
    ) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.rate_limiter = rate_limiter
        self.retry_policy = retry_policy
        self.timeout = timeout
        self.logger = logger
        self._session_expired = False
        self._kite = kite if kite is not None else self._build_kite(api_key, access_token, timeout)
        self._install_session_expiry_hook()

    # -- construction -------------------------------------------------------
    @staticmethod
    def _build_kite(api_key: str, access_token: str | None, timeout: float) -> Any:
        from kiteconnect import KiteConnect

        client = KiteConnect(api_key=api_key, timeout=timeout)
        if access_token:
            client.set_access_token(access_token)
        return client

    def _install_session_expiry_hook(self) -> None:
        hook = getattr(self._kite, "set_session_expiry_hook", None)
        if hook is None:  # fake clients used in tests need not implement it
            return

        def _on_session_expiry() -> None:
            self._session_expired = True
            if self.logger is not None:
                self.logger.error(
                    "Kite session expired (HTTP 403). Re-authenticate; the stored access token "
                    "is no longer valid."
                )

        hook(_on_session_expiry)

    @classmethod
    def from_config(
        cls,
        config: Any,
        access_token: str,
        *,
        kite: Any = None,
        logger: Any = None,
        limiter: RateLimiter | None = None,
        environ: Any = None,
    ) -> ZerodhaClient:
        """Build a client from the resolved :class:`AnjaneyaConfig`."""
        import os

        zerodha = config.data_sources.zerodha
        env = os.environ if environ is None else environ
        api_key = (env.get(zerodha.api_key_env) or "").strip()
        if not api_key:
            raise ZerodhaClientError(
                f"Missing {zerodha.api_key_env} in the environment; cannot call Kite Connect."
            )
        return cls(
            api_key,
            access_token,
            rate_limiter=limiter or RateLimiter.from_config(zerodha.rate_limits),
            retry_policy=RetryPolicy.from_config(zerodha.retry),
            timeout=zerodha.timeouts.request_seconds,
            kite=kite,
            logger=logger,
        )

    # -- properties ---------------------------------------------------------
    @property
    def session_expired(self) -> bool:
        """True once the SDK reported a 403/TokenException."""
        return self._session_expired

    @property
    def kite(self) -> Any:
        """The underlying SDK client (exposed for advanced/rare use)."""
        return self._kite

    # -- internals ----------------------------------------------------------
    def _call(self, operation: Callable[[], Any], *, description: str, endpoint: str) -> Any:
        """Run ``operation`` with rate limiting + retry + error normalisation."""

        def _on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            if self.logger is not None:
                self.logger.warning(
                    f"{description}: attempt {attempt} failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {delay:.2f}s"
                )

        try:
            return retry_call(
                operation,
                policy=self.retry_policy,
                description=description,
                endpoint=endpoint,
                limiter=self.rate_limiter,
                on_retry=_on_retry,
            )
        except SessionExpiredError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to client errors below
            if getattr(exc, "code", None) == 403 or self._session_expired:
                raise SessionExpiredError(
                    f"{description} failed: Kite rejected the access token (403). "
                    "Delete the cached token and re-authenticate."
                ) from exc
            raise ZerodhaClientError(f"{description} failed: {type(exc).__name__}: {exc}") from exc

    # -- endpoints ----------------------------------------------------------
    def profile(self) -> dict[str, Any]:
        """Authenticated user profile - the standard token sanity check."""
        return self._call(self._kite.profile, description="profile()", endpoint=ENDPOINT_OTHER)

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        """Instrument dump for an exchange (published daily by Kite)."""
        return self._call(
            lambda: self._kite.instruments(exchange),
            description=f"instruments({exchange!r})",
            endpoint=ENDPOINT_OTHER,
        )

    def ltp(self, instruments: Sequence[str]) -> dict[str, Any]:
        """Last traded price for up to a few hundred instruments."""
        symbols = list(instruments)
        if not symbols:
            raise ZerodhaClientError("ltp() called with no instruments")
        return self._call(
            lambda: self._kite.ltp(*symbols),
            description=f"ltp({len(symbols)} instruments)",
            endpoint=ENDPOINT_QUOTE,
        )

    def quote(self, instruments: Sequence[str]) -> dict[str, Any]:
        """Full quotes (last price, OHLC, OI, depth) for up to ~500 instruments."""
        symbols = list(instruments)
        if not symbols:
            raise ZerodhaClientError("quote() called with no instruments")
        return self._call(
            lambda: self._kite.quote(*symbols),
            description=f"quote({len(symbols)} instruments)",
            endpoint=ENDPOINT_QUOTE,
        )

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime | date,
        to_date: datetime | date,
        interval: str,
        *,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        """Historical OHLCV candles.

        Reminder (spec 4.3): this endpoint serves the NIFTY index, NIFTY futures
        and **active** option contracts only. Expired contracts are unavailable,
        which is why ANJANEYA accumulates its own option history.
        """
        if interval not in VALID_INTERVALS:
            raise ZerodhaClientError(
                f"Invalid interval {interval!r}; expected one of {VALID_INTERVALS}"
            )
        return self._call(
            lambda: self._kite.historical_data(
                instrument_token, from_date, to_date, interval, continuous=continuous, oi=oi
            ),
            description=f"historical_data(token={instrument_token}, {interval})",
            endpoint=ENDPOINT_HISTORICAL,
        )

    def rate_limiter_stats(self) -> dict[str, Any]:
        """Current rate-limiter state (for logs / health checks)."""
        return self.rate_limiter.stats()
