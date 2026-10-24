"""Rate limiting and retry/backoff for the Zerodha data source (spec 4.5).

Why this is not optional
------------------------
Kite Connect's REST API is rate limited (~1 quote request/second, ~3
historical-candle requests/second). A burst of 429s during the 09:15 open is
exactly when the data is most valuable, so ANJANEYA paces its own requests
*below* the documented limits and backs off exponentially when throttled.

Design notes
------------
* Token buckets are keyed by *endpoint class* (``quote``, ``historical``,
  ``other``) because the broker meters them separately.
* The clock and sleep function are injectable, so the tests neither sleep nor
  depend on wall-clock time.
* Nothing here knows about ``kiteconnect``: throttling is detected from the
  HTTP status code carried by the raised exception. (Verified on the installed
  SDK: it raises ``KiteException`` subclasses with a ``.code`` attribute and
  has no dedicated 429 class.)
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ENDPOINT_QUOTE",
    "ENDPOINT_HISTORICAL",
    "ENDPOINT_OTHER",
    "RateLimitTimeout",
    "RateLimiter",
    "RetryPolicy",
    "TokenBucket",
    "http_status_code",
    "retry_call",
]

ENDPOINT_QUOTE = "quote"
ENDPOINT_HISTORICAL = "historical"
ENDPOINT_OTHER = "other"

#: Endpoint classes understood by :class:`RateLimiter`.
ENDPOINT_CLASSES = (ENDPOINT_QUOTE, ENDPOINT_HISTORICAL, ENDPOINT_OTHER)


class RateLimitTimeout(TimeoutError):
    """Raised when a token could not be acquired within the allowed wait."""


def http_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from a third-party error.

    Handles the three shapes seen in practice:

    * ``kiteconnect`` exceptions expose ``.code``,
    * ``requests`` errors expose ``.response.status_code``,
    * anything else may expose ``.status_code``.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return None


@dataclass(slots=True)
class TokenBucket:
    """A classic token bucket with a monotonic clock.

    ``capacity`` is the burst size; ``refill_per_second`` is the sustained rate.
    The bucket starts full, so the first request never waits.
    """

    name: str
    refill_per_second: float
    capacity: float | None = None
    clock: Callable[[], float] = time.monotonic

    _tokens: float = field(init=False, default=0.0)
    _updated: float = field(init=False, default=0.0)
    _paused_until: float = field(init=False, default=0.0)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be > 0, got {self.refill_per_second}")
        if self.capacity is None:
            # The burst allowance is one second's worth of requests, never less
            # than a single request: a bucket too small to hold one token could
            # never serve a request at all.
            self.capacity = max(1.0, self.refill_per_second)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")
        self._tokens = float(self.capacity)
        self._updated = self.clock()

    # -- internals ---------------------------------------------------------
    def _refill(self) -> None:
        now = self.clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.refill_per_second)
            self._updated = now

    # -- public API --------------------------------------------------------
    @property
    def tokens(self) -> float:
        """Currently available tokens after applying elapsed refill."""
        with self._lock:
            self._refill()
            return self._tokens

    def wait_time(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` tokens are available (0.0 when ready)."""
        if tokens > float(self.capacity):
            raise ValueError(
                f"requested {tokens} tokens but bucket {self.name!r} capacity is {self.capacity}"
            )
        with self._lock:
            self._refill()
            deficit = tokens - self._tokens
            wait = 0.0 if deficit <= 0 else deficit / self.refill_per_second
            pause = self._paused_until - self.clock()
            return max(wait, pause, 0.0)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Consume ``tokens`` immediately; return False when not available."""
        if tokens > float(self.capacity):
            raise ValueError(
                f"requested {tokens} tokens but bucket {self.name!r} capacity is {self.capacity}"
            )
        with self._lock:
            if self.clock() < self._paused_until:
                return False
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def pause(self, seconds: float) -> None:
        """Block the bucket for ``seconds`` (used after a 429 response)."""
        if seconds <= 0:
            return
        with self._lock:
            self._paused_until = max(self._paused_until, self.clock() + seconds)

    def stats(self) -> dict[str, float]:
        """Current bucket state (for logging / health checks)."""
        with self._lock:
            self._refill()
            return {
                "tokens": round(self._tokens, 4),
                "capacity": float(self.capacity),
                "refill_per_second": self.refill_per_second,
                "paused_seconds": max(self._paused_until - self.clock(), 0.0),
            }


class RateLimiter:
    """Per-endpoint-class token buckets plus throttle memory.

    Parameters
    ----------
    limits:
        Map of endpoint class -> requests per second. Missing classes fall back
        to ``default_limit``.
    safety_factor:
        Multiplies every configured limit (``0.9`` keeps ANJANEYA 10% below the
        documented broker limit on purpose).
    clock / sleep:
        Injectable time source and sleep function (tests pass a fake clock).
    """

    def __init__(
        self,
        limits: Mapping[str, float],
        *,
        safety_factor: float = 1.0,
        default_limit: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        throttle_pause_seconds: float = 1.0,
    ) -> None:
        if not 0 < safety_factor <= 1:
            raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
        self.safety_factor = safety_factor
        self.clock = clock
        self.sleep = sleep
        self.throttle_pause_seconds = throttle_pause_seconds
        self._buckets: dict[str, TokenBucket] = {}
        for name in ENDPOINT_CLASSES:
            rate = float(limits.get(name, default_limit)) * safety_factor
            self._buckets[name] = TokenBucket(name=name, refill_per_second=rate, clock=clock)
        for name, rate in limits.items():
            if name not in self._buckets:
                self._buckets[name] = TokenBucket(
                    name=name, refill_per_second=float(rate) * safety_factor, clock=clock
                )
        self._throttle_counts: dict[str, int] = dict.fromkeys(self._buckets, 0)

    @classmethod
    def from_config(
        cls,
        rate_limits: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> RateLimiter:
        """Build a limiter from ``config/data_sources.yaml`` Zerodha rate limits."""
        return cls(
            {
                ENDPOINT_QUOTE: rate_limits.quote_per_second,
                ENDPOINT_HISTORICAL: rate_limits.historical_per_second,
                ENDPOINT_OTHER: rate_limits.other_per_second,
            },
            safety_factor=rate_limits.safety_factor,
            clock=clock,
            sleep=sleep,
        )

    def _bucket(self, endpoint: str) -> TokenBucket:
        try:
            return self._buckets[endpoint]
        except KeyError:
            raise KeyError(
                f"Unknown endpoint class {endpoint!r}; known: {sorted(self._buckets)}"
            ) from None

    def bucket_names(self) -> list[str]:
        """Names of every configured bucket."""
        return sorted(self._buckets)

    def acquire(
        self,
        endpoint: str = ENDPOINT_OTHER,
        *,
        tokens: float = 1.0,
        timeout: float | None = None,
    ) -> float:
        """Block until a token is available; return the seconds waited.

        Raises :class:`RateLimitTimeout` if ``timeout`` is exceeded - after that
        ANJANEYA would rather fail loudly than hammer the broker.
        """
        bucket = self._bucket(endpoint)
        waited = 0.0
        while True:
            if bucket.try_acquire(tokens):
                return waited
            delay = bucket.wait_time(tokens)
            delay = max(delay, 0.001)  # never busy-spin
            if timeout is not None and waited + delay > timeout:
                raise RateLimitTimeout(
                    f"Could not acquire {tokens} token(s) for endpoint {endpoint!r} "
                    f"within {timeout}s (bucket {bucket.stats()})"
                )
            self.sleep(delay)
            waited += delay

    def try_acquire(self, endpoint: str = ENDPOINT_OTHER, *, tokens: float = 1.0) -> bool:
        """Non-blocking acquisition attempt."""
        return self._bucket(endpoint).try_acquire(tokens)

    def register_throttle(self, endpoint: str = ENDPOINT_OTHER, *, seconds: float | None = None) -> None:
        """Record a 429/throttle response and pause the bucket."""
        bucket = self._bucket(endpoint)
        bucket.pause(self.throttle_pause_seconds if seconds is None else seconds)
        self._throttle_counts[endpoint] = self._throttle_counts.get(endpoint, 0) + 1

    def stats(self) -> dict[str, Any]:
        """Buckets plus throttle counters, for logs and health checks."""
        return {
            "safety_factor": self.safety_factor,
            "buckets": {name: bucket.stats() for name, bucket in self._buckets.items()},
            "throttle_counts": dict(self._throttle_counts),
        }


@dataclass(slots=True)
class RetryPolicy:
    """Exponential-backoff policy (mirrors ``config/data_sources.yaml``)."""

    max_attempts: int = 5
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    jitter: bool = True
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        # A zero backoff is allowed (useful in tests and for a local mock server);
        # a negative one never makes sense.
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")

    @classmethod
    def from_config(cls, retry: Any) -> RetryPolicy:
        """Build a policy from the parsed ``data_sources.yaml`` retry section."""
        return cls(
            max_attempts=retry.max_attempts,
            initial_backoff_seconds=retry.initial_backoff_seconds,
            backoff_multiplier=retry.backoff_multiplier,
            max_backoff_seconds=retry.max_backoff_seconds,
            jitter=retry.jitter,
            retry_on_status=tuple(retry.retry_on_status),
        )

    def backoff_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Backoff before retry number ``attempt`` (1-based, after the first failure)."""
        raw = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        delay = min(raw, self.max_backoff_seconds)
        if self.jitter:
            generator = rng or random
            delay = delay * (0.5 + 0.5 * generator.random())
        return delay

    def should_retry(self, exc: BaseException) -> bool:
        """True when ``exc`` looks transient (a retry-worthy status or a socket error)."""
        status = http_status_code(exc)
        if status is None:
            # Network-level failures (DNS, reset, timeout) carry no status code.
            return isinstance(exc, (ConnectionError, TimeoutError, OSError))
        return status in self.retry_on_status


def retry_call(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy,
    description: str = "request",
    endpoint: str = ENDPOINT_OTHER,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    retry_exceptions: tuple[type[BaseException], ...] | None = None,
) -> Any:
    """Call ``operation`` with rate limiting and exponential backoff.

    Behaviour
    ---------
    * Every attempt first acquires a rate-limit token (unless ``limiter`` is
      None), so retries never bypass the pacing policy.
    * A 429/throttle response additionally pauses the endpoint bucket, so a
      whole capture loop backs off rather than just this call.
    * ``on_retry`` is invoked with ``(attempt, exception, delay)`` for logging.
    * Non-transient errors (e.g. 400/403) are re-raised immediately: retrying a
      bad request wastes the rate-limit budget.

    Returns the operation's return value, or re-raises the last exception.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        if limiter is not None:
            limiter.acquire(endpoint)
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - deliberate, re-raised below
            if retry_exceptions is not None and not isinstance(exc, retry_exceptions):
                raise
            last_exc = exc
            if not policy.should_retry(exc):
                raise
            status = http_status_code(exc)
            if limiter is not None and status == 429:
                limiter.register_throttle(endpoint)
            if attempt >= policy.max_attempts:
                break
            delay = policy.backoff_for(attempt, rng=rng)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
    assert last_exc is not None  # noqa: S101 - loop always sets it before break
    raise last_exc
