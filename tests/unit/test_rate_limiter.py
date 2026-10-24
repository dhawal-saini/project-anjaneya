"""Tests for :mod:`core.rate_limiter` - pacing, throttling and retry behaviour.

Everything uses a fake clock, so the suite never sleeps and is deterministic.
"""

from __future__ import annotations

import random

import pytest

from core.rate_limiter import (
    ENDPOINT_HISTORICAL,
    ENDPOINT_OTHER,
    ENDPOINT_QUOTE,
    RateLimiter,
    RateLimitTimeout,
    RetryPolicy,
    TokenBucket,
    http_status_code,
    retry_call,
)


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeError(Exception):
    """A stand-in for a broker/SDK error carrying an HTTP status code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------
def test_bucket_starts_full_and_depletes() -> None:
    clock = FakeClock()
    bucket = TokenBucket(name="test", refill_per_second=2.0, clock=clock)
    assert bucket.tokens == pytest.approx(2.0)
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False  # empty
    assert bucket.wait_time() == pytest.approx(0.5)  # 1 token per 0.5s at 2/s


def test_bucket_refills_over_time_and_caps_at_capacity() -> None:
    clock = FakeClock()
    bucket = TokenBucket(name="test", refill_per_second=2.0, clock=clock)
    assert bucket.try_acquire(2.0) is True
    clock.advance(0.25)
    assert bucket.tokens == pytest.approx(0.5)
    clock.advance(100.0)
    assert bucket.tokens == pytest.approx(2.0)  # capped


def test_bucket_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="refill_per_second"):
        TokenBucket(name="bad", refill_per_second=0.0)
    bucket = TokenBucket(name="sized", refill_per_second=1.0, capacity=2.0)
    with pytest.raises(ValueError, match="capacity"):
        bucket.try_acquire(3.0)


def test_bucket_pause_blocks_acquisition() -> None:
    clock = FakeClock()
    bucket = TokenBucket(name="test", refill_per_second=10.0, clock=clock)
    bucket.pause(5.0)
    assert bucket.try_acquire() is False
    assert bucket.wait_time() == pytest.approx(5.0)
    clock.advance(5.0)
    assert bucket.try_acquire() is True
    assert bucket.stats()["paused_seconds"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
def test_limiter_applies_safety_factor_to_every_bucket() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {ENDPOINT_QUOTE: 1.0, ENDPOINT_HISTORICAL: 3.0, ENDPOINT_OTHER: 10.0},
        safety_factor=0.5,
        clock=clock,
        sleep=clock.sleep,
    )
    stats = limiter.stats()["buckets"]
    assert stats[ENDPOINT_QUOTE]["refill_per_second"] == pytest.approx(0.5)
    assert stats[ENDPOINT_HISTORICAL]["refill_per_second"] == pytest.approx(1.5)
    assert stats[ENDPOINT_OTHER]["refill_per_second"] == pytest.approx(5.0)
    assert set(limiter.bucket_names()) == {ENDPOINT_QUOTE, ENDPOINT_HISTORICAL, ENDPOINT_OTHER}


def test_limiter_from_config_reads_the_shipped_limits(config) -> None:
    limiter = RateLimiter.from_config(config.data_sources.zerodha.rate_limits)
    stats = limiter.stats()
    expected = config.data_sources.zerodha.rate_limits
    assert stats["buckets"][ENDPOINT_QUOTE]["refill_per_second"] == pytest.approx(
        expected.quote_per_second * expected.safety_factor
    )
    assert stats["safety_factor"] == pytest.approx(expected.safety_factor)


def test_limiter_rejects_out_of_range_safety_factor() -> None:
    with pytest.raises(ValueError, match="safety_factor"):
        RateLimiter({ENDPOINT_QUOTE: 1.0}, safety_factor=1.5)


def test_acquire_waits_for_a_token_and_reports_the_wait() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {ENDPOINT_QUOTE: 2.0}, safety_factor=1.0, clock=clock, sleep=clock.sleep
    )
    assert limiter.acquire(ENDPOINT_QUOTE) == pytest.approx(0.0)
    assert limiter.acquire(ENDPOINT_QUOTE) == pytest.approx(0.0)
    waited = limiter.acquire(ENDPOINT_QUOTE)  # bucket is empty now
    assert waited == pytest.approx(0.5)
    assert clock.slept == [pytest.approx(0.5)]


def test_acquire_times_out_instead_of_hammering() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {ENDPOINT_QUOTE: 0.1}, safety_factor=1.0, clock=clock, sleep=clock.sleep
    )
    limiter.acquire(ENDPOINT_QUOTE)
    with pytest.raises(RateLimitTimeout, match=ENDPOINT_QUOTE):
        limiter.acquire(ENDPOINT_QUOTE, timeout=1.0)


def test_unknown_endpoint_class_is_rejected() -> None:
    limiter = RateLimiter({ENDPOINT_QUOTE: 1.0})
    with pytest.raises(KeyError, match="Unknown endpoint class"):
        limiter.acquire("telepathy")


def test_register_throttle_pauses_the_bucket_and_counts() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {ENDPOINT_QUOTE: 1.0}, safety_factor=1.0, clock=clock, sleep=clock.sleep
    )
    limiter.register_throttle(ENDPOINT_QUOTE, seconds=3.0)
    assert limiter.try_acquire(ENDPOINT_QUOTE) is False
    assert limiter.stats()["throttle_counts"][ENDPOINT_QUOTE] == 1


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------
def test_status_code_extraction_from_common_error_shapes() -> None:
    assert http_status_code(FakeError("throttled", 429)) == 429

    class WithResponse(Exception):
        response = type("R", (), {"status_code": 503})()

    assert http_status_code(WithResponse()) == 503

    class WithStatus(Exception):
        status_code = 502

    assert http_status_code(WithStatus()) == 502
    assert http_status_code(ValueError("no status")) is None


def test_backoff_grows_exponentially_and_is_capped() -> None:
    policy = RetryPolicy(
        max_attempts=6,
        initial_backoff_seconds=1.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=4.0,
        jitter=False,
    )
    delays = [policy.backoff_for(attempt) for attempt in range(1, 6)]
    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_backoff_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(jitter=True, initial_backoff_seconds=2.0, backoff_multiplier=1.0)
    rng = random.Random(1234)
    delays = [policy.backoff_for(1, rng=rng) for _ in range(50)]
    assert all(1.0 <= delay <= 2.0 for delay in delays)
    assert len(set(delays)) > 1  # actually jittering


def test_retry_policy_only_retries_transient_errors() -> None:
    policy = RetryPolicy(retry_on_status=(429, 500, 502, 503, 504))
    assert policy.should_retry(FakeError("throttled", 429)) is True
    assert policy.should_retry(FakeError("bad request", 400)) is False
    assert policy.should_retry(FakeError("forbidden", 403)) is False
    assert policy.should_retry(ConnectionError("reset")) is True
    assert policy.should_retry(TimeoutError("timed out")) is True
    assert policy.should_retry(ValueError("logic bug")) is False


def test_retry_policy_from_config(config) -> None:
    policy = RetryPolicy.from_config(config.data_sources.zerodha.retry)
    assert policy.max_attempts == config.data_sources.zerodha.retry.max_attempts
    assert 429 in policy.retry_on_status


def test_retry_policy_validation() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="initial_backoff_seconds"):
        RetryPolicy(initial_backoff_seconds=-1.0)
    with pytest.raises(ValueError, match="backoff_multiplier"):
        RetryPolicy(backoff_multiplier=0.5)
    # Zero backoff is explicitly allowed (tests / local mock servers).
    assert RetryPolicy(initial_backoff_seconds=0.0).backoff_for(1) == 0.0


# ---------------------------------------------------------------------------
# retry_call
# ---------------------------------------------------------------------------
def test_retry_call_returns_the_first_success() -> None:
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        return "ok"

    result = retry_call(operation, policy=RetryPolicy(), sleep=lambda _s: None)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_call_retries_then_succeeds() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise FakeError("throttled", 429)
        return "recovered"

    policy = RetryPolicy(max_attempts=4, jitter=False, initial_backoff_seconds=0.5)
    result = retry_call(operation, policy=policy, sleep=slept.append)
    assert result == "recovered"
    assert len(attempts) == 3
    assert slept == [0.5, 1.0]


def test_retry_call_gives_up_and_reraises() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        raise FakeError("still throttled", 429)

    policy = RetryPolicy(max_attempts=3, jitter=False)
    with pytest.raises(FakeError):
        retry_call(operation, policy=policy, sleep=lambda _s: None)
    assert len(attempts) == 3


def test_retry_call_does_not_retry_client_errors() -> None:
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        raise FakeError("bad request", 400)

    with pytest.raises(FakeError):
        retry_call(operation, policy=RetryPolicy(max_attempts=5), sleep=lambda _s: None)
    assert len(attempts) == 1


def test_retry_call_pauses_the_bucket_on_429() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {ENDPOINT_QUOTE: 100.0}, safety_factor=1.0, clock=clock, sleep=clock.sleep
    )
    state = {"first": True}

    def operation() -> str:
        if state["first"]:
            state["first"] = False
            raise FakeError("throttled", 429)
        return "ok"

    result = retry_call(
        operation,
        policy=RetryPolicy(max_attempts=3, jitter=False),
        endpoint=ENDPOINT_QUOTE,
        limiter=limiter,
        sleep=clock.sleep,
    )
    assert result == "ok"
    assert limiter.stats()["throttle_counts"][ENDPOINT_QUOTE] == 1


def test_retry_call_invokes_on_retry_callback() -> None:
    seen: list[tuple[int, str, float]] = []
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise FakeError("throttled", 429)
        return "ok"

    retry_call(
        operation,
        policy=RetryPolicy(max_attempts=2, jitter=False, initial_backoff_seconds=1.0),
        sleep=lambda _s: None,
        on_retry=lambda attempt, exc, delay: seen.append((attempt, type(exc).__name__, delay)),
    )
    assert seen == [(1, "FakeError", pytest.approx(1.0))]
