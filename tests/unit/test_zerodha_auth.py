"""Tests for :mod:`core.auth.zerodha_auth` and the token cache.

No test performs network I/O: the SDK class and the HTTP POST helper are both
injected. The conftest network guard would fail the test if that ever slipped.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.auth import zerodha_auth as auth
from core.auth.zerodha_auth import (
    AccessToken,
    AuthenticationError,
    TokenStore,
    ZerodhaCredentials,
    compute_token_expiry,
    exchange_request_token,
    extract_request_token,
    login_with_totp,
    resolve_access_token,
)

IST = ZoneInfo("Asia/Kolkata")
LOGIN_TIME = datetime(2026, 9, 29, 9, 20, tzinfo=IST)


def make_credentials(**overrides) -> ZerodhaCredentials:
    payload = {
        "api_key": "key1234567890",
        "api_secret": "secret1234567890",
        "user_id": "AB1234",
        "password": "hunter2hunter2",
        "totp_key": "JBSWY3DPEHPK3PXP",
        "redirect_url": None,
    }
    payload.update(overrides)
    return ZerodhaCredentials(**payload)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict | None = None, status_code: int = 200, headers=None, url="") -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def test_credentials_are_read_from_the_environment(config) -> None:
    environ = {
        "ZERODHA_API_KEY": "key-abc",
        "ZERODHA_API_SECRET": "secret-xyz",
        "ZERODHA_USER_ID": "AB1234",
        "ZERODHA_PASSWORD": "pw",
        "ZERODHA_TOTP_KEY": "TOTPSECRET",
    }
    creds = ZerodhaCredentials.from_environ(config.data_sources.zerodha, environ)
    assert creds.api_key == "key-abc"
    assert creds.can_login_with_totp is True


def test_credentials_reject_missing_api_key(config) -> None:
    with pytest.raises(AuthenticationError, match="ZERODHA_API_KEY"):
        ZerodhaCredentials.from_environ(config.data_sources.zerodha, {})


def test_partial_credentials_disable_totp_login(config) -> None:
    creds = ZerodhaCredentials.from_environ(
        config.data_sources.zerodha,
        {"ZERODHA_API_KEY": "k", "ZERODHA_API_SECRET": "s", "ZERODHA_USER_ID": "AB1234"},
    )
    assert creds.can_login_with_totp is False


def test_describe_masks_secrets() -> None:
    described = make_credentials().describe()
    assert "hunter2hunter2" not in json.dumps(described)
    assert described["user_id"] == "AB1234"
    assert described["totp_login_available"] == "True"
    assert described["password"].startswith("hunt")


# ---------------------------------------------------------------------------
# Token expiry model
# ---------------------------------------------------------------------------
def test_token_expiry_is_the_next_morning_cutoff() -> None:
    expiry = compute_token_expiry(LOGIN_TIME, time(6, 0))
    assert expiry == datetime(2026, 9, 30, 6, 0, tzinfo=IST)


def test_token_expiry_after_midnight_login_stays_the_same_day() -> None:
    """A token obtained at 00:30 expires at 06:00 the same calendar day."""
    login = datetime(2026, 9, 29, 0, 30, tzinfo=IST)
    assert compute_token_expiry(login, time(6, 0)) == datetime(2026, 9, 29, 6, 0, tzinfo=IST)


def test_token_expiry_requires_an_aware_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_token_expiry(datetime(2026, 9, 29, 9, 20))  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Request-token parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,token",
    [
        ("https://kite.trade/connect/login?request_token=abc123XYZ&action=login&status=success", "abc123XYZ"),
        ("https://127.0.0.1/?request_token=tok_9&status=success", "tok_9"),
        ("request_token=plain", "plain"),
    ],
)
def test_extract_request_token(text: str, token: str) -> None:
    assert extract_request_token(text) == token


def test_extract_request_token_fails_loudly() -> None:
    with pytest.raises(AuthenticationError, match="request_token"):
        extract_request_token("https://kite.trade/?status=error")
    with pytest.raises(AuthenticationError, match="request_token"):
        extract_request_token(None)


# ---------------------------------------------------------------------------
# Login flow (stubbed HTTP)
# ---------------------------------------------------------------------------
def patch_http(monkeypatch, responses: list[FakeResponse]) -> list[tuple[str, dict]]:
    """Replace the login HTTP helper and record the calls it received."""
    calls: list[tuple[str, dict]] = []

    def fake_post(url, data, *, session, timeout, allow_redirects=True):
        calls.append((url, dict(data)))
        return responses.pop(0)

    monkeypatch.setattr(auth, "_http_post_json", fake_post)
    return calls


def test_login_with_totp_reads_the_token_from_the_redirect(monkeypatch) -> None:
    calls = patch_http(
        monkeypatch,
        [
            FakeResponse({"status": "success", "data": {"request_id": "rid-1"}}),
            FakeResponse(
                headers={
                    "Location": "https://kite.trade/connect/login?request_token=rt-123&status=success"
                },
                status_code=302,
            ),
        ],
    )
    token = login_with_totp(make_credentials(), totp_code="123456")
    assert token == "rt-123"
    assert calls[0][0] == auth.LOGIN_URL
    assert calls[0][1] == {"user_id": "AB1234", "password": "hunter2hunter2"}
    assert calls[1][0] == auth.TWOFA_URL
    assert calls[1][1]["twofa_type"] == "totp"
    assert calls[1][1]["twofa_value"] == "123456"
    assert calls[1][1]["request_id"] == "rid-1"


def test_login_with_totp_generates_the_code_when_not_supplied(monkeypatch) -> None:
    calls = patch_http(
        monkeypatch,
        [
            FakeResponse({"status": "success", "data": {"request_id": "rid-1"}}),
            FakeResponse(headers={"Location": "https://kite.trade/?request_token=rt-9"}),
        ],
    )
    assert login_with_totp(make_credentials()) == "rt-9"
    code = calls[1][1]["twofa_value"]
    assert code.isdigit() and len(code) == 6


def test_login_with_totp_reports_a_rejected_password(monkeypatch) -> None:
    patch_http(
        monkeypatch,
        [FakeResponse({"status": "error", "message": "Invalid username or password"}, status_code=403)],
    )
    with pytest.raises(AuthenticationError, match="Invalid username or password"):
        login_with_totp(make_credentials())


def test_login_with_totp_reports_a_missing_request_id(monkeypatch) -> None:
    patch_http(monkeypatch, [FakeResponse({"status": "success", "data": {}})])
    with pytest.raises(AuthenticationError, match="request_id"):
        login_with_totp(make_credentials())


def test_login_with_totp_reports_a_failed_2fa(monkeypatch) -> None:
    patch_http(
        monkeypatch,
        [
            FakeResponse({"status": "success", "data": {"request_id": "rid-1"}}),
            FakeResponse({"status": "error", "message": "TwoFA failed"}, status_code=200),
        ],
    )
    with pytest.raises(AuthenticationError, match="2FA failed"):
        login_with_totp(make_credentials(), totp_code="000000")


def test_login_with_totp_requires_credentials() -> None:
    creds = make_credentials(user_id=None)
    with pytest.raises(AuthenticationError, match="ZERODHA_USER_ID"):
        login_with_totp(creds)
    creds = make_credentials(totp_key=None)
    with pytest.raises(AuthenticationError, match="ZERODHA_TOTP_KEY"):
        login_with_totp(creds)


def test_login_errors_never_contain_secret_values() -> None:
    """Failure messages name environment variables, never their values."""
    with pytest.raises(AuthenticationError) as excinfo:
        login_with_totp(make_credentials(password=None))
    message = str(excinfo.value)
    assert "ZERODHA_PASSWORD" in message or "ZERODHA_USER_ID" in message
    assert "hunter2hunter2" not in message
    assert "JBSWY3DPEHPK3PXP" not in message


# ---------------------------------------------------------------------------
# Request-token exchange (stubbed SDK)
# ---------------------------------------------------------------------------
class FakeKite:
    """Records the exchange call and returns a canned session payload."""

    def __init__(self, api_key: str, response: dict | None = None, error: Exception | None = None):
        self.api_key = api_key
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate_session(self, request_token: str, api_secret: str) -> dict:
        self.calls.append((request_token, api_secret))
        if self.error is not None:
            raise self.error
        return self.response or {}


def test_exchange_request_token_builds_an_access_token() -> None:
    factory_state: dict = {}

    def factory(api_key: str) -> FakeKite:
        kite = FakeKite(
            api_key,
            response={
                "access_token": "access-abc",
                "user_id": "AB1234",
                "public_token": "public-xyz",
                "login_time": "2026-09-29 09:20:00",
            },
        )
        factory_state["kite"] = kite
        return kite

    token = exchange_request_token(
        make_credentials(),
        "rt-1",
        validity_cutoff=time(6, 0),
        kite_factory=factory,
    )
    assert token.access_token == "access-abc"
    assert token.public_token == "public-xyz"
    assert token.user_id == "AB1234"
    assert token.login_time == LOGIN_TIME
    assert token.expires_at == datetime(2026, 9, 30, 6, 0, tzinfo=IST)
    assert token.obtained_via == "request_token"
    assert factory_state["kite"].calls == [("rt-1", "secret1234567890")]


def test_exchange_request_token_reports_sdk_failure() -> None:
    def factory(api_key: str) -> FakeKite:
        return FakeKite(api_key, error=RuntimeError("Token is invalid or has expired"))

    with pytest.raises(AuthenticationError, match="single-use"):
        exchange_request_token(make_credentials(), "rt-bad", kite_factory=factory)


def test_exchange_request_token_rejects_an_empty_token() -> None:
    with pytest.raises(AuthenticationError, match="empty request_token"):
        exchange_request_token(make_credentials(), "")


def test_exchange_request_token_rejects_a_payload_without_token() -> None:
    def factory(api_key: str) -> FakeKite:
        return FakeKite(api_key, response={"user_id": "AB1234"})

    with pytest.raises(AuthenticationError, match="access_token"):
        exchange_request_token(make_credentials(), "rt-1", kite_factory=factory)


# ---------------------------------------------------------------------------
# TokenStore
# ---------------------------------------------------------------------------
def make_token(*, access_token: str = "access-abc", login: datetime = LOGIN_TIME) -> AccessToken:
    return AccessToken(
        access_token=access_token,
        user_id="AB1234",
        api_key="key1234567890",
        login_time=login,
        expires_at=compute_token_expiry(login, time(6, 0)),
        public_token="public-xyz",
        obtained_via="totp_login",
    )


def test_token_store_round_trip(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / ".tokens" / "zerodha_token.json")
    assert store.load() is None
    path = store.save(make_token())
    assert path.exists()

    loaded = store.load()
    assert loaded is not None
    assert loaded.access_token == "access-abc"
    assert loaded.login_time == LOGIN_TIME
    assert loaded.expires_at == datetime(2026, 9, 30, 6, 0, tzinfo=IST)
    assert loaded.obtained_via == "totp_login"

    store.clear()
    assert store.load() is None
    store.clear()  # clearing twice is not an error


def test_token_store_ignores_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "zerodha_token.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert TokenStore(path).load() is None

    path.write_text(json.dumps({"access_token": "x"}), encoding="utf-8")  # missing fields
    assert TokenStore(path).load() is None


def test_token_usability_uses_the_expiry_and_minimum_remaining(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "token.json")
    token = make_token()
    # 21:00 IST on the login day: still valid, ~9 hours left.
    assert store.is_usable(token, now=datetime(2026, 9, 29, 21, 0, tzinfo=IST)) is True
    # 05:00 IST next morning: 1 hour left, so a 30-minute floor still passes.
    assert (
        store.is_usable(
            token, now=datetime(2026, 9, 30, 5, 0, tzinfo=IST), min_remaining_minutes=30
        )
        is True
    )
    # 05:45 IST: only 15 minutes left, below the 30-minute floor.
    assert (
        store.is_usable(
            token, now=datetime(2026, 9, 30, 5, 45, tzinfo=IST), min_remaining_minutes=30
        )
        is False
    )
    # 06:01 IST: expired.
    assert store.is_usable(token, now=datetime(2026, 9, 30, 6, 1, tzinfo=IST)) is False
    assert store.is_usable(None, now=LOGIN_TIME) is False


def test_load_usable_rejects_a_token_from_a_different_api_key(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "token.json")
    store.save(make_token())
    assert store.load_usable(api_key="key1234567890", now=LOGIN_TIME) is not None
    assert store.load_usable(api_key="a-different-key", now=LOGIN_TIME) is None


# ---------------------------------------------------------------------------
# resolve_access_token - the three resolution paths
# ---------------------------------------------------------------------------
def test_resolve_returns_the_cached_token_without_any_login(config, tmp_path, monkeypatch) -> None:
    store = TokenStore(tmp_path / "token.json")
    store.save(make_token())
    environ = {"ZERODHA_API_KEY": "key1234567890", "ZERODHA_API_SECRET": "secret1234567890"}

    def explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("resolve_access_token performed an unnecessary login")

    monkeypatch.setattr(auth, "login_with_totp", explode)
    monkeypatch.setattr(auth, "exchange_request_token", explode)

    token = resolve_access_token(
        config.data_sources.zerodha,
        environ=environ,
        token_store=store,
        now=LOGIN_TIME + timedelta(hours=1),
        timezone_name="Asia/Kolkata",
    )
    assert token.access_token == "access-abc"


def test_resolve_exchanges_an_explicit_request_token(config, tmp_path) -> None:
    store = TokenStore(tmp_path / "token.json")
    captured: dict = {}

    def factory(api_key: str) -> FakeKite:
        kite = FakeKite(
            api_key,
            response={"access_token": "fresh-access", "login_time": "2026-09-29 09:20:00"},
        )
        captured["kite"] = kite
        return kite

    token = resolve_access_token(
        config.data_sources.zerodha,
        credentials=make_credentials(),
        request_token="rt-manual",
        environ={},
        token_store=store,
        now=LOGIN_TIME,
        kite_factory=factory,
        timezone_name="Asia/Kolkata",
    )
    assert token.access_token == "fresh-access"
    assert token.obtained_via == "cli_request_token"
    # The token was cached for subsequent runs.
    assert store.load() is not None
    assert captured["kite"].calls == [("rt-manual", "secret1234567890")]


def test_resolve_performs_a_totp_login_when_credentials_allow_it(
    config, tmp_path, monkeypatch
) -> None:
    store = TokenStore(tmp_path / "token.json")
    monkeypatch.setattr(auth, "login_with_totp", lambda credentials, **kwargs: "rt-from-totp")

    def factory(api_key: str) -> FakeKite:
        return FakeKite(
            api_key, response={"access_token": "totp-access", "login_time": "2026-09-29 09:20:00"}
        )

    token = resolve_access_token(
        config.data_sources.zerodha,
        credentials=make_credentials(),
        environ={},
        token_store=store,
        now=LOGIN_TIME,
        kite_factory=factory,
        timezone_name="Asia/Kolkata",
    )
    assert token.access_token == "totp-access"
    assert token.obtained_via == "totp_login"


def test_resolve_respects_refresh_on_startup_false(config, tmp_path, monkeypatch) -> None:
    """A read-only run must not create a fresh session behind the operator's back."""
    store = TokenStore(tmp_path / "token.json")

    def explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("TOTP login must be skipped when refresh_on_startup is False")

    monkeypatch.setattr(auth, "login_with_totp", explode)
    with pytest.raises(AuthenticationError, match="No valid Kite access token"):
        resolve_access_token(
            config.data_sources.zerodha,
            credentials=make_credentials(),
            environ={},
            token_store=store,
            now=LOGIN_TIME,
            refresh_on_startup=False,
        )


def test_resolve_explains_all_the_options_when_it_cannot_authenticate(config, tmp_path) -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        resolve_access_token(
            config.data_sources.zerodha,
            credentials=make_credentials(user_id=None, password=None, totp_key=None),
            environ={},
            token_store=TokenStore(tmp_path / "token.json"),
            now=LOGIN_TIME,
        )
    message = str(excinfo.value)
    assert "ZERODHA_USER_ID" in message
    assert "--request-token" in message
