"""Zerodha Kite Connect authentication (spec 4.2).

Every morning a Kite access token expires and must be re-created. This module
owns that flow so the apps never deal with cookies, TOTP codes or tokens:

* :class:`ZerodhaCredentials` - read the *names* configured in
  ``config/data_sources.yaml`` out of the environment (never the YAML itself).
* :func:`login_with_totp` - automated login: ``/api/login`` then ``/api/twofa``
  with a TOTP code, returning the one-time ``request_token``.
* :func:`exchange_request_token` - the supported SDK call
  ``KiteConnect.generate_session(request_token, api_secret)`` returning the
  ``access_token``.
* :class:`TokenStore` - cache the access token (git-ignored location) with a
  computed expiry, so a restart during the session never forces a fresh login.

Security rules enforced here

* Credentials are read from the environment only.
* Secrets never appear in log messages: only ``user_id`` and masked
  fingerprints are logged.
* The token file lives under ``data/raw/.tokens/`` which is git-ignored.

Note on the login flow: the SDK does **not** implement
``/api/login`` / ``/api/twofa``, so that part is ANJANEYA's own code, isolated
in :func:`_http_post_json` to keep it mockable in tests (no test ever performs
network I/O).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pyotp

from core.constants import get_timezone

__all__ = [
    "AccessToken",
    "AuthenticationError",
    "ZerodhaAuthError",
    "ZerodhaCredentials",
    "TokenStore",
    "compute_token_expiry",
    "exchange_request_token",
    "extract_request_token",
    "login_with_totp",
    "resolve_access_token",
]

#: Kite's login endpoints (not part of the SDK).
LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"

_REQUEST_TOKEN_PATTERN = re.compile(r"request_token=([^&\s]+)")


class ZerodhaAuthError(RuntimeError):
    """Base class for authentication problems."""


class AuthenticationError(ZerodhaAuthError):
    """Raised when a valid Kite access token could not be obtained."""


def _mask(value: str | None, keep: int = 4) -> str:
    """Return a log-safe fingerprint of a secret value."""
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}...({len(value)} chars)"


@dataclass(frozen=True, slots=True)
class ZerodhaCredentials:
    """Zerodha credentials resolved from the environment.

    ``password`` and ``totp_key`` are optional: manual request tokens are fully
    supported, and keeping the account password off disk is a valid choice.
    """

    api_key: str
    api_secret: str
    user_id: str | None = None
    password: str | None = None
    totp_key: str | None = None
    redirect_url: str | None = None
    environ: Mapping[str, str] | None = None

    @classmethod
    def from_environ(
        cls,
        zerodha_config: Any,
        environ: Mapping[str, str] | None = None,
    ) -> ZerodhaCredentials:
        """Build credentials using the env-var *names* from ``data_sources.yaml``."""
        env = os.environ if environ is None else environ
        api_key = (env.get(zerodha_config.api_key_env) or "").strip()
        api_secret = (env.get(zerodha_config.api_secret_env) or "").strip()
        if not api_key or not api_secret:
            raise AuthenticationError(
                f"Missing {zerodha_config.api_key_env} / {zerodha_config.api_secret_env}. "
                "Create a .env file from .env.example and fill in the Kite Connect app "
                "credentials (see the README quickstart)."
            )
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            user_id=(env.get(zerodha_config.user_id_env) or "").strip() or None,
            password=(env.get(zerodha_config.password_env) or "").strip() or None,
            totp_key=(env.get(zerodha_config.totp_key_env) or "").strip() or None,
            redirect_url=(env.get(zerodha_config.redirect_url_env) or "").strip() or None,
            environ=env,
        )

    @property
    def can_login_with_totp(self) -> bool:
        """True when user id, password and TOTP secret are all present."""
        return bool(self.user_id and self.password and self.totp_key)

    def describe(self) -> dict[str, str]:
        """Log-safe summary (never a secret in clear text)."""
        return {
            "api_key": _mask(self.api_key),
            "api_secret": _mask(self.api_secret),
            "user_id": self.user_id or "<unset>",
            "password": _mask(self.password),
            "totp_key": _mask(self.totp_key),
            "totp_login_available": str(self.can_login_with_totp),
        }


@dataclass(slots=True)
class AccessToken:
    """A Kite access token plus the metadata needed to decide its validity."""

    access_token: str
    user_id: str | None
    api_key: str | None
    login_time: datetime
    expires_at: datetime
    public_token: str | None = None
    obtained_via: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the on-disk cache (ISO strings, no surprises)."""
        payload = asdict(self)
        payload["login_time"] = self.login_time.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, timezone_name: str = "Asia/Kolkata"
    ) -> AccessToken:
        """Deserialise from the on-disk cache."""
        tz = get_timezone(timezone_name)
        login_time = datetime.fromisoformat(str(payload["login_time"]))
        expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        if login_time.tzinfo is None:
            login_time = login_time.replace(tzinfo=tz)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=tz)
        return cls(
            access_token=str(payload["access_token"]),
            user_id=payload.get("user_id"),
            api_key=payload.get("api_key"),
            login_time=login_time,
            expires_at=expires_at,
            public_token=payload.get("public_token"),
            obtained_via=str(payload.get("obtained_via", "unknown")),
        )

    def remaining_minutes(self, now: datetime) -> float:
        """Minutes until the token expires (negative once expired)."""
        return (self.expires_at - now).total_seconds() / 60.0


def compute_token_expiry(login_time: datetime, cutoff: time = time(6, 0)) -> datetime:
    """When a Kite access token stops working.

    Zerodha invalidates access tokens the next morning. ANJANEYA models this as
    the first occurrence of ``cutoff`` (IST, ``06:00`` by default) strictly
    after the login time, so a token issued at 09:20 IST on Monday is treated as
    valid until 06:00 IST on Tuesday.

    Being conservative is deliberate: assuming a token is valid when it is not
    would abort a capture session in the middle of the day.
    """
    if login_time.tzinfo is None:
        raise ValueError("login_time must be timezone-aware")
    candidate = datetime.combine(login_time.date(), cutoff, tzinfo=login_time.tzinfo)
    if candidate <= login_time:
        candidate += timedelta(days=1)
    return candidate


def extract_request_token(text: str | None) -> str:
    """Pull ``request_token`` out of a redirect URL / Location header.

    Kite returns it as ``...?request_token=abcdef&action=login&status=success``.
    Raises :class:`AuthenticationError` when no token is present, quoting only a
    truncated form of the input so an accidental token cannot be logged.
    """
    if not text:
        raise AuthenticationError("Login response contained no redirect URL or request_token")
    match = _REQUEST_TOKEN_PATTERN.search(text)
    if not match:
        snippet = text[:120].replace("\n", " ")
        raise AuthenticationError(
            f"Could not find request_token in the login response: {snippet!r}"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------
class TokenStore:
    """Read/write the access-token cache file (git-ignored location).

    A corrupt or partially written cache is treated as "no token" rather than a
    crash: the caller then re-authenticates.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str = "Asia/Kolkata",
        validity_cutoff: time = time(6, 0),
    ) -> None:
        self.path = Path(path)
        self.timezone_name = timezone_name
        self.validity_cutoff = validity_cutoff

    def load(self) -> AccessToken | None:
        """Return the cached token, or ``None`` when absent/unreadable."""
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return AccessToken.from_dict(payload, timezone_name=self.timezone_name)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, token: AccessToken) -> Path:
        """Persist the token, tightening permissions where the OS supports it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(token.to_dict(), indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):  # POSIX only; a no-op on Windows
            os.chmod(self.path, 0o600)
        return self.path

    def clear(self) -> None:
        """Delete the cached token if present."""
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def is_usable(
        self,
        token: AccessToken | None,
        *,
        now: datetime | None = None,
        min_remaining_minutes: int = 0,
    ) -> bool:
        """True when ``token`` exists, matches the API key and is not expired.

        ``min_remaining_minutes`` guards against starting a full trading-day
        capture with a token that expires shortly (e.g. 10 minutes).
        """
        if token is None or not token.access_token:
            return False
        moment = now or datetime.now(get_timezone(self.timezone_name))
        return token.remaining_minutes(moment) > float(min_remaining_minutes)

    def load_usable(
        self,
        *,
        api_key: str | None = None,
        now: datetime | None = None,
        min_remaining_minutes: int = 0,
    ) -> AccessToken | None:
        """Load the cached token and return it only when it is still usable."""
        token = self.load()
        if not self.is_usable(token, now=now, min_remaining_minutes=min_remaining_minutes):
            return None
        if api_key and token is not None and token.api_key and token.api_key != api_key:
            # A different Kite app was configured - the cached token is useless.
            return None
        return token


# ---------------------------------------------------------------------------
# Login flow
# ---------------------------------------------------------------------------
def _http_post_json(
    url: str,
    data: Mapping[str, Any],
    *,
    session: Any,
    timeout: float,
    allow_redirects: bool = True,
) -> Any:
    """POST form data and return the ``requests`` response.

    Kept as a separate, easily-patched function: the automated login flow is the
    one piece of ANJANEYA that talks to Kite outside the official SDK, and tests
    replace *this* function (never the network) with a stub.
    """
    import requests

    client = session if session is not None else requests.Session()
    return client.post(url, data=dict(data), timeout=timeout, allow_redirects=allow_redirects)


def _response_json(response: Any) -> dict[str, Any]:
    """Parse a JSON body, raising :class:`AuthenticationError` on failure."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthenticationError(
            f"Kite login returned a non-JSON response (HTTP {getattr(response, 'status_code', '?')})"
        ) from exc
    if not isinstance(payload, dict):
        raise AuthenticationError("Kite login returned an unexpected JSON payload")
    return payload


def login_with_totp(
    credentials: ZerodhaCredentials,
    *,
    session: Any = None,
    timeout: float = 15.0,
    totp_code: str | None = None,
) -> str:
    """Perform the two-step Kite login and return the one-time ``request_token``.

    Step 1 ``POST /api/login`` -> ``request_id``
    Step 2 ``POST /api/twofa`` -> redirect whose query string carries
                                  ``request_token``

    The TOTP code is generated locally with ``pyotp`` unless ``totp_code`` is
    supplied (tests / manual recovery).

    Raises
    ------
    AuthenticationError
        For any failure, with a message that never contains the password, the
        TOTP secret or the generated TOTP code.
    """
    if not credentials.user_id or not credentials.password:
        raise AuthenticationError(
            "Automated login needs ZERODHA_USER_ID and ZERODHA_PASSWORD in the environment"
        )
    if not credentials.totp_key and not totp_code:
        raise AuthenticationError(
            "Automated login needs ZERODHA_TOTP_KEY (or an explicit totp_code)"
        )

    login_response = _http_post_json(
        LOGIN_URL,
        {"user_id": credentials.user_id, "password": credentials.password},
        session=session,
        timeout=timeout,
    )
    login_payload = _response_json(login_response)
    if login_response.status_code != 200 or login_payload.get("status") == "error":
        raise AuthenticationError(
            f"Kite login failed: {login_payload.get('message', 'unknown error')}"
        )
    request_id = (login_payload.get("data") or {}).get("request_id")
    if not request_id:
        raise AuthenticationError("Kite login succeeded but returned no request_id")

    code = totp_code or pyotp.TOTP(credentials.totp_key).now()
    twofa_response = _http_post_json(
        TWOFA_URL,
        {
            "user_id": credentials.user_id,
            "request_id": request_id,
            "twofa_value": code,
            "twofa_type": "totp",
        },
        session=session,
        timeout=timeout,
        allow_redirects=False,
    )

    # The request_token arrives either in the Location header or, when the
    # client follows redirects, in the URL of the final response.
    headers = getattr(twofa_response, "headers", None) or {}
    location = headers.get("Location") or headers.get("location")
    if location:
        return extract_request_token(location)

    body = _response_json(twofa_response)
    if body.get("status") == "error":
        raise AuthenticationError(f"Kite 2FA failed: {body.get('message', 'unknown error')}")
    nested = (body.get("data") or {}).get("request_token")
    if nested:
        return str(nested)
    return extract_request_token(getattr(twofa_response, "url", None))


def exchange_request_token(
    credentials: ZerodhaCredentials,
    request_token: str,
    *,
    timezone_name: str = "Asia/Kolkata",
    validity_cutoff: time = time(6, 0),
    obtained_via: str = "request_token",
    kite_factory: Any = None,
) -> AccessToken:
    """Exchange a ``request_token`` for an ``access_token`` via the SDK.

    ``kite_factory`` exists for dependency injection in tests; in production the
    real ``kiteconnect.KiteConnect`` class is used.
    """
    if not request_token:
        raise AuthenticationError("An empty request_token cannot be exchanged")

    if kite_factory is None:
        from kiteconnect import KiteConnect as kite_factory  # noqa: PLC0415 - keep imports light

    kite = kite_factory(credentials.api_key)
    try:
        session_data = kite.generate_session(request_token, api_secret=credentials.api_secret)
    except Exception as exc:  # noqa: BLE001 - normalised into AuthenticationError
        raise AuthenticationError(
            "Kite rejected the request_token exchange. The token is single-use and expires "
            "within a couple of minutes - generate a fresh one and retry."
        ) from exc

    if not isinstance(session_data, Mapping) or not session_data.get("access_token"):
        raise AuthenticationError("Kite session response did not contain an access_token")

    login_time = _parse_login_time(session_data.get("login_time"), timezone_name)
    return AccessToken(
        access_token=str(session_data["access_token"]),
        user_id=session_data.get("user_id") or credentials.user_id,
        api_key=credentials.api_key,
        login_time=login_time,
        expires_at=compute_token_expiry(login_time, validity_cutoff),
        public_token=session_data.get("public_token"),
        obtained_via=obtained_via,
    )


def _parse_login_time(value: Any, timezone_name: str) -> datetime:
    """Interpret the SDK's ``login_time`` field (naive IST string or datetime)."""
    tz = get_timezone(timezone_name)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace(" ", "T", 1))
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)
    return datetime.now(tz)


def resolve_access_token(
    zerodha_config: Any,
    *,
    credentials: ZerodhaCredentials | None = None,
    request_token: str | None = None,
    environ: Mapping[str, str] | None = None,
    token_store: TokenStore | None = None,
    timezone_name: str = "Asia/Kolkata",
    validity_cutoff: time = time(6, 0),
    now: datetime | None = None,
    refresh_on_startup: bool = True,
    allow_totp_login: bool = True,
    kite_factory: Any = None,
    session: Any = None,
) -> AccessToken:
    """Return a usable access token, authenticating only when necessary.

    Resolution order (first success wins):

    1. **Cached token** on disk that is still valid (and matches the API key).
    2. **Explicit ``request_token``** (``--request-token`` on the CLI, or a
       token pasted from the Kite login redirect).
    3. **Automated TOTP login** when ``allow_totp_login`` and the credentials
       include user id + password + TOTP secret.

    Raises :class:`AuthenticationError` with actionable instructions when none
    of the three paths can produce a token.
    """
    env = environ if environ is not None else os.environ
    creds = credentials or ZerodhaCredentials.from_environ(zerodha_config, env)

    store = token_store or TokenStore(
        Path(zerodha_config.token_store.path),
        timezone_name=timezone_name,
        validity_cutoff=validity_cutoff,
    )

    cached = store.load_usable(
        api_key=creds.api_key,
        now=now,
        min_remaining_minutes=zerodha_config.token_store.min_remaining_minutes,
    )
    if cached is not None:
        return cached

    if request_token:
        token = exchange_request_token(
            creds,
            request_token,
            timezone_name=timezone_name,
            validity_cutoff=validity_cutoff,
            obtained_via="cli_request_token",
            kite_factory=kite_factory,
        )
        store.save(token)
        return token

    if refresh_on_startup and allow_totp_login and creds.can_login_with_totp:
        request_token = login_with_totp(
            creds,
            session=session,
            timeout=zerodha_config.timeouts.login_seconds,
        )
        token = exchange_request_token(
            creds,
            request_token,
            timezone_name=timezone_name,
            validity_cutoff=validity_cutoff,
            obtained_via="totp_login",
            kite_factory=kite_factory,
        )
        store.save(token)
        return token

    raise AuthenticationError(
        "No valid Kite access token available. Either\n"
        "  1. put ZERODHA_USER_ID, ZERODHA_PASSWORD and ZERODHA_TOTP_KEY in .env for automated "
        "login, or\n"
        "  2. log in at https://kite.trade/connect/login?api_key=<your_api_key>, copy the "
        "request_token from the redirect URL, and pass it via --request-token,\n"
        "  3. or wait for a stored token to become valid again (tokens expire daily around 06:00 "
        "IST)."
    )
