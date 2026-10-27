"""Authentication helpers for the Zerodha Kite Connect data source.

Phase 1 provides the full auth surface (credentials, TOTP login, request-token
exchange, on-disk token cache). The WebSocket ticker and REST client live in
:mod:`core.auth.zerodha_client`.
"""

from __future__ import annotations

from core.auth.browser_login import (
    DEFAULT_REDIRECT_URL,
    RedirectCatcher,
    login_url,
    obtain_request_token,
    parse_redirect_url,
)
from core.auth.zerodha_auth import (
    AccessToken,
    AuthenticationError,
    TokenStore,
    ZerodhaAuthError,
    ZerodhaCredentials,
    compute_token_expiry,
    exchange_request_token,
    extract_request_token,
    login_with_totp,
    normalise_request_token,
    resolve_access_token,
)
from core.auth.zerodha_client import (
    SessionExpiredError,
    ZerodhaClient,
    ZerodhaClientError,
)

__all__ = [
    "AccessToken",
    "AuthenticationError",
    "DEFAULT_REDIRECT_URL",
    "RedirectCatcher",
    "SessionExpiredError",
    "TokenStore",
    "ZerodhaAuthError",
    "ZerodhaClient",
    "ZerodhaClientError",
    "ZerodhaCredentials",
    "compute_token_expiry",
    "exchange_request_token",
    "extract_request_token",
    "login_url",
    "login_with_totp",
    "normalise_request_token",
    "obtain_request_token",
    "parse_redirect_url",
    "resolve_access_token",
]
