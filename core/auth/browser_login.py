"""Browser-assisted Kite login: catch the ``request_token`` from the redirect.

Why this exists
---------------
The documented manual login is: open the Kite login URL, finish the 2FA in the
browser, then copy ``request_token`` out of the address bar. That copy is the one
genuinely manual step of the morning routine, and a ``request_token`` lives for
roughly a minute, so a botched copy costs another full attempt.

Kite delivers the token to the **registered redirect URL** of the API app. When
that URL is a loopback address (for example ``http://127.0.0.1:8444/``) this
module can serve it for exactly one request, read the token out of the query
string and shut down - no copy, no paste, and the token never appears in a
terminal, a log file or a shell history.

Design notes
------------
* Only loopback hosts are accepted, and only over plain ``http``: a redirect URL
  that points at a public host (or at ``https://127.0.0.1``) cannot be served
  from the operator's machine, so both cases fail with an actionable message
  instead of hanging until the timeout.
* The listener binds one loopback address and answers one useful request - the
  login redirect. Anything else (a browser favicon probe, a curious manual
  visit) gets a small page and the wait continues.
* The catcher is a context manager and is always shut down, including on timeout
  and ``KeyboardInterrupt``.
* Importing this module performs no I/O: no socket is created until
  :func:`obtain_request_token` (or :meth:`RedirectCatcher.start`) is called.
* The token is *returned*, never printed; the caller decides what happens to it.
"""

from __future__ import annotations

import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from core.auth.zerodha_auth import AuthenticationError

__all__ = [
    "DEFAULT_REDIRECT_URL",
    "KITE_CONNECT_LOGIN_URL",
    "LOOPBACK_HOSTS",
    "RedirectCatcher",
    "login_url",
    "obtain_request_token",
    "parse_redirect_url",
]

#: The Kite Connect v3 browser login (this is where the *human* logs in).
KITE_CONNECT_LOGIN_URL = "https://kite.zerodha.com/connect/login"

#: Redirect URL assumed when neither ``--redirect-url`` nor ``ZERODHA_REDIRECT_URL``
#: is set. The same URL must be registered in the Kite app console (runbook 2.1).
DEFAULT_REDIRECT_URL = "http://127.0.0.1:8444/"

#: Hosts that mean "this machine" and can therefore be served locally.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_SUCCESS_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ANJANEYA - login complete</title></head>
<body style="font-family: sans-serif; margin: 4rem auto; max-width: 40rem">
<h1>Login complete</h1>
<p>The access token has been handed to ANJANEYA. You can close this tab and
return to the terminal.</p></body></html>"""

_ERROR_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ANJANEYA - login failed</title></head>
<body style="font-family: sans-serif; margin: 4rem auto; max-width: 40rem">
<h1>Kite reported an error</h1>
<p>The reason is printed in the terminal. Close this tab and try again.</p>
</body></html>"""

_WAITING_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ANJANEYA - waiting for the Kite login</title></head>
<body style="font-family: sans-serif; margin: 4rem auto; max-width: 40rem">
<h1>Nothing to catch here</h1>
<p>This address only works as the <em>redirect</em> of a Kite login. Start the
login from the terminal and let Kite send you back here.</p></body></html>"""


def login_url(api_key: str) -> str:
    """The URL the operator opens to log in (Kite Connect v3).

    Kite appends ``?request_token=...&action=login&status=success`` to the app's
    registered redirect URL once the password *and* the 2FA step have passed.
    """
    if not api_key:
        raise AuthenticationError("Cannot build the Kite login URL without an API key")
    return f"{KITE_CONNECT_LOGIN_URL}?v=3&api_key={quote(api_key)}"


def parse_redirect_url(redirect_url: str) -> tuple[str, int]:
    """Split a redirect URL into the loopback ``(host, port)`` to listen on.

    Raises :class:`AuthenticationError` when the URL cannot be served from this
    machine - which is the difference between "works in one command" and "hangs
    until the timeout".
    """
    candidate = redirect_url if "://" in redirect_url else f"http://{redirect_url}"
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "http":
        raise AuthenticationError(
            f"Redirect URL {redirect_url!r} must use plain http: a loopback https URL needs a "
            "certificate this tool does not have. Register http://127.0.0.1:<port>/ in the Kite "
            "app console, or use --paste to paste the redirect URL by hand."
        )
    if host not in LOOPBACK_HOSTS:
        raise AuthenticationError(
            f"Redirect URL {redirect_url!r} is not a loopback address, so this machine cannot "
            "serve it. Register http://127.0.0.1:<port>/ as the Redirect URL of your Kite app "
            "(developers.kite.trade), or use --paste to paste the redirect URL by hand."
        )
    if parsed.port is None:
        raise AuthenticationError(
            f"Redirect URL {redirect_url!r} has no explicit port. Kite redirects to the exact "
            "registered URL, so set one (for example http://127.0.0.1:8444/)."
        )
    return host, int(parsed.port)


class _LoginHttpServer(ThreadingHTTPServer):
    """The listening socket plus a link back to the catcher that owns it."""

    catcher: RedirectCatcher
    allow_reuse_address = True


class _RedirectHandler(BaseHTTPRequestHandler):
    """Answers the login redirect and ignores everything else."""

    server_version = "ANJANEYA-login"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server's API
        """Handle one redirect (``request_token``, ``status=error``, or noise)."""
        query = {key: value[0] for key, value in parse_qs(urlparse(self.path).query).items()}
        token = query.get("request_token")
        status = (query.get("status") or "").lower()
        if token:
            self._respond(200, _SUCCESS_PAGE)
            self.server.catcher.record({"status": status or "success", "request_token": token})
        elif status == "error" or query.get("message"):
            self._respond(400, _ERROR_PAGE)
            self.server.catcher.record(
                {"status": "error", "message": query.get("message") or "unknown error"}
            )
        else:
            self._respond(200, _WAITING_PAGE)

    def _respond(self, code: int, page: str) -> None:
        """Send a small HTML page with an explicit length (keep-alive friendly)."""
        body = page.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's API
        """Stay silent: the operator sees ANJANEYA's messages, not HTTP noise."""


class RedirectCatcher:
    """A one-shot loopback HTTP listener for Kite's login redirect.

    ``port=0`` binds a free port (what the tests use); the port actually in use is
    available as :attr:`port` once :meth:`start` has run.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = int(port)
        self.outcome: dict[str, Any] = {}
        self._done = threading.Event()
        self._server: _LoginHttpServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> RedirectCatcher:
        """Bind the port and serve in a daemon thread."""
        try:
            server = _LoginHttpServer((self.host, self.port), _RedirectHandler)
        except OSError as exc:
            raise AuthenticationError(
                f"Could not listen on {self.host}:{self.port} ({exc}). Close whatever is using "
                "that port, or pass --port with another value - and register the *same* URL in "
                "the Kite app console."
            ) from exc
        server.catcher = self
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="kite-login-redirect",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving. Safe to call twice, and safe when never started."""
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> RedirectCatcher:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- result -------------------------------------------------------------
    def record(self, outcome: dict[str, Any]) -> None:
        """Remember the redirect outcome and release :meth:`wait`."""
        self.outcome = dict(outcome)
        self._done.set()

    def wait(self, timeout: float) -> dict[str, Any]:
        """Block until a redirect arrives, or raise when ``timeout`` elapses."""
        if not self._done.wait(timeout):
            raise AuthenticationError(
                f"Timed out after {timeout:.0f}s waiting for the browser to finish the Kite login. "
                "Nothing was cached - check that the Kite app's Redirect URL is exactly "
                f"http://{self.host}:{self.port}/ and run the command again."
            )
        return dict(self.outcome)




def obtain_request_token(
    api_key: str,
    *,
    redirect_url: str = DEFAULT_REDIRECT_URL,
    timeout: float = 300.0,
    open_browser: bool = True,
    port: int | None = None,
    opener: Callable[[str], Any] | None = None,
    notify: Callable[[str], None] | None = None,
) -> str:
    """Open the Kite login URL and return the ``request_token`` it redirects with.

    Parameters
    ----------
    api_key:
        The Kite Connect API key (public - it is part of the login URL).
    redirect_url:
        The app's registered redirect URL; must be a loopback ``http`` URL.
    timeout:
        Seconds to wait for the browser. Kite only ever sends the token to the
        registered URL, so an expired wait simply means "nothing was cached".
    open_browser:
        When ``False`` the URL is only announced (via ``notify``) - useful on a
        headless machine, or when the operator wants a different browser.
    port:
        Override the port taken from ``redirect_url`` (which must then be the
        port the Kite app actually redirects to).
    opener:
        Injection seam for tests; defaults to :func:`webbrowser.open`.
    notify:
        Callback for operator-facing messages; defaults to no output.
    """
    host, url_port = parse_redirect_url(redirect_url)
    target = login_url(api_key)
    say = notify or (lambda _message: None)
    with RedirectCatcher(host, port or url_port) as catcher:
        if open_browser:
            if not (opener or webbrowser.open)(target):
                say("Could not open a browser automatically; open this URL yourself:")
                say(f"  {target}")
        else:
            say("Open this URL in a browser and complete the Kite login (password + 2FA):")
            say(f"  {target}")
        say(
            f"Waiting for the Kite redirect on http://{host}:{catcher.port}/ "
            f"(timeout {timeout:.0f}s)"
        )
        outcome = catcher.wait(timeout)
    if outcome.get("status") == "error":
        raise AuthenticationError(f"Kite reported a login error: {outcome.get('message')}")
    token = outcome.get("request_token")
    if not token:
        raise AuthenticationError("The Kite redirect did not contain a request_token")
    return str(token)
