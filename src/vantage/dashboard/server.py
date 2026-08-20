"""A dashboard on the standard library: HTTP, MJPEG, and one HTML file.

Why not FastAPI
---------------
It was the specification's own suggestion, and it would work. It was declined on
the same grounds as SciPy for the assignment solver, shapely for point-in-polygon
and psutil for memory: the standard library already does this, and the
alternative brings a dependency tree - and, with React, a Node build chain - for
a single-camera local UI.

Everything the dashboard needs is already here. Queries come from the Phase 8
store, which is indexed. Live video is MJPEG, which is multipart JPEG over plain
HTTP and needs no WebSocket. The page is one self-contained file with no build
step.

That trade reverses when there are many cameras, remote access and
authentication to think about. This module is deliberately thin so that becomes
a transport change rather than a rewrite: the JSON payloads are built in
:mod:`vantage.dashboard.api` and know nothing about how they are served.

Binding, and why it is loopback by default
-------------------------------------------
This serves **live camera footage with no authentication**. Binding it to all
interfaces would put that on the network for anyone who can reach the port. The
default is 127.0.0.1, and choosing otherwise requires saying so explicitly and
produces a warning that says what it means.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from vantage.core.logging import get_logger
from vantage.dashboard.api import DashboardApi
from vantage.dashboard.live import LiveFeed

log = get_logger(__name__)

_BOUNDARY = "vantageframe"
_STATIC = Path(__file__).parent / "static"

LOOPBACK = "127.0.0.1"


class DashboardServer:
    """Serves the dashboard until stopped."""

    def __init__(
        self,
        api: DashboardApi,
        feed: LiveFeed | None = None,
        *,
        host: str = LOOPBACK,
        port: int = 8080,
    ) -> None:
        self._api = api
        self._feed = feed
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        if host not in (LOOPBACK, "localhost", "::1"):
            log.warning(
                "dashboard bound beyond loopback",
                extra={
                    "vantage_fields": {
                        "host": host,
                        "meaning": (
                            "live camera footage and stored observations are now "
                            "reachable by anything that can reach this port, with "
                            "no authentication of any kind"
                        ),
                    }
                },
            )

    @property
    def url(self) -> str:
        shown = "localhost" if self._host in ("0.0.0.0", LOOPBACK) else self._host
        return f"http://{shown}:{self._port}"

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> str:
        """Start serving on a background thread. Returns the URL."""
        handler = _make_handler(self._api, self._feed)
        # Port 0 asks the OS for a free one, which is what the tests use; the
        # bound port is read back rather than assumed.
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler)
        self._httpd.daemon_threads = True
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="vantage-dashboard", daemon=True
        )
        self._thread.start()
        log.info(
            "dashboard listening",
            extra={"vantage_fields": {"url": self.url, "host": self._host}},
        )
        return self.url

    def serve_forever(self) -> None:
        """Block here. For the standalone ``vantage dashboard`` command."""
        if self._httpd is None:
            self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None

    def __enter__(self) -> DashboardServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def _make_handler(api: DashboardApi, feed: LiveFeed | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vantage"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            # BaseHTTPRequestHandler writes to stderr by default, which would
            # interleave a line per request into the run's own log output.
            log.debug("dashboard request", extra={"vantage_fields": {"request": fmt % args}})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            try:
                if route == "/":
                    self._send_page()
                elif route == "/stream.mjpg":
                    self._send_stream()
                elif route == "/snapshot.jpg":
                    self._send_snapshot()
                elif route.startswith("/api/"):
                    self._send_json(api.handle(route[len("/api/") :], params))
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, f"no route {route}")
            except BrokenPipeError:
                # A viewer closed the tab mid-stream. Entirely normal and not
                # worth a traceback in the log.
                pass
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                log.warning(
                    "dashboard request failed",
                    exc_info=True,
                    extra={"vantage_fields": {"route": route}},
                )
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        # -- responses ----------------------------------------------------

        def _send_page(self) -> None:
            body = (_STATIC / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # The page is served from disk on every request rather than cached
            # in memory: it is read once per page load, not per frame, and
            # editing it should not need a restart.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_snapshot(self) -> None:
            jpeg = feed.latest_jpeg() if feed is not None else None
            if jpeg is None:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no live frame")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def _send_stream(self) -> None:
            if feed is None:
                self._send_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "no live feed: this dashboard was started without a running pipeline",
                )
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.send_header("Cache-Control", "no-store, no-cache")
            # A multipart stream has no length and must not be kept alive as a
            # normal response would be.
            self.send_header("Connection", "close")
            self.end_headers()

            feed.viewer_opened()
            sequence = -1
            try:
                while True:
                    sequence, jpeg = feed.wait_for_frame(sequence, timeout=5.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--"
                        + _BOUNDARY.encode()
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The normal way a stream ends: the viewer navigated away.
                pass
            finally:
                feed.viewer_closed()

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            body = json.dumps({"error": message, "status": int(status)}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
