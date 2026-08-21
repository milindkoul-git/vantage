"""Tests for the packaged executable's entry point.

Why this file exists
--------------------
This module had no tests, and it shipped broken in a way every other check
missed. The bundle built, ran, served the dashboard at 31 fps and answered HTTP
200 - and looked completely dead to the person who ran it, because the one line
naming the URL never reached the screen.

Nothing in the test suite could have caught that, because nothing tested what
the program *says*. These do. They are not about the pipeline, which has its own
940 tests; they are about whether a working program can be told apart from a
broken one by looking at it.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "packaging" / "entrypoint.py"


@pytest.fixture(scope="module")
def ep():
    spec = importlib.util.spec_from_file_location("vantage_entrypoint", ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dead_port() -> int:
    """A port nothing is listening on, for the stall path."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def server(delay: float = 0.0):
    """A socket bound now, accepting only after ``delay``.

    Binding immediately and calling ``listen`` late is what makes this
    deterministic. The obvious version - pick a free port, close it, re-bind it
    from a thread after a sleep - leaves the port unowned in between, so
    anything else on the machine can take it and the test then waits for a
    server that will never exist. That version failed about one run in three.

    A bound socket that has not been listened on refuses connections, so the
    delay is still a real "server is not up yet" while the port stays reserved.
    """
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])

    def start() -> None:
        time.sleep(delay)
        sock.listen(4)
        sock.settimeout(5.0)
        try:
            conn, _ = sock.accept()
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        thread.join(timeout=6.0)
        sock.close()


class TestTheStartupMessage:
    def test_say_flushes(self, ep, capsys) -> None:
        """The actual bug.

        ``print`` is block-buffered when stdout is not a terminal, so the banner
        sat in an 8 KB buffer while the logs - which go to stderr - filled the
        console. The URL never appeared, and a working app looked dead.
        """
        import inspect

        source = inspect.getsource(ep._say)
        assert "flush=True" in source

    def test_the_banner_names_the_url_and_the_missing_window(self, ep) -> None:
        text = ep.BANNER.format(port=8080)
        assert "http://localhost:8080" in text
        assert "NO video window" in text, (
            "the packaged app runs --no-display, so a person expecting a camera "
            "window has to be told there will not be one"
        )
        assert "40 MB" in text, "first launch downloads models and looks stalled"

    def test_every_message_is_ascii(self, ep) -> None:
        """The Windows console encodes cp1252 and raises on anything else."""
        for name in ("BANNER", "READY", "NO_BROWSER"):
            getattr(ep, name).format(port=8080).encode("cp1252")
        ep.STALLED.format(seconds=180.0).encode("cp1252")


class TestWaitingForTheServer:
    def test_a_slow_server_is_still_announced(self, ep, capsys, monkeypatch) -> None:
        """A first run spends a minute downloading models before it listens."""
        opened: list[str] = []
        monkeypatch.setattr(ep.webbrowser, "open", lambda url: opened.append(url) or True)

        with server(delay=0.6) as port:
            ep._open_browser_when_ready(port, timeout_s=30.0)

        out = capsys.readouterr().out
        assert "READY" in out
        assert f"http://localhost:{port}" in out
        assert opened == [f"http://localhost:{port}"]

    def test_the_url_is_printed_before_the_browser_is_tried(
        self, ep, capsys, monkeypatch
    ) -> None:
        """So a failing browser cannot take the URL down with it."""

        def explode(url: str) -> bool:
            raise RuntimeError("no registered handler")

        monkeypatch.setattr(ep.webbrowser, "open", explode)
        with server() as port:
            ep._open_browser_when_ready(port, timeout_s=30.0)

        out = capsys.readouterr().out
        assert "READY" in out
        assert "no browser could be opened" in out
        assert out.index("READY") < out.index("no browser could be opened")

    def test_a_browser_that_returns_false_is_reported(self, ep, capsys, monkeypatch) -> None:
        monkeypatch.setattr(ep.webbrowser, "open", lambda url: False)
        with server() as port:
            ep._open_browser_when_ready(port, timeout_s=30.0)
        assert "no browser could be opened" in capsys.readouterr().out

    def test_a_server_that_never_answers_says_why(self, ep, capsys) -> None:
        """Silence here is what "it isn't running" actually looked like."""
        ep._open_browser_when_ready(dead_port(), timeout_s=1.0)
        out = capsys.readouterr().out
        assert "has not answered" in out
        assert "webcam" in out, "the usual cause is the camera being in use"
        assert "--source" in out, "and the user needs the way out"

    def test_the_timeout_outlasts_a_first_run_download(self, ep) -> None:
        """Thirty seconds expired mid-download, so the browser never opened
        on exactly the run where the user most needed it to."""
        import inspect

        signature = inspect.signature(ep._open_browser_when_ready)
        assert signature.parameters["timeout_s"].default >= 120.0


class TestArgumentHandling:
    def test_arguments_bypass_the_application_mode(self, ep) -> None:
        """``vantage.exe run ...`` must stay the ordinary CLI."""
        import inspect

        source = inspect.getsource(ep.main)
        assert "len(sys.argv) > 1" in source
        assert "cli_main(sys.argv[1:])" in source

    def test_the_default_run_has_no_display_and_a_dashboard(self, ep) -> None:
        assert "--no-display" in ep.DEFAULT_ARGS
        assert "--dashboard" in ep.DEFAULT_ARGS
