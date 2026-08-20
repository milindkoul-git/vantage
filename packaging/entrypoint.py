"""Entry point for the packaged executable.

Not the same as the ``vantage`` console script, and the difference is the point.
A CLI is run by someone who typed a command and will read its output. A
double-clicked executable is run by someone who wants a window: bare
``vantage.exe`` therefore starts the pipeline with the dashboard and opens a
browser at it, while any arguments hand straight to the normal CLI.

    vantage.exe                       the application: camera + dashboard
    vantage.exe run --source file.mp4 the CLI, unchanged
    vantage.exe history events        likewise
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
import webbrowser

DEFAULT_ARGS = [
    "run",
    "--source",
    "webcam:0",
    "--track",
    "--pose",
    "--model",
    "yolox-tiny",
    "--dashboard",
    "--store",
    # No window: the dashboard is the interface here, and an OpenCV window
    # behind the browser would be two views of the same thing, one of which
    # cannot be closed except by pressing q in it.
    "--no-display",
]

BANNER = """
  Vantage - starting the camera and dashboard.

  The dashboard will open in your browser at http://localhost:{port}
  Close this window, or press Ctrl+C, to stop.
"""


def _open_browser_when_ready(port: int, timeout_s: float = 30.0) -> None:
    """Open a browser once the server answers, not before.

    Opening immediately shows a connection error while the models load, which
    takes several seconds on first run and looks like a failure. This polls the
    port and opens only when something is listening.
    """
    import socket

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                webbrowser.open(f"http://localhost:{port}")
                return
        except OSError:
            time.sleep(0.25)


def main() -> int:
    # PyInstaller re-executes the bundle for each child process; without this a
    # program that uses multiprocessing forks itself endlessly on Windows.
    multiprocessing.freeze_support()

    from vantage.cli import main as cli_main

    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])

    port = int(os.environ.get("VANTAGE_PORT", "8080"))
    print(BANNER.format(port=port))
    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()

    args = [*DEFAULT_ARGS, "--dashboard-port", str(port)]
    try:
        return cli_main(args)
    except KeyboardInterrupt:
        print("\n  Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
