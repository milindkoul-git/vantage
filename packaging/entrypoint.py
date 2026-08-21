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
==============================================================
  VANTAGE - starting up.

  There is NO video window. This app's interface is a web page,
  and it opens by itself once the camera is live:

      http://localhost:{port}

  First launch downloads about 40 MB of models, which takes a
  minute. The log below is that happening, not an error.

  Close this window, or press Ctrl+C, to stop.
==============================================================
"""

READY = """
==============================================================
  READY - open this if a browser did not:

      http://localhost:{port}

  Watching the camera. Close this window to stop.
==============================================================
"""

NO_BROWSER = """
==============================================================
  The dashboard is running, but no browser could be opened.
  Paste this into one:

      http://localhost:{port}
==============================================================
"""

STALLED = r"""
==============================================================
  The dashboard has not answered after {seconds:.0f}s.

  Most likely: another program is using the webcam, or the
  first-run model download is still going. The log above says
  which. To use a different camera or a video file instead:

      vantage.exe run --source webcam:1 --track --pose --dashboard
      vantage.exe run --source C:\path\to\clip.mp4 --track --pose
==============================================================
"""


def _say(text: str) -> None:
    """Print so it actually arrives.

    ``print`` alone is block-buffered whenever stdout is not a terminal - piped
    to a file, captured by a launcher, read by another process - so the startup
    banner sat unflushed in a 8 KB buffer and was lost entirely if the app was
    killed rather than closed. Meanwhile the logs go to stderr, which is not
    buffered that way, so the console filled with INFO lines and the one message
    naming the URL never appeared at all. That is the whole difference between
    "it isn't doing anything" and "it is working, look here".
    """
    print(text, flush=True)


def _open_browser_when_ready(port: int, timeout_s: float = 180.0) -> None:
    """Open a browser once the server answers, and say so either way.

    Opening immediately shows a connection error while the models load, so this
    polls the port and opens only when something is listening.

    The timeout is three minutes rather than thirty seconds because the thing it
    is waiting for, on a first run, is a 40 MB download over whatever connection
    the machine has. At thirty seconds the browser silently never opened and the
    app looked dead precisely when it was busiest.

    Every outcome prints something. A silent helper thread is how a working
    program comes to look like a broken one.
    """
    import socket

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    else:
        _say(STALLED.format(seconds=timeout_s))
        return

    # The server is up; the page exists whether or not a browser can be found.
    # Announce it first, so the URL is on screen even if the next call fails.
    _say(READY.format(port=port))
    try:
        if not webbrowser.open(f"http://localhost:{port}"):
            _say(NO_BROWSER.format(port=port))
    except Exception:
        # webbrowser raises on machines with no registered handler - a server
        # install, a stripped Windows image. Not a reason to look broken.
        _say(NO_BROWSER.format(port=port))


def main() -> int:
    # PyInstaller re-executes the bundle for each child process; without this a
    # program that uses multiprocessing forks itself endlessly on Windows.
    multiprocessing.freeze_support()

    from vantage.cli import main as cli_main

    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])

    port = int(os.environ.get("VANTAGE_PORT", "8080"))
    _say(BANNER.format(port=port))
    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()

    args = [*DEFAULT_ARGS, "--dashboard-port", str(port)]
    try:
        return cli_main(args)
    except KeyboardInterrupt:
        print("\n  Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
