"""Deprecated entry point for the multi-camera pipeline.

Superseded by ``vantage facility --cameras ID=URI ...``. This module remained
because it was the only way to start the facility pipeline at all, but its
defaults named four video files by bare relative path - clips that existed on
one machine and nowhere else - so running it as documented failed everywhere.

Kept as a thin forwarder so ``python -m vantage.multicam.runner`` still works
and says where the command went.
"""

from __future__ import annotations

import sys


def main() -> int:
    from vantage.cli import main as cli_main

    sys.stderr.write("vantage.multicam.runner is deprecated; use 'vantage facility' instead.\n")
    return cli_main(["facility", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
