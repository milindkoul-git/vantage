"""Allows ``python -m vantage`` as well as the installed ``vantage`` script."""

from vantage.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
