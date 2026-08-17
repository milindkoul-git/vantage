"""Cross-cutting primitives shared by every subsystem.

Nothing in :mod:`vantage.core` may import from :mod:`vantage.ingestion`,
:mod:`vantage.viz` or any future perception package. The dependency arrow
points inwards only.
"""
