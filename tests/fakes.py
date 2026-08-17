"""Scriptable test doubles.

:class:`FakeSource` implements the three :class:`~vantage.ingestion.base.FrameSource`
hooks and nothing else, which is exactly what makes it useful: if the pipeline
works against it, the pipeline genuinely depends only on the interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vantage.core.errors import SourceExhausted
from vantage.ingestion.base import FrameSource, SourceInfo, SourceKind


class FakeSource(FrameSource):
    """Returns frames, or raises, according to a scripted list.

    Each script entry is either an ``int`` (emit a frame filled with that value)
    or an ``Exception`` (raise it). An empty script means the source is exhausted.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        source_id: str = "fake",
        *,
        is_live: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(source_id=source_id, uri="fake://", **kwargs)
        self.script = list(script or [])
        self.opens = 0
        self.closes = 0
        self.open_error: Exception | None = None
        self._is_live = is_live

    def _open_impl(self) -> SourceInfo:
        self.opens += 1
        if self.open_error is not None:
            raise self.open_error
        return SourceInfo(
            source_id=self.source_id,
            kind=SourceKind.CAMERA if self._is_live else SourceKind.FILE,
            uri=self.uri,
            width=8,
            height=6,
            backend="fake",
            is_live=self._is_live,
        )

    def _read_impl(self) -> tuple[np.ndarray, float | None, dict[str, Any] | None]:
        if not self.script:
            raise SourceExhausted("script finished")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return np.full((6, 8, 3), item % 256, dtype=np.uint8), None, {"scripted": item}

    def _close_impl(self) -> None:
        self.closes += 1
