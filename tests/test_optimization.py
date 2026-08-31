"""Tests for model optimization and OpenVINO IR weight compression."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.perception.optimization import optimize_model


def test_optimize_nonexistent_model_raises() -> None:
    with pytest.raises(FileNotFoundError):
        optimize_model("nonexistent_model.onnx")


def test_optimize_existing_model(tmp_path: Path) -> None:
    # Use yolox_nano.onnx if present
    p = Path("models/yolox_nano.onnx")
    if not p.is_file():
        pytest.skip("models/yolox_nano.onnx not available")

    res = optimize_model(p, output_dir=tmp_path / "optimized")
    assert res.optimized_size_mb > 0
    assert res.size_reduction_pct >= 0
    assert Path(res.optimized_path).is_file()
