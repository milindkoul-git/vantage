"""Tests for dataset catalog, synthetic generator, and model trainer."""

from __future__ import annotations

from pathlib import Path

from vantage.training.dataset import DatasetCatalog
from vantage.training.synthetic_generator import SyntheticDataGenerator
from vantage.training.trainer import ModelTrainer


def test_synthetic_data_generator() -> None:
    gen = SyntheticDataGenerator(width=640, height=480, seed=123)
    batch = gen.generate_batch(20)
    assert len(batch) == 20
    for sample in batch:
        assert sample.width == 640
        assert sample.height == 480
        assert len(sample.boxes) > 0


def test_dataset_catalog(tmp_path: Path) -> None:
    catalog = DatasetCatalog(data_dir=tmp_path / "data")
    datasets = catalog.list_datasets()
    assert len(datasets) >= 3
    assert "crowdhuman_mini" in datasets


def test_model_trainer() -> None:
    gen = SyntheticDataGenerator(seed=42)
    train_samples = gen.generate_batch(30)
    val_samples = gen.generate_batch(10)

    trainer = ModelTrainer(learning_rate=1e-3)
    loss = trainer.train_epoch(train_samples)
    assert loss > 0

    metrics = trainer.evaluate(val_samples)
    assert metrics.mAP_50 > 0.80
    assert metrics.interaction_precision > 0.80
    assert metrics.interaction_f1 > 0.80
