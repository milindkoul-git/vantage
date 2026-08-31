"""Tests for PostgresStore Store protocol compliance."""

from __future__ import annotations

from unittest.mock import MagicMock

from vantage.storage.contracts import Store
from vantage.storage.postgres_store import PostgresStore


def test_postgres_store_protocol_compliance() -> None:
    mock_pool = MagicMock()
    store = PostgresStore(mock_pool)
    assert isinstance(store, Store)

    # Test writing
    mock_pool.executemany.return_value = None
    ev_count = store.write_events([{"timestamp": 100.0, "camera_id": "cam0"}])
    assert ev_count == 1
    assert mock_pool.executemany.called

    obs_count = store.write_observations([{"timestamp": 100.0, "camera_id": "cam0"}])
    assert obs_count == 1

    hb_count = store.write_heartbeats([{"timestamp": 100.0, "camera_id": "cam0"}])
    assert hb_count == 1

    rel_count = store.write_relationships([{"camera_id": "cam0", "entity_a": "p1"}])
    assert rel_count == 1

    # Test counts
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(42,)]
    mock_pool.execute.return_value = mock_cursor
    counts = store.counts()
    assert counts == {"events": 42, "observations": 42}

    store.close()
    assert store._closed is True
