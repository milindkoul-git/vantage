"""Storage: schema, batched writes, queries, retention and the write path.

No camera, no model, no runtime. Every database is a temporary file, so these
run in milliseconds and leave nothing behind.

Two of these tests exist because a functional run found what unit tests had
not: connections crossing threads, and an ISO timestamp stored into a numeric
column. Both are marked as regressions where they appear.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from vantage.core.errors import VantageError
from vantage.storage.contracts import Query, WriteStats
from vantage.storage.schema import SCHEMA_VERSION, like_term, unwrap_list, wrap_list
from vantage.storage.sqlite_store import SqliteStore
from vantage.storage.writer import StoreWriter


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    created = SqliteStore(tmp_path / "test.db")
    yield created
    created.close()


def event_row(
    *,
    timestamp: float,
    rule: str = "fall",
    severity: str = "alert",
    entity_id: str | None = "person_1",
    zone: str | None = None,
    evidence: dict | None = None,
) -> dict:
    return {
        "timestamp": timestamp,
        "camera_id": "cam0",
        "rule": rule,
        "severity": severity,
        "summary": f"{entity_id} {rule}",
        "entity_id": entity_id,
        "identity": None,
        "related_id": None,
        "zone": zone,
        "frame_index": 1,
        "elapsed_s": 1.0,
        "evidence": evidence or {"confidence": 0.9},
    }


def observation_row(
    *,
    timestamp: float,
    entity_id: str = "person_1",
    zones: list[str] | None = None,
    activities: list[str] | None = None,
    entity_type: str = "person",
) -> dict:
    return {
        "timestamp": timestamp,
        "camera_id": "cam0",
        "entity_id": entity_id,
        "identity": None,
        "entity_type": entity_type,
        "motion": "moving",
        "speed": 0.7,
        "posture": "standing",
        "zones": wrap_list(zones or []),
        "activities": wrap_list(activities or []),
        "frame_index": 1,
        "elapsed_s": 1.0,
    }


class TestSchema:
    def test_a_fresh_database_records_its_version(self, store: SqliteStore) -> None:
        """Impossible to add later: by the time a migration is needed there are
        databases in the field with no version marker."""
        assert store.schema_version == SCHEMA_VERSION

    def test_reopening_does_not_duplicate_the_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "reopen.db"
        SqliteStore(path).close()
        second = SqliteStore(path)
        assert second.schema_version == SCHEMA_VERSION
        second.close()

    def test_a_newer_schema_is_refused(self, tmp_path: Path) -> None:
        """A schema this code does not understand may have columns it would
        silently ignore, and a store that discards fields is worse than one
        that will not open."""
        path = tmp_path / "future.db"
        first = SqliteStore(path)
        first._require().execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION + 5, time.time()),
        )
        first.close()
        with pytest.raises(VantageError, match="newer than this build"):
            SqliteStore(path)

    def test_opening_a_missing_store_read_only_explains_why(self, tmp_path: Path) -> None:
        with pytest.raises(VantageError, match="--store"):
            SqliteStore(tmp_path / "absent.db", read_only=True)

    def test_list_columns_round_trip(self) -> None:
        assert unwrap_list(wrap_list(["till", "lobby"])) == ("till", "lobby")
        assert wrap_list([]) is None
        assert unwrap_list(None) == ()

    def test_wrapping_prevents_a_prefix_match(self) -> None:
        """Without the delimiters, 'till' silently matches 'till_annexe'."""
        wrapped = wrap_list(["till_annexe"])
        assert like_term("till").strip("%") not in wrapped


class TestWritingAndReading:
    def test_events_round_trip_with_evidence(self, store: SqliteStore) -> None:
        now = time.time()
        store.write_events([event_row(timestamp=now, evidence={"why": "measured", "n": 3})])
        rows = store.events(Query())
        assert len(rows) == 1
        assert rows[0].evidence == {"why": "measured", "n": 3}
        assert rows[0].identity is None

    def test_the_timestamp_is_numeric_not_a_string(self, store: SqliteStore) -> None:
        """Regression, found by storing a real run rather than a fixture.

        Event.to_record() renders the timestamp as an ISO string, which is right
        for JSON and wrong for a REAL column that range queries sort on. SQLite
        is dynamically typed and accepted it, and the first query that formatted
        one failed with "'str' object cannot be interpreted as an integer".
        """
        store.write_events([event_row(timestamp=time.time())])
        row = store.events(Query())[0]
        assert isinstance(row.timestamp, float)
        row.when.isoformat()  # would raise if a string had been stored

    def test_observations_round_trip(self, store: SqliteStore) -> None:
        store.write_observations(
            [observation_row(timestamp=time.time(), zones=["till"], activities=["walking"])]
        )
        row = store.observations(Query())[0]
        assert row.zones == "till"
        assert row.activities == "walking"

    def test_an_empty_batch_writes_nothing(self, store: SqliteStore) -> None:
        assert store.write_events([]) == 0
        assert store.counts()["events"] == 0

    def test_malformed_evidence_does_not_break_the_query(self, store: SqliteStore) -> None:
        """One bad row must not make the other ten thousand unreadable."""
        store.write_events([event_row(timestamp=time.time())])
        store._require().execute("UPDATE events SET evidence = 'not json'")
        row = store.events(Query())[0]
        assert "_unparsed" in row.evidence


class TestQueries:
    def populate(self, store: SqliteStore, now: float) -> None:
        store.write_events(
            [
                event_row(timestamp=now - 3600, rule="fall", severity="alert"),
                event_row(timestamp=now - 60, rule="loitering", severity="notice"),
                event_row(
                    timestamp=now - 10,
                    rule="zone_entry",
                    severity="info",
                    entity_id="person_2",
                    zone="till",
                ),
            ]
        )

    def test_time_window(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        assert len(store.events(Query(since=now - 120))) == 2

    def test_severity_filter(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        assert len(store.events(Query(severity="alert"))) == 1

    def test_rule_filter(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        assert store.events(Query(rule="loitering"))[0].rule == "loitering"

    def test_entity_filter(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        assert len(store.events(Query(entity_id="person_2"))) == 1

    def test_zone_filter_on_events(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        assert len(store.events(Query(zone="till"))) == 1

    def test_ordering_and_limit(self, store: SqliteStore) -> None:
        now = time.time()
        self.populate(store, now)
        newest = store.events(Query(limit=1))
        oldest = store.events(Query(limit=1, newest_first=False))
        assert newest[0].timestamp > oldest[0].timestamp

    def test_timeline_is_oldest_first(self, store: SqliteStore) -> None:
        now = time.time()
        store.write_events([event_row(timestamp=now - offset) for offset in (300, 200, 100)])
        stamps = [row.timestamp for row in store.timeline("person_1")]
        assert stamps == sorted(stamps)

    def test_zone_filter_does_not_match_a_prefix(self, store: SqliteStore) -> None:
        """The reason list columns are stored delimiter-wrapped."""
        now = time.time()
        store.write_observations([observation_row(timestamp=now, zones=["till_annexe"])])
        assert store.observations(Query(zone="till")) == []
        assert len(store.observations(Query(zone="till_annexe"))) == 1

    def test_an_inverted_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="inverted"):
            Query(since=100.0, until=50.0)

    def test_a_zero_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            Query(limit=0)


class TestRetention:
    def test_prune_removes_only_old_rows(self, store: SqliteStore) -> None:
        now = time.time()
        store.write_events([event_row(timestamp=now - 86400), event_row(timestamp=now)])
        store.write_observations(
            [observation_row(timestamp=now - 86400), observation_row(timestamp=now)]
        )
        removed = store.prune(now - 3600)
        assert removed == {"observations": 1, "events": 1, "heartbeat": 0}
        assert store.counts()["events"] == 1

    def test_prune_on_an_empty_store_is_harmless(self, store: SqliteStore) -> None:
        assert store.prune(time.time()) == {"observations": 0, "events": 0, "heartbeat": 0}

    def test_size_includes_the_wal_sidecar(self, store: SqliteStore) -> None:
        """The main file alone reported 4096 bytes for five hundred rows -
        the size of an empty database."""
        store.write_observations(
            [observation_row(timestamp=time.time() + i) for i in range(200)]
        )
        assert store.size_bytes() > 10_000


class TestWriter:
    def test_records_reach_the_store(self, store: SqliteStore) -> None:
        writer = StoreWriter(store, batch_size=10, flush_interval_s=0.05)
        for i in range(25):
            writer.add_observation(observation_row(timestamp=time.time() + i))
        writer.add_event(event_row(timestamp=time.time()))
        writer.close()
        counts = store.counts()
        assert counts["observations"] == 25
        assert counts["events"] == 1

    def test_the_connection_may_cross_threads(self, store: SqliteStore) -> None:
        """Regression: one shared connection failed on the first batch with
        "SQLite objects created in a thread can only be used in that same
        thread". No unit test caught it because unit tests do not cross
        threads; the first functional run did."""
        writer = StoreWriter(store, batch_size=2, flush_interval_s=0.05)
        writer.add_event(event_row(timestamp=time.time()))
        writer.close()
        assert store.counts()["events"] == 1
        assert writer.stats.write_errors == 0

    def test_a_full_event_queue_is_reported_not_swallowed(self, store: SqliteStore) -> None:
        """An event is the output of a rule that already decided it mattered."""
        writer = StoreWriter(store, batch_size=10_000, flush_interval_s=999.0, event_queue=2)
        accepted = [writer.add_event(event_row(timestamp=time.time())) for _ in range(10)]
        assert accepted.count(False) > 0
        assert writer.stats.events_dropped > 0
        writer.close()

    def test_observations_cannot_crowd_out_events(self, store: SqliteStore) -> None:
        """Separate queues: a flood of one must not lose the other."""
        writer = StoreWriter(
            store,
            batch_size=10_000,
            flush_interval_s=999.0,
            observation_queue=3,
            event_queue=50,
        )
        for _ in range(100):
            writer.add_observation(observation_row(timestamp=time.time()))
        assert writer.add_event(event_row(timestamp=time.time())) is True
        assert writer.stats.observations_dropped > 0
        assert writer.stats.events_dropped == 0
        writer.close()

    def test_close_flushes_what_is_queued(self, store: SqliteStore) -> None:
        """Otherwise the last batch is lost on every clean shutdown - the most
        reproducible data loss the system could have."""
        writer = StoreWriter(store, batch_size=10_000, flush_interval_s=999.0)
        writer.add_event(event_row(timestamp=time.time()))
        writer.close()
        assert store.counts()["events"] == 1

    def test_a_write_failure_does_not_kill_the_thread(self, store: SqliteStore) -> None:
        class Broken:
            def write_events(self, records):
                raise RuntimeError("disk on fire")

            def write_observations(self, records):
                raise RuntimeError("disk on fire")

        writer = StoreWriter(Broken(), batch_size=1, flush_interval_s=0.05)
        writer.add_event(event_row(timestamp=time.time()))
        time.sleep(0.2)
        writer.add_event(event_row(timestamp=time.time()))
        stats = writer.close()
        assert stats.write_errors >= 1
        assert "disk on fire" in stats.last_error

    def test_close_is_idempotent(self, store: SqliteStore) -> None:
        writer = StoreWriter(store)
        writer.close()
        writer.close()

    def test_stats_report_dropped_events_prominently(self) -> None:
        stats = WriteStats(events_written=3, events_dropped=2)
        assert "EVENTS DROPPED" in stats.describe()
        assert not stats.healthy


class TestRecorder:
    def build(self, store: SqliteStore, **kwargs):
        from vantage.storage.recorder import Recorder

        writer = StoreWriter(store, batch_size=1000, flush_interval_s=0.05)
        return Recorder(writer, camera_id="cam0", **kwargs), writer

    def state_result(self, index: int = 0):
        from vantage.state.contracts import EntityState, MotionState, StateResult

        return StateResult(
            states=(
                EntityState(
                    track_id=1,
                    entity_id="person_1",
                    label="person",
                    motion=MotionState.MOVING,
                    speed=0.6,
                    dwell_s=2.0,
                    bearing_deg=90.0,
                    distance=1.0,
                    age_s=5.0,
                    observed=True,
                ),
            ),
            source_id="s",
            frame_index=index,
            capture_wall=1000.0 + index,
            elapsed_s=1 / 30,
        )

    def test_observations_are_sampled_by_interval(self, store: SqliteStore) -> None:
        recorder, writer = self.build(store, observation_interval=10)
        for index in range(100):
            recorder.record(state=self.state_result(index))
        writer.close()
        assert store.counts()["observations"] == 10

    def test_observations_can_be_turned_off_entirely(self, store: SqliteStore) -> None:
        recorder, writer = self.build(store, store_observations=False)
        for index in range(50):
            recorder.record(state=self.state_result(index))
        writer.close()
        assert store.counts()["observations"] == 0

    def test_events_are_never_sampled(self, store: SqliteStore) -> None:
        from vantage.events.contracts import Event, EventResult, Severity

        recorder, writer = self.build(store, observation_interval=1000)
        for index in range(5):
            recorder.record(
                events=EventResult(
                    events=(
                        Event(
                            "fall",
                            Severity.ALERT,
                            "person_1 fell",
                            "person_1",
                            1,
                            index,
                            1000.0 + index,
                            float(index),
                        ),
                    ),
                    source_id="s",
                    frame_index=index,
                    capture_wall=1000.0 + index,
                )
            )
        writer.close()
        assert store.counts()["events"] == 5

    def test_an_invalid_interval_is_refused(self, store: SqliteStore) -> None:
        with pytest.raises(ValueError, match="observation_interval"):
            self.build(store, observation_interval=0)


class TestConfig:
    def test_storage_is_off_by_default(self) -> None:
        """A tool that silently created a growing database would be a surprise."""
        from vantage.config.schema import VantageConfig

        assert VantageConfig().storage.enabled is False

    def test_events_may_not_be_pruned_sooner_than_observations(self) -> None:
        from vantage.config.schema import StorageConfig

        with pytest.raises(VantageError, match="event_retention_days"):
            StorageConfig(retention_days=30, event_retention_days=7)

    def test_store_flag_enables_it(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(build_parser().parse_args(["run", "--store"]))
        assert "storage.enabled=true" in overrides


class TestDurationParsing:
    def test_units(self) -> None:
        from vantage.storage.query_cli import parse_duration

        assert parse_duration("30m") == 1800.0
        assert parse_duration("2h") == 7200.0
        assert parse_duration("7d") == 604800.0

    def test_an_unknown_unit_says_what_is_valid(self) -> None:
        from vantage.storage.query_cli import parse_duration

        with pytest.raises(VantageError, match="30m, 6h, 7d"):
            parse_duration("5 fortnights")

    def test_a_negative_duration_is_refused(self) -> None:
        from vantage.storage.query_cli import parse_duration

        with pytest.raises(VantageError, match="negative"):
            parse_duration("-3h")


class TestPipelineIntegration:
    def test_a_run_writes_and_the_store_can_be_read(self, tmp_path: Path) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            AppConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            StorageConfig,
            TrackingConfig,
            VantageConfig,
        )

        path = tmp_path / "run.db"
        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=80"),
            ingest=IngestConfig(max_frames=60),
            app=AppConfig(resource_interval_s=0),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            storage=StorageConfig(
                enabled=True, path=str(path), observation_interval=5, retention_days=0
            ),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)

        assert "storage:" in result.summary()
        assert result.storage_summary["observations_written"] > 0
        assert result.storage_summary["events_dropped"] == 0

        with SqliteStore(path, read_only=True) as store:
            rows = store.observations(Query(limit=1000))
            assert rows
            assert all(row.identity is None for row in rows)
            json.dumps(store.counts())

    def test_storage_off_writes_no_file(self, tmp_path: Path) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            AppConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            StorageConfig,
            VantageConfig,
        )

        path = tmp_path / "never.db"
        engine, _ = make_engine()
        run_ingestion(
            VantageConfig(
                source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=20"),
                ingest=IngestConfig(max_frames=10),
                app=AppConfig(resource_interval_s=0),
                detection=DetectionConfig(enabled=True),
                storage=StorageConfig(enabled=False, path=str(path)),
                display=DisplayConfig(enabled=False),
            ),
            engine=engine,
        )
        assert not path.exists()


class TestConcurrency:
    def test_a_reader_is_not_blocked_by_a_writer(self, tmp_path: Path) -> None:
        """The whole reason for WAL: the CLI queries a file a run is writing."""
        path = tmp_path / "concurrent.db"
        writer_store = SqliteStore(path)
        writer = StoreWriter(writer_store, batch_size=50, flush_interval_s=0.05)

        stop = threading.Event()

        def produce():
            index = 0
            while not stop.is_set() and index < 2000:
                writer.add_observation(observation_row(timestamp=time.time() + index))
                index += 1

        thread = threading.Thread(target=produce)
        thread.start()
        try:
            time.sleep(0.15)
            reader = SqliteStore(path, read_only=True)
            reader.events(Query(limit=10))
            reader.observations(Query(limit=10))
            reader.close()
        finally:
            stop.set()
            thread.join(timeout=5)
            writer.close()
            writer_store.close()
