"""The dashboard: live feed, JSON API, and the HTTP server.

Every test starts a real server on port 0 - an OS-assigned free port - and talks
to it over real HTTP. Mocking the transport would test the routing table rather
than the thing that actually broke in every previous phase, which was always the
integration rather than the unit.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from vantage.core.errors import ConfigError
from vantage.dashboard.api import MAX_LIMIT, DashboardApi
from vantage.dashboard.live import LiveFeed, LiveSnapshot
from vantage.dashboard.server import DashboardServer
from vantage.storage.sqlite_store import SqliteStore


def frame(width: int = 64, height: int = 48) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def snapshot(index: int = 1, **kwargs) -> LiveSnapshot:
    return LiveSnapshot(frame_index=index, captured_at=time.time(), **kwargs)


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    created = SqliteStore(tmp_path / "dash.db")
    yield created
    created.close()


@pytest.fixture
def server_factory():
    started: list[DashboardServer] = []

    def build(api: DashboardApi, feed: LiveFeed | None = None) -> DashboardServer:
        # Port 0: the OS picks a free one, so tests never collide with each
        # other or with anything already listening.
        server = DashboardServer(api, feed, host="127.0.0.1", port=0)
        server.start()
        started.append(server)
        return server

    yield build
    for server in started:
        server.stop()


def get(url: str, timeout: float = 5.0):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


class TestLiveFeed:
    def test_holds_only_the_latest_frame(self) -> None:
        """A queue would let a slow browser grow memory in the analysis process."""
        feed = LiveFeed()
        for index in range(50):
            feed.publish(frame(), snapshot(index))
        assert feed.snapshot().frame_index == 49

    def test_encodes_once_per_frame_not_per_viewer(self) -> None:
        feed = LiveFeed()
        feed.publish(frame(), snapshot())
        first = feed.latest_jpeg()
        assert first is not None
        assert feed.latest_jpeg() is first

    def test_downscales_a_large_frame(self) -> None:
        """A 1080p stream is megabytes a second per viewer for detail nobody reads."""
        small = LiveFeed(max_width=160)
        small.publish(frame(width=1920, height=1080), snapshot())
        big = LiveFeed(max_width=1920)
        big.publish(frame(width=1920, height=1080), snapshot())
        assert len(small.latest_jpeg()) < len(big.latest_jpeg())

    def test_waiting_returns_when_a_frame_arrives(self) -> None:
        feed = LiveFeed()
        result: list = []

        def wait():
            result.append(feed.wait_for_frame(-1, timeout=3.0))

        waiter = threading.Thread(target=wait)
        waiter.start()
        time.sleep(0.05)
        feed.publish(frame(), snapshot())
        waiter.join(timeout=3)
        assert result and result[0][1] is not None

    def test_waiting_times_out_with_a_stale_frame(self) -> None:
        """A frozen picture is a clearer signal than a dropped connection."""
        feed = LiveFeed()
        feed.publish(frame(), snapshot())
        _sequence, jpeg = feed.wait_for_frame(999, timeout=0.1)
        assert jpeg is not None

    def test_an_empty_frame_does_not_replace_a_good_one(self) -> None:
        feed = LiveFeed()
        feed.publish(frame(), snapshot(1))
        feed.publish(np.zeros((0, 0, 3), dtype=np.uint8), snapshot(2))
        assert feed.latest_jpeg() is not None
        assert feed.snapshot().frame_index == 2

    def test_viewer_count_tracks_open_streams(self) -> None:
        feed = LiveFeed()
        feed.viewer_opened()
        feed.viewer_opened()
        feed.viewer_closed()
        assert feed.viewers == 1

    def test_viewer_count_cannot_go_negative(self) -> None:
        feed = LiveFeed()
        feed.viewer_closed()
        assert feed.viewers == 0

    def test_invalid_quality_is_refused(self) -> None:
        with pytest.raises(ValueError, match="jpeg_quality"):
            LiveFeed(jpeg_quality=0)


class TestApi:
    def test_live_says_why_when_there_is_no_pipeline(self) -> None:
        """Empty data is indistinguishable from a quiet scene; a reason is not."""
        payload = DashboardApi().live({})
        assert payload["available"] is False
        assert "history only" in payload["reason"]

    def test_events_say_why_when_there_is_no_store(self) -> None:
        payload = DashboardApi().events({})
        assert payload["available"] is False
        assert "--store" in payload["reason"]

    def test_limit_is_capped(self) -> None:
        """Without a ceiling, one URL could ask for a million rows."""
        api = DashboardApi()
        assert api._limit_for_test({"limit": "999999"}) == MAX_LIMIT

    def test_a_bad_limit_is_reported(self) -> None:
        api = DashboardApi()
        with pytest.raises(ValueError, match="whole number"):
            api._limit_for_test({"limit": "many"})

    def test_unknown_route_lists_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match="available"):
            DashboardApi().handle("teleport", {})

    def test_timeline_requires_an_entity(self, store: SqliteStore) -> None:
        with pytest.raises(ValueError, match="entity"):
            DashboardApi(store=store).timeline({})

    def test_parameters_are_validated_even_without_a_store(self) -> None:
        """A malformed request is malformed whether or not there is data."""
        with pytest.raises(ValueError, match="whole number"):
            DashboardApi().events({"limit": "lots"})

    def test_events_come_from_the_store(self, store: SqliteStore) -> None:
        store.write_events(
            [
                {
                    "timestamp": time.time(),
                    "camera_id": "cam0",
                    "rule": "fall",
                    "severity": "alert",
                    "summary": "person_1 fell",
                    "entity_id": "person_1",
                    "identity": None,
                    "related_id": None,
                    "zone": "till",
                    "frame_index": 1,
                    "elapsed_s": 1.0,
                    "evidence": {"confidence": 0.9},
                }
            ]
        )
        payload = DashboardApi(store=store).events({"since": "1h"})
        assert payload["count"] == 1
        assert payload["events"][0]["severity"] == "alert"
        assert payload["events"][0]["identity"] is None


class TestServer:
    def test_serves_the_page(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        with urlopen(server.url + "/", timeout=5) as response:
            body = response.read()
        assert b"<!DOCTYPE html>" in body
        assert b"stream.mjpg" in body

    def test_serves_stats(self, server_factory, store: SqliteStore) -> None:
        server = server_factory(DashboardApi(store=store, camera_id="cam9"))
        payload = get(server.url + "/api/stats")
        assert payload["camera_id"] == "cam9"
        assert payload["store"]["events"] == 0

    def test_unknown_route_is_a_404(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        with pytest.raises(HTTPError) as excinfo:
            urlopen(server.url + "/nowhere", timeout=5)
        assert excinfo.value.code == 404

    def test_a_bad_parameter_is_a_400_not_a_500(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        with pytest.raises(HTTPError) as excinfo:
            urlopen(server.url + "/api/events?limit=lots", timeout=5)
        assert excinfo.value.code == 400

    def test_stream_without_a_feed_refuses_clearly(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        with pytest.raises(HTTPError) as excinfo:
            urlopen(server.url + "/stream.mjpg", timeout=5)
        assert excinfo.value.code == 503
        assert b"without a running pipeline" in excinfo.value.read()

    def test_snapshot_serves_a_jpeg(self, server_factory) -> None:
        feed = LiveFeed()
        feed.publish(frame(), snapshot())
        server = server_factory(DashboardApi(feed=feed), feed)
        with urlopen(server.url + "/snapshot.jpg", timeout=5) as response:
            body = response.read()
        assert body.startswith(b"\xff\xd8\xff")

    def test_mjpeg_delivers_multiple_frames(self, server_factory) -> None:
        """The property the live view depends on: it is a stream, not one image."""
        feed = LiveFeed()
        feed.publish(frame(), snapshot(0))
        server = server_factory(DashboardApi(feed=feed), feed)

        stop = threading.Event()

        def produce():
            index = 1
            while not stop.is_set():
                feed.publish(frame(), snapshot(index))
                index += 1
                time.sleep(0.01)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            connection = socket.create_connection(("127.0.0.1", server.port), timeout=5)
            connection.sendall(
                b"GET /stream.mjpg HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            connection.settimeout(1.0)
            data = b""
            deadline = time.time() + 1.5
            while time.time() < deadline:
                try:
                    chunk = connection.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    break
                data += chunk
            connection.close()
        finally:
            stop.set()
            producer.join(timeout=2)

        assert b"multipart/x-mixed-replace" in data
        assert data.count(b"--vantageframe") >= 2

    def test_live_reports_the_current_snapshot(self, server_factory) -> None:
        feed = LiveFeed()
        feed.publish(
            frame(),
            snapshot(
                42,
                entities=({"entity_id": "person_1", "label": "person", "motion": "moving"},),
            ),
        )
        server = server_factory(DashboardApi(feed=feed), feed)
        payload = get(server.url + "/api/live")
        assert payload["frame_index"] == 42
        assert payload["entities"][0]["entity_id"] == "person_1"
        assert payload["has_frame"] is True

    def test_concurrent_requests_are_served(self, server_factory, store: SqliteStore) -> None:
        """ThreadingHTTPServer, so one open MJPEG stream cannot block the API."""
        server = server_factory(DashboardApi(store=store))
        results: list[int] = []

        def hit():
            results.append(get(server.url + "/api/stats")["uptime_s"] is not None)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert len(results) == 8

    def test_stopping_is_idempotent(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        server.stop()
        server.stop()


class TestBinding:
    def test_loopback_by_default(self) -> None:
        """It serves live camera footage with no authentication."""
        from vantage.config.schema import DashboardConfig

        assert DashboardConfig().host == "127.0.0.1"

    def test_binding_wider_warns_about_what_it_means(self, caplog) -> None:
        import logging

        caplog.set_level(logging.WARNING, logger="vantage")
        DashboardServer(DashboardApi(), None, host="0.0.0.0", port=0)
        assert any("beyond loopback" in record.message for record in caplog.records)
        assert any("no authentication" in str(record.__dict__) for record in caplog.records)

    def test_an_impossible_port_is_refused(self) -> None:
        from vantage.config.schema import DashboardConfig

        with pytest.raises(ConfigError, match="port"):
            DashboardConfig(port=70000)

    def test_port_zero_is_allowed(self) -> None:
        """0 asks the OS for a free port - what embedding and tests need."""
        from vantage.config.schema import DashboardConfig

        assert DashboardConfig(port=0).port == 0

    def test_dashboard_flag_enables_it(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(build_parser().parse_args(["run", "--dashboard"]))
        assert "dashboard.enabled=true" in overrides


class TestPipelineIntegration:
    def test_a_run_serves_a_live_view(self, tmp_path: Path) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            AppConfig,
            DashboardConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=200"),
            ingest=IngestConfig(max_frames=60),
            app=AppConfig(resource_interval_s=0),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            dashboard=DashboardConfig(enabled=True, port=0),
            display=DisplayConfig(enabled=False),
        )
        # The run is short; this asserts the wiring holds end to end rather than
        # racing the server, which the server tests above already cover.
        result = run_ingestion(config, engine=engine)
        assert result.frames == 60

    def test_headless_still_renders_overlays_for_the_dashboard(self) -> None:
        """Otherwise the browser shows raw video and the analysis is invisible."""
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            AppConfig,
            DashboardConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=40"),
            ingest=IngestConfig(max_frames=20),
            app=AppConfig(resource_interval_s=0),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            dashboard=DashboardConfig(enabled=True, port=0),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)
        # The run completing at all proves the display block ran headless; the
        # regression it guards is a `continue` that skipped the shutdown check,
        # the stats log and the resource sample.
        assert result.frames == 20
        assert result.stats["frames_delivered"] == 20


class TestIdentityReachesTheDashboard:
    """The seam between identity and what a browser is shown.

    Every stage before this one was proved by its own tests, and the dashboard
    was proved against observations. Nothing covered the join: identity could
    work perfectly and the live view could still show anonymous entity ids
    forever, because ``_live_snapshot`` simply never asked for it.
    """

    def snapshot_for(self, identity):
        from vantage.app import _live_snapshot
        from vantage.core.frame import Frame
        from vantage.state.contracts import EntityState, MotionState

        entity = EntityState(
            track_id=7,
            entity_id="person_7",
            label="person",
            motion=MotionState.STATIONARY,
            speed=0.0,
            dwell_s=1.0,
            bearing_deg=None,
            distance=0.0,
            age_s=1.0,
            observed=True,
        )
        frame = Frame(
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            index=3,
            source_id="cam",
            capture_wall=0.0,
            capture_monotonic=0.0,
        )
        return _live_snapshot(
            frame,
            SimpleNamespace(delivery_fps=10.0, frames_dropped=0, source_id="cam"),
            [entity],
            None,
            None,
            None,
            None,
            identity,
            SimpleNamespace(to_dict=lambda: {}),
        )

    def identity_result(self, *, name: str, resolved: bool):
        from vantage.identity.contracts import EntityIdentity, IdentityResult

        return IdentityResult(
            identities=(
                EntityIdentity(
                    track_id=7,
                    entity_id="person_7",
                    name=name,
                    similarity=0.62,
                    votes=3,
                    resolved=resolved,
                    attempts=3,
                ),
            ),
            source_id="cam",
            frame_index=3,
            capture_wall=0.0,
        )

    def test_a_committed_name_reaches_the_live_entity(self) -> None:
        snapshot = self.snapshot_for(self.identity_result(name="alice", resolved=True))
        assert snapshot.entities[0]["identity"] == "alice"

    def test_a_provisional_name_does_not(self) -> None:
        """Half-decided is not a name.

        Rendered in a table, a provisional guess is indistinguishable from a
        committed one, which would defeat the vote threshold entirely.
        """
        snapshot = self.snapshot_for(self.identity_result(name="alice", resolved=False))
        assert snapshot.entities[0]["identity"] is None

    def test_unknown_is_not_shown_as_a_name(self) -> None:
        snapshot = self.snapshot_for(self.identity_result(name="unknown", resolved=True))
        assert snapshot.entities[0]["identity"] is None

    def test_the_key_exists_when_identity_is_off(self) -> None:
        """So the browser can tell "not running" from "running, nobody known"."""
        snapshot = self.snapshot_for(None)
        assert snapshot.entities[0]["identity"] is None


class TestThePageMatchesTheBackend:
    """Guards a bug class the dashboard hit twice in one sitting.

    The page is HTML and the contracts are Python enums, so nothing connects
    them and nothing complains when they drift. Both drifts shipped:

    * The health panel branched on ``stage.broken`` and ``stage.circuit_open``.
      Neither field exists - ``StageRegistry`` publishes ``disabled`` - so a
      stage whose circuit had opened rendered as "ok", in the one panel whose
      whole purpose is to say when something has stopped.
    * The event list styled ``info`` / ``warning`` / ``critical``. The real
      severities are ``info`` / ``notice`` / ``alert``, so two of the three
      rendered as unknown and the severity filter offered two options that could
      never match a single row.

    Both look right when read and are wrong against the data. These assert the
    page names things the backend actually produces.
    """

    @staticmethod
    def page() -> str:
        from vantage.dashboard import server as server_module

        return (Path(server_module.__file__).parent / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_every_severity_is_styled_and_filterable(self) -> None:
        from vantage.events.contracts import Severity

        page = self.page()
        for severity in Severity:
            assert f".ev.{severity.value} " in page or f".ev.{severity.value}." in page, (
                f"severity {severity.value!r} has no styling, so it renders "
                "indistinguishably from an unrecognised value"
            )
            assert f'<option value="{severity.value}">' in page, (
                f"severity {severity.value!r} is missing from the filter"
            )

    def test_the_filter_offers_no_severity_that_cannot_exist(self) -> None:
        import re

        from vantage.events.contracts import Severity

        real = {s.value for s in Severity}
        block = self.page().split('id="ev-sev"')[1].split("</select>")[0]
        offered = set(re.findall(r'<option value="([^"]*)"', block)) - {""}
        assert offered <= real, f"filter offers {offered - real}, which no event can be"

    def test_the_health_panel_reads_fields_that_exist(self) -> None:
        from vantage.core.resilience import StageRegistry

        registry = StageRegistry()
        registry.guard("detection").run(lambda: None)
        published = set(next(iter(registry.to_dict().values())))

        page = self.page()
        for field in ("disabled", "failures", "calls", "last_error"):
            assert field in published, f"StageRegistry no longer publishes {field!r}"
            assert f"stage.{field}" in page, f"the health panel ignores {field!r}"

    def test_every_chart_metric_exists(self) -> None:
        import re

        from vantage.analytics.contracts import Metric

        block = self.page().split('id="an-metric"')[1].split("</select>")[0]
        offered = set(re.findall(r'<option value="([^"]*)"', block))
        assert offered <= {m.value for m in Metric}, (
            f"the chart offers metrics the analytics engine does not have: "
            f"{offered - {m.value for m in Metric}}"
        )

    def test_the_page_makes_no_external_requests(self) -> None:
        """It is served locally to a machine that may have no internet.

        A font, an icon set or a charting library from a CDN turns a working
        dashboard into a broken-looking one the moment the network is gone -
        which, for something watching a camera, is exactly when it matters.
        """
        page = self.page()
        for marker in ("http://", "https://", "//cdn", "//unpkg", "//fonts."):
            offenders = [
                line.strip()
                for line in page.splitlines()
                if marker in line and "localhost" not in line and "://" in line
            ]
            assert not offenders, f"external reference: {offenders[:2]}"

    def test_the_javascript_parses(self) -> None:
        """A syntax error here kills the whole page, silently.

        Every other test in this file talks to the server and passes regardless
        of whether the browser can run what it was sent - the JSON is fine, the
        HTML is fine, and the page is blank. Node is used when present because
        it is the only parser available that agrees with a browser; where it is
        absent the check is skipped rather than faked.
        """
        import re
        import shutil
        import subprocess
        import tempfile

        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed; cannot parse the page's JavaScript")

        script = re.search(r"<script>(.*?)</script>", self.page(), re.S)
        assert script, "the page has no script block"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.js"
            path.write_text(script.group(1), encoding="utf-8")
            result = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True
            )
        assert result.returncode == 0, result.stderr
