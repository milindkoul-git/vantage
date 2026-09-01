"""The dashboard: live feed, JSON API, and the HTTP server.

Every test starts a real server on port 0 - an OS-assigned free port - and talks
to it over real HTTP. Mocking the transport would test the routing table rather
than the thing that actually broke in every previous phase, which was always the
integration rather than the unit.
"""

from __future__ import annotations

import json
import re
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

from ._frontend_contract import FRONTEND, STATIC, built, object_keys, source, string_array


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
        # The shell loads the application; the application is what references
        # the stream. Asserting on the shell alone would pass for a page whose
        # script tag pointed at nothing.
        assert b'<div id="root">' in body
        assert b"assets/index.js" in body

    def test_serves_the_application_bundle(self, server_factory) -> None:
        server = server_factory(DashboardApi())
        with urlopen(server.url + "/assets/index.js", timeout=5) as response:
            assert response.status == 200
            body = response.read()
        assert b"stream.mjpg" in body, "the bundle never asks for the live stream"

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
    """Guards a bug class the dashboard hit three times.

    The page and the contracts are written in different languages, so nothing
    connects them and nothing complains when they drift. Three drifts shipped:

    * The health panel branched on ``stage.broken`` and ``stage.circuit_open``.
      Neither field exists - ``StageRegistry`` publishes ``disabled`` - so a
      stage whose circuit had opened rendered as "ok", in the one panel whose
      whole purpose is to say when something has stopped.
    * The event list styled ``info`` / ``warning`` / ``critical``. The real
      severities are ``info`` / ``notice`` / ``alert``, so two of the three
      rendered as unknown and the severity filter offered two options that could
      never match a single row.
    * The whole TypeScript contract file was written against a fixture set
      rather than against the server - ``start_time`` for ``first_seen``,
      ``"ACTIVE"`` for ``"active"``, a severity breakdown sharing no field name
      with the seven the server publishes. The page worked only in demo mode,
      which is why demo mode was the default.

    These read the front end's own vocabulary module, which is what its
    components import, and check it against the Python enums. Reading the served
    HTML instead is what let a hidden block of decoy ``<select>`` elements keep
    an earlier version of these tests green while the real controls did not
    exist.
    """

    @staticmethod
    def vocabulary() -> str:
        return source("src/contracts/vocabulary.ts")

    def test_every_severity_is_offered_and_no_others(self) -> None:
        from vantage.events.contracts import Severity

        offered = string_array(self.vocabulary(), "SEVERITIES")
        assert set(offered) == {s.value for s in Severity}, (
            "the front end's severity list has drifted from the Severity enum; "
            "a value it invents is a filter option no row can match, and one it "
            "misses renders as an unrecognised value"
        )

    def test_every_severity_is_labelled_ranked_and_coloured(self) -> None:
        from vantage.events.contracts import Severity

        real = {s.value for s in Severity}
        vocabulary = self.vocabulary()
        for name in ("SEVERITY_LABELS", "SEVERITY_RANK"):
            assert set(object_keys(vocabulary, name)) == real, f"{name} does not cover Severity"

        colours = object_keys(vocabulary, "SEVERITY_COLOR")
        assert set(colours) == real, "SEVERITY_COLOR does not cover Severity"

    def test_every_severity_has_a_style_rule(self) -> None:
        from vantage.events.contracts import Severity

        css = source("src/index.css")
        for severity in Severity:
            assert f".ev-{severity.value}" in css, (
                f"severity {severity.value!r} has no styling, so it renders "
                "indistinguishably from an unrecognised value"
            )

    def test_the_severity_filter_is_generated_from_the_vocabulary(self) -> None:
        """Not typed out again in the component, which is how it drifted before."""
        workspace = source("src/features/investigate/InvestigateWorkspace.tsx")
        assert 'id="ev-sev"' in workspace, "the severity filter is gone from the event log"
        assert "SEVERITIES.map(" in workspace, (
            "the severity filter enumerates its own options instead of mapping "
            "over SEVERITIES, which is exactly how it came to offer 'warning'"
        )

    def test_every_chart_metric_exists(self) -> None:
        from vantage.analytics.contracts import Metric

        offered = set(string_array(self.vocabulary(), "METRICS"))
        real = {m.value for m in Metric}
        assert offered <= real, (
            f"the chart offers metrics the analytics engine does not have: {offered - real}"
        )
        assert offered == real, f"the chart cannot show these metrics at all: {real - offered}"

    def test_the_metric_and_window_controls_are_real(self) -> None:
        """They were once a display:none block that satisfied this test alone."""
        workspace = source("src/features/analytics/AnalyticsWorkspace.tsx")
        for control in ('id="an-metric"', 'id="an-since"'):
            assert control in workspace, f"{control} is missing from the analytics panel"
        assert "METRICS.map(" in workspace and "ANALYTICS_WINDOWS.map(" in workspace
        assert "api.analytics(" in workspace, (
            "the analytics panel does not call the analytics endpoint, so its "
            "controls would be decoration"
        )

    def test_every_analytics_window_parses(self) -> None:
        from vantage.analytics.cli import parse_window

        offered = re.findall(r"value: '([^']+)'", self.vocabulary())
        assert offered, "the window selector offers nothing"
        for window in offered:
            assert parse_window(window) > 0, (
                f"the page offers a window {window!r} the CLI rejects"
            )

    def test_the_health_panel_reads_fields_that_exist(self) -> None:
        from vantage.core.resilience import StageRegistry

        registry = StageRegistry()
        registry.guard("detection").run(lambda: None)
        published = set(next(iter(registry.to_dict().values())))

        declared = string_array(self.vocabulary(), "STAGE_FIELDS")
        assert set(declared) <= published, (
            f"STAGE_FIELDS names fields StageRegistry does not publish: {set(declared) - published}"
        )

        drawer = source("src/components/shell/OperationsDrawer.tsx")
        for field in ("disabled", "failures", "calls", "last_error"):
            assert field in published, f"StageRegistry no longer publishes {field!r}"
            assert f"stage.{field}" in drawer, f"the health panel ignores {field!r}"

    def test_the_incident_states_match(self) -> None:
        from vantage.incident.models import IncidentState

        offered = set(string_array(self.vocabulary(), "INCIDENT_STATES"))
        assert offered == {s.value for s in IncidentState}, (
            "incident states have drifted; the page once expected 'ACTIVE' while "
            "the API has only ever sent 'active', so no incident matched any filter"
        )

    def test_the_page_ships_no_fixture_data(self) -> None:
        """The demo fixtures are gone, and must not come back by the front door.

        A hand-written intelligence snapshot shipped inside the bundle, switched
        on by default, is the specific thing this project forbids: a feature that
        is only mocked but presented as functional.
        """
        assert not (FRONTEND / "src" / "data" / "fixtures").exists(), (
            "the demo fixture directory is back"
        )
        for name in ("src/data/source.ts", "src/store/useInvestigationStore.ts"):
            # Comments are stripped first: both files explain the demo mode that
            # was removed, and a check that cannot tell a mention from a
            # declaration would forbid recording why it went.
            text = re.sub(r"/\*.*?\*/", "", source(name), flags=re.S)
            text = re.sub(r"//.*", "", text)
            for banned in ("DemoDataSource", "isDemoMode", "getDataSource"):
                assert banned not in text, (
                    f"{name} reintroduces {banned}; the dashboard must show the "
                    "pipeline's own output or say it has none"
                )

    def test_the_page_makes_no_external_requests(self) -> None:
        """It is served locally to a machine that may have no internet.

        A font, an icon set or a charting library from a CDN turns a working
        dashboard into a broken-looking one the moment the network is gone -
        which, for something watching a camera, is exactly when it matters.

        Checks the built artefacts, not the sources: a stylesheet that imports
        Google Fonts leaves no trace in the HTML shell.
        """
        page = built("index.html")
        css = built("assets/index.css")

        # Inline data: URIs are not requests, and an SVG one legitimately
        # contains http://www.w3.org/2000/svg - an XML namespace *name*, which no
        # browser ever fetches. Removing them first is the difference between
        # testing for network access and testing for the letters "http".
        for text, what in ((page, "index.html"), (css, "index.css")):
            without_data_uris = re.sub(r'"data:[^"]*"', '"data:"', text)
            without_data_uris = re.sub(r"'data:[^']*'", "'data:'", without_data_uris)
            without_data_uris = re.sub(r"url\(data:[^)]*\)", "url(data:)", without_data_uris)

            pattern = r"""(?:src|href)\s*=\s*["']([^"']+)|url\(\s*["']?([^)"']+)"""
            for a, b in re.findall(pattern, without_data_uris):
                target = (a or b).strip()
                if not target or target.startswith(("data:", "#", "/", "./", "../")):
                    continue
                assert not target.startswith(("http://", "https://", "//")), (
                    f"{what} fetches {target!r} from the network"
                )

    def test_the_fonts_the_design_names_are_actually_bundled(self) -> None:
        """Naming a family in a font stack does not load it.

        The design specified Source Serif 4, IBM Plex Mono and Inter and shipped
        with none of them: the browser silently fell back to Georgia, Consolas
        and Segoe UI, so the page never looked the way it was drawn. Fetching
        them from a CDN is ruled out by the test above, so they are bundled.
        """
        css = built("assets/index.css")
        assert "@font-face" in css, "no font is bundled; the design's families would not load"
        for family in ("Inter", "IBM Plex Mono", "Source Serif 4"):
            assert family in css, f"{family} is named in the design but not bundled"

    def test_the_built_javascript_parses(self) -> None:
        """A syntax error here kills the whole page, silently.

        Every other test in this file talks to the server and passes regardless
        of whether the browser can run what it was sent - the JSON is fine, the
        HTML is fine, and the page is blank. That happened: a temporal dead zone
        error at module scope threw before the first render and the dashboard was
        entirely dead while the suite was entirely green.

        Node is used because it is the only parser available that agrees with a
        browser; where it is absent the check is skipped rather than faked.
        """
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed; cannot parse the page's JavaScript")

        bundle = STATIC / "assets" / "index.js"
        if not bundle.is_file():
            pytest.skip("no built bundle; run `npm run build` in frontend/")

        # Via a file rather than stdin: the bundle contains characters outside
        # this console's code page, and piping it through a cp1252 stdin raises
        # in Python before node ever sees the script.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.mjs"
            path.write_bytes(bundle.read_bytes())
            result = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True
            )
        assert result.returncode == 0, result.stderr

    def test_the_bundle_is_not_absurd(self) -> None:
        """The dashboard is served over localhost, but not always to a fast machine.

        three.js is most of the JavaScript this app can load and exactly one
        panel uses it, so it is split out. This is a floor under that decision:
        without it the split silently reverts on the next refactor and every
        workspace pays for the 3D renderer again.
        """
        entry = STATIC / "assets" / "index.js"
        if not entry.is_file():
            pytest.skip("no built bundle; run `npm run build` in frontend/")
        size_kb = entry.stat().st_size / 1024
        assert size_kb < 400, (
            f"the entry bundle is {size_kb:.0f} kB; three.js has probably been "
            "pulled back onto the critical path"
        )

    def test_motion_is_budgeted_in_one_place(self) -> None:
        """Durations live in a token file, not scattered through components.

        The console is watched for a whole shift by someone looking for the one
        thing that changed, so every animation is a claim that something
        happened. Thirty independently-chosen durations is thirty small
        decisions nobody reviewed; one file is a budget somebody can argue with.
        """
        motion = source("src/lib/motion.ts")
        for name in ("micro", "panel", "view"):
            assert f"{name}:" in motion, f"the motion budget has no {name} duration"
        assert "prefersReducedMotion" in motion
        assert "staggerStep" in motion

    def test_reduced_motion_removes_rather_than_shortens(self) -> None:
        """It is a request to take the animation away, not to hurry it.

        A 40ms version of a transition is still a transition to somebody it
        makes ill, so `duration()` returns zero rather than something smaller.
        """
        motion = source("src/lib/motion.ts")
        block = motion.split("export function duration(")[1].split("}")[0]
        assert "return prefersReducedMotion() ? 0" in block, (
            "duration() shortens under reduced motion instead of removing"
        )

    def test_the_page_has_a_skip_link_to_the_workspace(self) -> None:
        """Six workspaces of navigation sit above the panel someone came to read."""
        app = source("src/App.tsx")
        assert 'className="skip-link"' in app
        assert 'href="#workspace"' in app
        assert 'id="workspace"' in app, "the skip link points at nothing"
        assert ".skip-link" in source("src/index.css")

    def test_focus_is_styled_beyond_the_global_ring(self) -> None:
        """Missing focus states are the most-cited failure in animated interfaces.

        A single global rule is the floor. A ring around a small chip and one
        around a full-width row should not be the same event.
        """
        css = source("src/index.css")
        assert css.count(":focus-visible") >= 4, (
            "only a global focus ring; per-surface treatments are missing"
        )

    def test_view_transitions_are_capped_and_defeatable(self) -> None:
        css = source("src/index.css")
        assert "::view-transition-old(root)" in css
        # A workspace change is the largest move the console allows, and even
        # that stays under the 200ms ceiling.
        match = re.search(
            r"::view-transition-old\(root\).*?animation-duration: (\d+)ms", css, re.S
        )
        assert match and int(match.group(1)) <= 200, "workspace transition exceeds the budget"
        assert "::view-transition-group(*)" in css, (
            "view transitions are animations the reduced-motion reset cannot reach; "
            "they need their own rule"
        )

    def test_the_twin_reports_its_own_renderer(self) -> None:
        """A page meant to run for a shift needs the leak check a demo does not.

        The three.js guidance for long-running scenes is to watch
        `renderer.info`: counts that climb while the scene is static mean
        something is not being disposed. It belongs in the operations drawer,
        beside the pipeline's own health, rather than in a floating overlay.
        """
        twin = source("src/components/visualizations/SpatialTwin3D.tsx")
        assert "renderer.info.memory" in twin
        drawer = source("src/components/shell/OperationsDrawer.tsx")
        for field in ("geometries", "textures", "calls"):
            assert field in drawer, f"the drawer does not report {field}"

    def test_the_twin_stops_rendering_when_nobody_can_see_it(self) -> None:
        """A hidden tab holding a WebGL loop open competes with the inference
        running in the same process, for as long as it stays hidden."""
        twin = source("src/components/visualizations/SpatialTwin3D.tsx")
        assert "visibilitychange" in twin
        assert "document.hidden" in twin

    def test_entities_are_drawn_as_instances(self) -> None:
        """One mesh per entity is one draw call per entity, plus a fresh geometry
        and material allocated on every poll."""
        twin = source("src/components/visualizations/SpatialTwin3D.tsx")
        assert "InstancedMesh" in twin
        assert "setColorAt" in twin, "colour per instance, not a material per entity"

    def test_a_type_scale_exists_and_components_use_it(self) -> None:
        """Juries reward consistent scale and rhythm at every breakpoint, which
        is not achievable while half the sizes are written inline."""
        config = source("tailwind.config.js")
        for step in ("micro", "tiny", "body", "lede", "title", "display"):
            assert f"'{step}'" in config, f"the type scale has no {step} step"

        graph = source("src/components/visualizations/ForceDirectedGraph.tsx")
        inline = re.findall(r"fontSize: '[^']+'", graph)
        # Three remain, and all three are the scale's own definitions.
        assert len(inline) <= 3, f"{len(inline)} inline font sizes left in the graph"
