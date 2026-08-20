"""Identity: the gallery, the store, the audit trail, and the safeguards.

Most of this needs no model. Templates are vectors, and the matching rule,
voting, revocation and audit behaviour are all checkable against synthetic ones -
which is fortunate, because a test suite for a face recogniser should not need a
collection of faces to run.

The tests that do need the real models are marked ``model`` and skip cleanly,
like every other weight-dependent test here.

What is deliberately not tested
-------------------------------
Whether two *real* people are told apart. That needs labelled faces of at least
two consenting individuals, which this project has no business collecting and
which no CI runner should hold. The mechanism is tested exhaustively with
synthetic templates; the accuracy claim is inherited from the model's authors
and the README says so.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vantage.core.errors import ConfigError
from vantage.identity.contracts import (
    EMBEDDING_DIM,
    UNKNOWN,
    AuditAction,
    AuditRecord,
    Enrollment,
    IdentityMatch,
)
from vantage.identity.gallery import Gallery, average_templates
from vantage.identity.store import IdentityStore


def template(seed: int) -> np.ndarray:
    """A deterministic unit-norm vector standing in for a face."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


def enrollment(name: str, seed: int, samples: int = 4) -> Enrollment:
    return Enrollment(
        name=name,
        template=tuple(float(v) for v in template(seed)),
        samples=samples,
        enrolled_at=time.time(),
        note="synthetic",
    )


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """A vector `weight` of the way from a to b - a controllable similarity."""
    mixed = (1.0 - weight) * a + weight * b
    return mixed / np.linalg.norm(mixed)


@pytest.fixture
def store(tmp_path: Path) -> IdentityStore:
    created = IdentityStore(tmp_path / "identities.db")
    yield created
    created.close()


class TestContracts:
    def test_a_wrong_length_template_is_refused(self) -> None:
        """A wrong-length vector produces meaningless similarities, not an error."""
        with pytest.raises(ValueError, match="dimensions"):
            Enrollment(name="x", template=(1.0, 2.0), samples=1, enrolled_at=0.0)

    def test_an_enrolment_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Enrollment(
                name="  ", template=tuple([0.0] * EMBEDDING_DIM), samples=1, enrolled_at=0.0
            )

    def test_unknown_is_not_known(self) -> None:
        assert not IdentityMatch(name=UNKNOWN, similarity=0.9).known

    def test_margin_reports_how_contested_a_match_is(self) -> None:
        match = IdentityMatch("a", 0.80, runner_up="b", runner_up_similarity=0.78)
        assert match.margin == pytest.approx(0.02)


class TestGallery:
    def test_an_empty_gallery_says_unknown_rather_than_failing(self) -> None:
        """A system with nobody enrolled does not recognise anyone. That is true."""
        match = Gallery().match(template(1))
        assert match.name == UNKNOWN

    def test_the_enrolled_person_matches_themselves(self) -> None:
        gallery = Gallery([enrollment("alice", 1)])
        assert gallery.match(template(1)).name == "alice"

    def test_a_stranger_is_unknown(self) -> None:
        gallery = Gallery([enrollment("alice", 1)])
        match = gallery.match(template(99))
        assert match.name == UNKNOWN
        assert match.similarity < gallery.threshold

    def test_two_enrolled_people_are_told_apart(self) -> None:
        gallery = Gallery([enrollment("alice", 1), enrollment("bob", 2)])
        assert gallery.match(template(1)).name == "alice"
        assert gallery.match(template(2)).name == "bob"

    def test_a_contested_match_is_refused(self) -> None:
        """The failure the margin exists for.

        Two templates a face scores almost equally against: naming the winner
        is how a system confidently calls one person by another's name. With
        two people enrolled a coin flip is right half the time, which looks
        like it works.
        """
        alice, bob = template(1), template(2)
        gallery = Gallery(
            [
                Enrollment("alice", tuple(float(v) for v in alice), 1, time.time()),
                Enrollment("bob", tuple(float(v) for v in bob), 1, time.time()),
            ],
            threshold=0.1,
            margin=0.2,
        )
        # Exactly between them: similar to both, clearly neither.
        match = gallery.match(blend(alice, bob, 0.5))
        assert match.name == UNKNOWN
        assert match.runner_up is not None

    def test_a_clear_winner_survives_the_margin(self) -> None:
        alice, bob = template(1), template(2)
        gallery = Gallery(
            [
                Enrollment("alice", tuple(float(v) for v in alice), 1, time.time()),
                Enrollment("bob", tuple(float(v) for v in bob), 1, time.time()),
            ],
            threshold=0.1,
            margin=0.05,
        )
        assert gallery.match(blend(alice, bob, 0.05)).name == "alice"

    def test_threshold_is_enforced(self) -> None:
        gallery = Gallery([enrollment("alice", 1)], threshold=0.99)
        assert gallery.match(blend(template(1), template(2), 0.3)).name == UNKNOWN

    def test_removal(self) -> None:
        gallery = Gallery([enrollment("alice", 1)])
        assert gallery.remove("alice") is True
        assert gallery.remove("alice") is False
        assert gallery.match(template(1)).name == UNKNOWN

    def test_an_impossible_threshold_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="threshold"):
            Gallery(threshold=5.0)

    def test_averaging_produces_a_unit_vector(self) -> None:
        averaged = average_templates([template(1), template(2), template(3)])
        assert np.linalg.norm(averaged) == pytest.approx(1.0, abs=1e-5)

    def test_averaging_needs_something_to_average(self) -> None:
        with pytest.raises(ValueError):
            average_templates([])


class TestStore:
    def test_round_trip(self, store: IdentityStore) -> None:
        original = enrollment("alice", 1)
        store.enroll(original)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].name == "alice"
        assert np.allclose(loaded[0].template, original.template, atol=1e-6)

    def test_re_enrolling_replaces_rather_than_duplicates(self, store: IdentityStore) -> None:
        store.enroll(enrollment("alice", 1, samples=2))
        store.enroll(enrollment("alice", 2, samples=9))
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].samples == 9

    def test_revocation_deletes_the_template(self, store: IdentityStore) -> None:
        """A "deleted" biometric still in the file is not deleted."""
        store.enroll(enrollment("alice", 1))
        assert store.revoke("alice") is True
        assert store.load() == []
        row = store._require().execute("SELECT COUNT(*) AS n FROM identities").fetchone()
        assert row["n"] == 0

    def test_revoking_someone_absent_is_harmless(self, store: IdentityStore) -> None:
        assert store.revoke("nobody") is False

    def test_a_malformed_row_is_skipped_not_fatal(self, store: IdentityStore) -> None:
        store.enroll(enrollment("alice", 1))
        store._require().execute(
            "INSERT INTO identities (name, template, samples, enrolled_at) VALUES (?,?,?,?)",
            ("broken", b"\x00\x01", 1, time.time()),
        )
        loaded = store.load()
        assert [e.name for e in loaded] == ["alice"]


class TestAudit:
    def test_enrolment_is_recorded(self, store: IdentityStore) -> None:
        store.enroll(enrollment("alice", 1))
        trail = store.audit_trail()
        assert trail[0].action is AuditAction.ENROLLED

    def test_revocation_is_recorded_and_survives_the_deletion(
        self, store: IdentityStore
    ) -> None:
        """Erasing the record of a deletion would defeat the point of a trail."""
        store.enroll(enrollment("alice", 1))
        store.revoke("alice")
        actions = [record.action for record in store.audit_trail()]
        assert AuditAction.REVOKED in actions
        assert AuditAction.ENROLLED in actions

    def test_rejections_are_recorded_too(self, store: IdentityStore) -> None:
        """ "Did this system look at me and decide it did not know me" needs an answer."""
        store.audit(
            AuditRecord(AuditAction.REJECTED, UNKNOWN, time.time(), "no template matched")
        )
        assert store.audit_trail()[0].action is AuditAction.REJECTED

    def test_filtering_by_name_and_time(self, store: IdentityStore) -> None:
        now = time.time()
        store.audit(AuditRecord(AuditAction.IDENTIFIED, "alice", now - 7200))
        store.audit(AuditRecord(AuditAction.IDENTIFIED, "bob", now))
        assert len(store.audit_trail(name="bob")) == 1
        assert len(store.audit_trail(since=now - 60)) == 1

    def test_records_are_serialisable(self, store: IdentityStore) -> None:
        import json

        store.enroll(enrollment("alice", 1))
        json.dumps([r.to_record() for r in store.audit_trail()])


class TestEnrolmentSafeguards:
    def test_enrolment_without_consent_is_refused(self) -> None:
        """The flag is the difference between choosing to enrol someone and
        finding out afterwards that you had."""
        from vantage.identity.enrollment import enroll_from_images

        with pytest.raises(ConfigError, match="consent"):
            enroll_from_images(None, "alice", ["x.png"], consent=False)

    def test_camera_enrolment_also_requires_consent(self) -> None:
        from vantage.identity.enrollment import enroll_from_camera

        with pytest.raises(ConfigError, match="consent"):
            enroll_from_camera(None, "alice", "synthetic://", consent=False)

    def test_a_nameless_enrolment_is_refused(self) -> None:
        from vantage.identity.enrollment import enroll_from_images

        with pytest.raises(ConfigError, match="name"):
            enroll_from_images(None, "   ", ["x.png"], consent=True)

    def test_consent_is_checked_before_anything_expensive(self) -> None:
        """It fails in a second rather than after a 40 MB download."""
        from vantage.identity.enrollment import enroll_from_images

        with pytest.raises(ConfigError, match="consent"):
            # Passing None as the recogniser proves nothing was loaded first.
            enroll_from_images(None, "alice", ["nonexistent.png"], consent=False)

    def test_there_is_no_enrolment_path_from_the_pipeline(self) -> None:
        """The central guarantee: a face cannot become a name by being seen.

        Checked structurally rather than by grepping for the word "enrol",
        which the engine legitimately uses when reporting how many people are
        enrolled. What must not exist is a *call*: the engine may read the
        gallery and write audit records, and must never add to either.
        """
        import ast
        import inspect

        from vantage.identity import engine

        source = inspect.getsource(engine)
        tree = ast.parse(source)

        # Asserted on the receiver, not the method name: the engine legitimately
        # calls set.add for its live-track bookkeeping, and a check for "add"
        # anywhere flags that too.
        mutations = {"_gallery.add", "_gallery.remove", "_store.enroll", "_store.revoke"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if isinstance(target, ast.Attribute):
                    assert f"{target.attr}.{node.func.attr}" not in mutations

        # And nothing from the enrolment module is reachable from here at all.
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
            if "enrollment" in (node.module or "")
        }
        assert not imported


class TestEngineVoting:
    def build(self, gallery: Gallery, **kwargs):
        from vantage.identity.engine import IdentityEngine, IdentityParams

        class StubRecognizer:
            """Returns a fixed template, so voting can be tested without faces."""

            def __init__(self, vector):
                self.vector = vector
                self.calls = 0

            def template_for(self, image):
                self.calls += 1
                return self.vector

        recognizer = StubRecognizer(kwargs.pop("vector", template(1)))
        engine = IdentityEngine(recognizer, gallery, params=IdentityParams(**kwargs))
        return engine, recognizer

    def frame_and_tracks(self, index: int = 0):
        from vantage.core.frame import Frame
        from vantage.perception.contracts import BoundingBox
        from vantage.tracking.contracts import Track, TrackingResult, TrackState

        frame = Frame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            index=index,
            source_id="t",
            capture_monotonic=float(index),
            capture_wall=float(index),
        )
        track = Track(
            track_id=1,
            entity_id="person_1",
            box=BoundingBox(100.0, 50.0, 300.0, 450.0),
            label="person",
            class_id=0,
            confidence=0.9,
            state=TrackState.CONFIRMED,
            age=index + 1,
            hits=index + 1,
            time_since_update=0,
            start_frame=0,
            last_frame=index,
        )
        return frame, TrackingResult(
            tracks=(track,),
            source_id="t",
            frame_index=index,
            capture_wall=float(index),
            frame_size=(640, 480),
            elapsed_s=1 / 30,
        )

    def test_a_name_is_not_committed_on_one_look(self) -> None:
        """One crop is one angle at one moment."""
        engine, _ = self.build(Gallery([enrollment("alice", 1)]), interval=1, min_votes=3)
        frame, tracking = self.frame_and_tracks(0)
        result = engine.update(frame, tracking)
        assert not result.identities[0].resolved

    def test_agreeing_observations_commit(self) -> None:
        engine, _ = self.build(Gallery([enrollment("alice", 1)]), interval=1, min_votes=3)
        for index in range(6):
            frame, tracking = self.frame_and_tracks(index)
            result = engine.update(frame, tracking)
        assert result.identities[0].resolved
        assert result.identities[0].name == "alice"

    def test_a_resolved_track_stops_costing_comparisons(self) -> None:
        """Identity does not change during a track; re-checking every frame
        would spend 41 ms on a settled question."""
        engine, recognizer = self.build(
            Gallery([enrollment("alice", 1)]), interval=1, min_votes=2, reverify_interval=0
        )
        for index in range(30):
            frame, tracking = self.frame_and_tracks(index)
            engine.update(frame, tracking)
        assert recognizer.calls < 10

    def test_an_unrecognised_person_settles_on_unknown(self) -> None:
        engine, _ = self.build(
            Gallery([enrollment("alice", 1)]), vector=template(99), interval=1, min_votes=3
        )
        for index in range(8):
            frame, tracking = self.frame_and_tracks(index)
            result = engine.update(frame, tracking)
        assert result.identities[0].name == UNKNOWN

    def test_attempts_are_bounded(self) -> None:
        """A person facing away cannot be identified, and retrying forever
        spends 41 ms an interval on an unanswerable question."""
        engine, recognizer = self.build(
            Gallery([enrollment("alice", 1)]),
            vector=template(99),
            interval=1,
            min_votes=99,
            max_attempts=5,
        )
        for index in range(60):
            frame, tracking = self.frame_and_tracks(index)
            engine.update(frame, tracking)
        assert recognizer.calls <= 6

    def test_retired_tracks_release_their_votes(self) -> None:
        """A recycled track id must not inherit a stranger's name."""
        from vantage.tracking.contracts import TrackingResult

        engine, _ = self.build(Gallery([enrollment("alice", 1)]), interval=1, min_votes=2)
        frame, tracking = self.frame_and_tracks(0)
        engine.update(frame, tracking)
        assert engine.tracked == 1
        engine.update(
            frame,
            TrackingResult(
                tracks=(), source_id="t", frame_index=1, capture_wall=1.0, frame_size=(640, 480)
            ),
        )
        assert engine.tracked == 0

    def test_non_people_are_never_identified(self) -> None:
        from vantage.perception.contracts import BoundingBox
        from vantage.tracking.contracts import Track, TrackingResult, TrackState

        engine, recognizer = self.build(Gallery([enrollment("alice", 1)]), interval=1)
        frame, _ = self.frame_and_tracks(0)
        car = Track(
            track_id=2,
            entity_id="car_2",
            box=BoundingBox(0.0, 0.0, 100.0, 100.0),
            label="car",
            class_id=2,
            confidence=0.9,
            state=TrackState.CONFIRMED,
            age=1,
            hits=1,
            time_since_update=0,
            start_frame=0,
            last_frame=0,
        )
        result = engine.update(
            frame,
            TrackingResult(
                tracks=(car,),
                source_id="t",
                frame_index=0,
                capture_wall=0.0,
                frame_size=(640, 480),
            ),
        )
        assert len(result) == 0
        assert recognizer.calls == 0


class TestConfigAndWiring:
    def test_identity_is_off_by_default(self) -> None:
        from vantage.config.schema import VantageConfig

        assert VantageConfig().identity.enabled is False

    def test_identity_requires_tracking(self) -> None:
        """It resolves an existing anonymous entity; without tracks there is none."""
        from vantage.config.schema import IdentityConfig, VantageConfig

        with pytest.raises(ConfigError, match="requires tracking"):
            VantageConfig(identity=IdentityConfig(enabled=True))

    def test_the_identify_flag_implies_tracking(self) -> None:
        from vantage.cli import _flag_overrides, build_parser

        overrides = _flag_overrides(build_parser().parse_args(["run", "--identify"]))
        assert "identity.enabled=true" in overrides
        assert "tracking.enabled=true" in overrides
        assert "detection.enabled=true" in overrides

    def test_the_observation_identity_column_is_filled(self, tmp_path: Path) -> None:
        """The seam every phase since 4 left open."""
        from vantage.identity.contracts import EntityIdentity, IdentityResult
        from vantage.state.contracts import EntityState, MotionState, StateResult
        from vantage.storage.recorder import Recorder
        from vantage.storage.sqlite_store import SqliteStore
        from vantage.storage.writer import StoreWriter

        store = SqliteStore(tmp_path / "obs.db")
        writer = StoreWriter(store, batch_size=1, flush_interval_s=0.05)
        recorder = Recorder(writer, camera_id="cam0", observation_interval=1)

        state = StateResult(
            states=(
                EntityState(
                    track_id=1,
                    entity_id="person_1",
                    label="person",
                    motion=MotionState.MOVING,
                    speed=0.5,
                    dwell_s=1.0,
                    bearing_deg=None,
                    distance=0.0,
                    age_s=1.0,
                    observed=True,
                ),
            ),
            source_id="t",
            frame_index=1,
            capture_wall=1000.0,
            elapsed_s=1 / 30,
        )
        identity = IdentityResult(
            identities=(
                EntityIdentity(
                    track_id=1,
                    entity_id="person_1",
                    name="alice",
                    similarity=0.8,
                    votes=3,
                    resolved=True,
                ),
            ),
            source_id="t",
            frame_index=1,
            capture_wall=1000.0,
        )
        recorder.record(state=state, identity=identity)
        writer.close()

        from vantage.storage.contracts import Query

        rows = store.observations(Query(limit=5))
        store.close()
        assert rows[0].identity == "alice"

    def test_identity_absent_leaves_the_column_null(self, tmp_path: Path) -> None:
        """Which is every deployment that never turns it on."""
        from vantage.state.contracts import EntityState, MotionState, StateResult
        from vantage.storage.contracts import Query
        from vantage.storage.recorder import Recorder
        from vantage.storage.sqlite_store import SqliteStore
        from vantage.storage.writer import StoreWriter

        store = SqliteStore(tmp_path / "obs.db")
        writer = StoreWriter(store, batch_size=1, flush_interval_s=0.05)
        recorder = Recorder(writer, camera_id="cam0", observation_interval=1)
        recorder.record(
            state=StateResult(
                states=(
                    EntityState(
                        track_id=1,
                        entity_id="person_1",
                        label="person",
                        motion=MotionState.STATIONARY,
                        speed=0.0,
                        dwell_s=1.0,
                        bearing_deg=None,
                        distance=0.0,
                        age_s=1.0,
                        observed=True,
                    ),
                ),
                source_id="t",
                frame_index=1,
                capture_wall=1000.0,
                elapsed_s=1 / 30,
            )
        )
        writer.close()
        rows = store.observations(Query(limit=5))
        store.close()
        assert rows[0].identity is None


class TestModels:
    """The parts that need the real weights."""

    pytestmark = pytest.mark.model

    def recognizer(self):
        from vantage.identity.recognizer import FaceRecognizer
        from vantage.perception.catalog import get_model_spec
        from vantage.perception.store import ModelStore

        store = ModelStore("models")
        specs = [get_model_spec("yunet-face"), get_model_spec("sface")]
        for spec in specs:
            if not store.is_cached(spec):
                pytest.skip(f"{spec.key} not downloaded")
        return FaceRecognizer(store.path_for(specs[0]), store.path_for(specs[1]))

    def test_both_models_are_permissively_licensed(self) -> None:
        """ArcFace weights are non-commercial research only; these are not."""
        from vantage.perception.catalog import get_model_spec

        assert get_model_spec("yunet-face").license == "MIT"
        assert get_model_spec("sface").license == "Apache-2.0"

    def test_a_blank_image_yields_no_face(self) -> None:
        recognizer = self.recognizer()
        assert recognizer.template_for(np.zeros((200, 200, 3), dtype=np.uint8)) is None

    def test_a_tiny_image_is_refused_rather_than_upscaled(self) -> None:
        """A 20-pixel face upscaled to 112x112 is a confident number from
        almost no information."""
        recognizer = self.recognizer()
        assert recognizer.detect(np.zeros((20, 20, 3), dtype=np.uint8)) is None
