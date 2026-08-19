"""The model store, including the archive path added for RTMPose.

No network: archives are built in a temporary directory and fetched over
``file://``, which exercises the real download, verify, extract and install
code rather than a mock of it.

The archive path carries two pins - one on the transferred bytes, one on the
extracted member - and each test below removes exactly one guarantee to check
that the failure is loud and that nothing is left installed. A store that
half-installs a bad model is worse than one that cannot download at all,
because the corruption is then indistinguishable from a working cache.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from vantage.core.errors import VantageError
from vantage.perception.catalog import ModelSpec
from vantage.perception.store import ModelStore, sha256_of

MEMBER = "bundle/nested/model.onnx"
PAYLOAD = b"not really an onnx graph, but bytes are bytes" * 40


def build_archive(directory: Path, member: str = MEMBER, payload: bytes = PAYLOAD) -> Path:
    path = directory / "release.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("bundle/readme.txt", "ignored")
        bundle.writestr(member, payload)
    return path


def archived_spec(archive: Path, **overrides) -> ModelSpec:
    defaults = dict(
        key="fake-pose",
        filename="fake_pose.onnx",
        url=archive.as_uri(),
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        size_bytes=len(PAYLOAD),
        archive_member=MEMBER,
        archive_sha256=sha256_of(archive),
        archive_size_bytes=archive.stat().st_size,
        adapter="rtmpose",
        input_size=(256, 192),
        label_set="coco-keypoints",
        license="Apache-2.0",
        source="https://example.invalid",
        description="test fixture",
        task="pose",
    )
    defaults.update(overrides)
    return ModelSpec(**defaults)


class TestArchiveInstall:
    def test_member_is_extracted_and_cached(self, tmp_path: Path) -> None:
        spec = archived_spec(build_archive(tmp_path))
        store = ModelStore(tmp_path / "models")

        path = store.ensure(spec)
        assert path.read_bytes() == PAYLOAD
        assert path.name == "fake_pose.onnx"

    def test_the_archive_is_not_left_behind(self, tmp_path: Path) -> None:
        """Only the verified member is kept; the zip was scratch space."""
        spec = archived_spec(build_archive(tmp_path))
        store = ModelStore(tmp_path / "models")
        store.ensure(spec)

        leftovers = [p.name for p in (tmp_path / "models").iterdir()]
        assert leftovers == ["fake_pose.onnx"]

    def test_cached_member_is_reverified_against_its_own_pin(self, tmp_path: Path) -> None:
        """Rule 1 has to survive the archive path.

        The cached file is the member, so it must be checked against the
        member's digest - not the archive's, which is not the file on disk.
        """
        spec = archived_spec(build_archive(tmp_path))
        store = ModelStore(tmp_path / "models")
        path = store.ensure(spec)

        path.write_bytes(b"corrupted on disk")
        with pytest.raises(VantageError, match="integrity verification"):
            store.ensure(spec)

    def test_download_is_skipped_when_the_member_is_already_present(
        self, tmp_path: Path
    ) -> None:
        archive = build_archive(tmp_path)
        spec = archived_spec(archive)
        store = ModelStore(tmp_path / "models")
        store.ensure(spec)

        archive.unlink()  # the remote is gone; the cache must still serve
        assert store.ensure(spec).read_bytes() == PAYLOAD


class TestArchiveFailures:
    def test_wrong_archive_digest_installs_nothing(self, tmp_path: Path) -> None:
        spec = archived_spec(build_archive(tmp_path), archive_sha256="00" * 32)
        store = ModelStore(tmp_path / "models")

        with pytest.raises(VantageError, match="pinned checksum"):
            store.ensure(spec)
        assert not store.path_for(spec).exists()
        assert not list((tmp_path / "models").glob("*.partial"))

    def test_wrong_member_digest_installs_nothing(self, tmp_path: Path) -> None:
        """The archive can verify while its contents do not - a rebuilt release."""
        spec = archived_spec(build_archive(tmp_path), sha256="11" * 32)
        store = ModelStore(tmp_path / "models")

        with pytest.raises(VantageError, match="archive verified but its contents"):
            store.ensure(spec)
        assert not store.path_for(spec).exists()
        assert not list((tmp_path / "models").glob("*.member"))

    def test_missing_member_names_what_was_found(self, tmp_path: Path) -> None:
        """An upstream layout change must be diagnosable without guessing."""
        spec = archived_spec(build_archive(tmp_path), archive_member="bundle/moved.onnx")
        store = ModelStore(tmp_path / "models")

        with pytest.raises(VantageError, match="does not contain the expected member"):
            store.ensure(spec)

    def test_a_non_zip_is_reported_as_such(self, tmp_path: Path) -> None:
        junk = tmp_path / "release.zip"
        junk.write_bytes(b"this is not a zip file at all")
        spec = archived_spec(
            junk,
            archive_sha256=sha256_of(junk),
            archive_size_bytes=junk.stat().st_size,
        )
        store = ModelStore(tmp_path / "models")

        with pytest.raises(VantageError, match="not a readable zip"):
            store.ensure(spec)


class TestPlainInstall:
    """The pre-existing single-file path, unchanged by the archive support."""

    def test_loose_file_is_downloaded_and_verified(self, tmp_path: Path) -> None:
        blob = tmp_path / "model.onnx"
        blob.write_bytes(PAYLOAD)
        spec = ModelSpec(
            key="fake-detector",
            filename="fake.onnx",
            url=blob.as_uri(),
            sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            size_bytes=len(PAYLOAD),
            adapter="yolox",
            input_size=(416, 416),
            label_set="coco80",
            license="Apache-2.0",
            source="https://example.invalid",
            description="test fixture",
        )
        store = ModelStore(tmp_path / "models")
        assert store.ensure(spec).read_bytes() == PAYLOAD

    def test_download_disabled_names_the_pull_command(self, tmp_path: Path) -> None:
        spec = archived_spec(build_archive(tmp_path))
        store = ModelStore(tmp_path / "models")
        with pytest.raises(VantageError, match="vantage models pull"):
            store.ensure(spec, allow_download=False)
