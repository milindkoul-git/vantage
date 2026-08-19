"""Model storage: fetch, verify, cache.

Weights are treated as untrusted remote content that becomes part of the
system's behaviour. That leads to three rules:

1. **Verify before use, every time.** A cached file is re-hashed on load, not
   trusted because it exists. Disk corruption and half-finished downloads both
   produce a file of plausible size.
2. **Download atomically.** Content is written to a temporary file and renamed
   only after the hash matches, so an interrupted download can never leave a
   corrupt file that later looks cached.
3. **Never overwrite silently.** A hash mismatch is an error naming both
   digests, not a re-download that papers over a substituted upstream file.

Some upstream projects ship the graph inside an archive rather than as a loose
file - OpenMMLab distributes RTMPose that way. Those are pinned twice: once on
the archive as downloaded, and once on the extracted member before it is
installed. The second pin is what keeps rule 1 intact, because the file that
ends up cached is the member rather than the archive, and a cached file is
always re-hashed against the pin for the file it actually is.

Models live outside the repository (``models/`` is git-ignored) because binary
weights in git history are a permanent tax on every clone.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.perception.catalog import ModelSpec

log = get_logger(__name__)

_CHUNK = 1 << 20  # 1 MiB
_TIMEOUT_S = 120


class ModelStore:
    """A directory of verified model files."""

    def __init__(self, directory: str | os.PathLike[str] = "models") -> None:
        self._directory = Path(directory).expanduser()

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, spec: ModelSpec) -> Path:
        return self._directory / spec.filename

    def is_cached(self, spec: ModelSpec) -> bool:
        """Whether the file exists locally. Says nothing about its integrity."""
        return self.path_for(spec).is_file()

    def ensure(
        self,
        spec: ModelSpec,
        *,
        allow_download: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Return a verified local path to the model, downloading if needed.

        Raises:
            VantageError: the file is absent and downloads are disabled, the
                download failed, or the content did not match the pinned hash.
        """
        path = self.path_for(spec)

        if path.is_file():
            actual = sha256_of(path)
            if actual == spec.sha256:
                return path
            raise VantageError(
                f"cached model {path} failed integrity verification.\n"
                f"  expected sha256: {spec.sha256}\n"
                f"  actual sha256:   {actual}\n"
                "Delete the file to re-download it. If the upstream release was "
                "replaced, the catalog pin must be reviewed rather than bumped blindly."
            )

        if not allow_download:
            raise VantageError(
                f"model {spec.key!r} is not present at {path} and downloading is "
                "disabled. Run: vantage models pull " + spec.key
            )

        return self._download(spec, path, progress)

    def _install_member(self, spec: ModelSpec, archive: Path, destination: Path) -> None:
        """Extract the one pinned member of ``archive`` to ``destination``.

        Only the named member is ever read - never ``extractall`` - so a crafted
        archive cannot write outside the model directory whatever its entries
        claim to be called.
        """
        assert spec.archive_member is not None
        temporary = destination.with_suffix(destination.suffix + ".member")
        try:
            with zipfile.ZipFile(archive) as bundle:
                try:
                    source = bundle.open(spec.archive_member)
                except KeyError:
                    raise VantageError(
                        f"archive for model {spec.key!r} does not contain the expected "
                        f"member {spec.archive_member!r}. The upstream layout has "
                        f"changed, so the catalog entry must be reviewed rather than "
                        f"guessed at. Members present: {bundle.namelist()[:6]}"
                    ) from None
                with source, temporary.open("wb") as handle:
                    shutil.copyfileobj(source, handle, _CHUNK)
        except zipfile.BadZipFile as exc:
            temporary.unlink(missing_ok=True)
            raise VantageError(
                f"archive for model {spec.key!r} is not a readable zip: {exc}"
            ) from exc

        actual = sha256_of(temporary)
        if actual != spec.sha256:
            temporary.unlink(missing_ok=True)
            raise VantageError(
                f"member {spec.archive_member!r} of model {spec.key!r} does not match "
                f"its pinned checksum.\n"
                f"  expected sha256: {spec.sha256}\n"
                f"  actual sha256:   {actual}\n"
                "The archive verified but its contents did not. Nothing was installed."
            )
        shutil.move(str(temporary), str(destination))

    def _download(
        self, spec: ModelSpec, destination: Path, progress: Callable[[int, int], None] | None
    ) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        # For an archived model the bytes on the wire are the archive, so that
        # is what this transfer is checked against; the member is verified
        # separately once it has been extracted.
        expected_sha = spec.archive_sha256 or spec.sha256

        log.info(
            "downloading model",
            extra={
                "vantage_fields": {
                    "model": spec.key,
                    "url": spec.url,
                    "size_mb": round(spec.download_size_bytes / 1e6, 1),
                    "license": spec.license,
                    "archived": spec.is_archived,
                }
            },
        )

        digest = hashlib.sha256()
        downloaded = 0
        try:
            request = urllib.request.Request(
                spec.url, headers={"User-Agent": "vantage/model-store"}
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                declared = int(
                    response.headers.get("Content-Length") or spec.download_size_bytes
                )
                with temporary.open("wb") as handle:
                    while chunk := response.read(_CHUNK):
                        handle.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress(downloaded, declared)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            raise VantageError(
                f"could not download model {spec.key!r} from {spec.url}: {exc}. "
                "Check network access, or place the file manually at "
                f"{destination}."
            ) from exc

        actual = digest.hexdigest()
        if actual != expected_sha:
            temporary.unlink(missing_ok=True)
            raise VantageError(
                f"downloaded model {spec.key!r} does not match its pinned checksum.\n"
                f"  expected sha256: {expected_sha}\n"
                f"  actual sha256:   {actual}\n"
                "The remote file has changed or the download was corrupted. "
                "Nothing was installed."
            )

        if spec.is_archived:
            # The archive is scratch space; only the verified member is kept.
            try:
                self._install_member(spec, temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # Rename only after verification: a file at the final path is, by
            # construction, one that already passed its hash check.
            shutil.move(str(temporary), str(destination))
        log.info(
            "model ready",
            extra={
                "vantage_fields": {
                    "model": spec.key,
                    "path": str(destination),
                    "sha256": actual[:16] + "...",
                }
            },
        )
        return destination

    def remove(self, spec: ModelSpec) -> bool:
        """Delete a cached model. Returns whether anything was removed."""
        path = self.path_for(spec)
        if path.is_file():
            path.unlink()
            return True
        return False


def sha256_of(path: Path) -> str:
    """Streaming SHA-256, so hashing a large model never loads it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
