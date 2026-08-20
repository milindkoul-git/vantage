"""``vantage identity`` - enrol, list, revoke, audit, verify.

Kept out of cli.py, which is already the longest file here. Every command in
this module is an administrative act performed by a person at the machine, which
is the whole security model: there is no remote interface to any of it, and the
dashboard is read-only by construction.
"""

from __future__ import annotations

import argparse
import json
import time

from vantage.core.errors import ConfigError, VantageError
from vantage.identity.contracts import UNKNOWN
from vantage.identity.gallery import Gallery
from vantage.identity.store import IdentityStore

CONSENT_HELP = (
    "affirm that the person being enrolled knows about it and agreed to it. "
    "Required: this system will not add a face to its gallery on anyone's behalf"
)


def run(args: argparse.Namespace, config) -> int:
    """Execute one ``vantage identity`` action."""
    settings = config.identity
    path = args.db or settings.path

    if args.action == "enroll":
        return _enroll(args, settings, path)

    store = IdentityStore(path)
    try:
        if args.action == "list":
            return _list(args, store, settings)
        if args.action == "forget":
            return _forget(args, store)
        if args.action == "audit":
            return _audit(args, store)
        if args.action == "verify":
            return _verify(args, store, settings)
        raise VantageError(f"unknown identity action {args.action!r}")
    finally:
        store.close()


def _enroll(args: argparse.Namespace, settings, path: str) -> int:
    from vantage.identity.enrollment import enroll_from_camera, enroll_from_images
    from vantage.identity.factory import build_recognizer

    if not args.name:
        raise ConfigError("enrolment needs --name")
    # Checked before the models load, so a missing flag fails in a second rather
    # than after a 40 MB download and a model init.
    if not args.consent:
        from vantage.identity.enrollment import CONSENT_REQUIRED

        raise ConfigError(CONSENT_REQUIRED)

    recognizer = build_recognizer(settings)
    if args.image:
        enrollment = enroll_from_images(
            recognizer, args.name, list(args.image), consent=True, note=args.note
        )
    else:
        print(f"Capturing {args.samples} face samples for {args.name!r} from {args.source}.")
        print("  Look at the camera and move your head a little between captures.")

        def progress(done: int, total: int) -> None:
            print(f"  captured {done}/{total}")

        enrollment = enroll_from_camera(
            recognizer,
            args.name,
            args.source,
            consent=True,
            samples=args.samples,
            note=args.note,
            progress=progress,
        )

    store = IdentityStore(path)
    try:
        existed = args.name in set(store.names())
        store.enroll(enrollment)
    finally:
        store.close()

    verb = "Re-enrolled" if existed else "Enrolled"
    print(f"\n{verb} {enrollment.name!r} from {enrollment.samples} samples.")
    print(f"  Stored in {path} as a {len(enrollment.template)}-dimensional template.")
    print("  No image was written to disk. Remove with: vantage identity forget --name ...")
    return 0


def _list(args: argparse.Namespace, store: IdentityStore, settings) -> int:
    enrollments = store.load()
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(store.path),
                    "threshold": settings.threshold,
                    "identities": [
                        {
                            "name": e.name,
                            "samples": e.samples,
                            "enrolled_at": e.enrolled_at,
                            "note": e.note,
                        }
                        for e in enrollments
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"Enrolled identities ({store.path})\n")
    if not enrollments:
        print("  Nobody is enrolled. Every face will be reported as unknown.")
        print("\n  Enrol with: vantage identity enroll --name NAME --consent")
        return 0
    for enrollment in enrollments:
        print(f"  {enrollment.describe()}")
    print(f"\n  Matching threshold {settings.threshold}, margin {settings.margin}.")
    print("  Templates only - no images are stored.")
    return 0


def _forget(args: argparse.Namespace, store: IdentityStore) -> int:
    if not args.name:
        raise ConfigError("forget needs --name")
    removed = store.revoke(args.name)
    if not removed:
        print(f"No enrolment named {args.name!r}. Nothing was removed.")
        return 0
    print(f"Removed {args.name!r}. The template is deleted from {store.path}.")
    # The audit entry recording the deletion stays: erasing the record of a
    # revocation would defeat the point of keeping a trail at all.
    print("  The audit record of this revocation is kept.")
    return 0


def _audit(args: argparse.Namespace, store: IdentityStore) -> int:
    from vantage.storage.query_cli import parse_duration

    since = time.time() - parse_duration(args.since) if args.since else None
    records = store.audit_trail(since=since, name=args.name, limit=args.limit)
    if args.json:
        print(json.dumps([r.to_record() for r in records], indent=2))
        return 0
    if not records:
        print("No audit records matched.")
        return 0
    print(f"Identity audit ({len(records)} shown, newest first)\n")
    for record in records:
        print(f"  {record.describe()}")
    return 0


def _verify(args: argparse.Namespace, store: IdentityStore, settings) -> int:
    """Compare an image against the gallery without changing anything.

    Exists so the threshold can be checked against real faces before it is
    trusted in a live run - and so "would this system recognise me" can be
    answered without being enrolled.
    """
    import cv2

    from vantage.identity.factory import build_recognizer

    if not args.image:
        raise ConfigError("verify needs --image")
    recognizer = build_recognizer(settings)
    gallery = Gallery(store.load(), threshold=settings.threshold, margin=settings.margin)

    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            raise VantageError(f"could not read {path}")
        template = recognizer.template_for(image)
        if template is None:
            print(f"  {path}: no usable face found")
            continue
        match = gallery.match(template)
        verdict = match.name if match.known else UNKNOWN
        print(f"  {path}: {verdict}  (best {match.similarity:.3f}, margin {match.margin:.3f})")
    if not len(gallery):
        print("\n  Nobody is enrolled, so everything is unknown by definition.")
    return 0
