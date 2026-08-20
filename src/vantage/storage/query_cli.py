"""The ``vantage history`` command, kept out of cli.py.

The CLI module is already the longest file in the project. Query rendering is
self-contained and belongs with the thing it renders, so it lives here and cli.py
calls one function.
"""

from __future__ import annotations

import argparse
import json
import time

from vantage.core.errors import VantageError
from vantage.storage.contracts import Query
from vantage.storage.sqlite_store import SqliteStore

_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_duration(text: str) -> float:
    """``30m``, ``6h``, ``7d`` into seconds.

    A duration rather than a timestamp because every question anyone actually
    asks of this data is relative - "what happened in the last hour" - and
    making them compute an epoch first would be a worse interface for the sake
    of a generality nobody wants.
    """
    raw = text.strip().lower()
    if not raw:
        raise VantageError("empty duration")
    problem = (
        f"unknown duration {text!r}. Use a number and one of s, m, h, d, w - "
        "for example 30m, 6h, 7d."
    )
    unit = raw[-1]
    if unit not in _UNITS:
        raise VantageError(problem)
    try:
        amount = float(raw[:-1])
    except ValueError:
        # "5 fortnights" ends in 's', a valid unit, so it reaches here rather
        # than the branch above. Same message either way: the caller's mistake
        # is the same and so is the fix.
        raise VantageError(problem) from None
    if amount < 0:
        raise VantageError(f"duration {text!r} must not be negative")
    return amount * _UNITS[unit]


def run(args: argparse.Namespace, default_path: str) -> int:
    """Execute one ``vantage history`` action."""
    path = args.db or default_path
    store = SqliteStore(path, read_only=True)
    try:
        return _dispatch(args, store)
    finally:
        store.close()


def _dispatch(args: argparse.Namespace, store: SqliteStore) -> int:
    if args.action == "stats":
        return _stats(args, store)
    if args.action == "prune":
        return _prune(args, store)

    since = time.time() - parse_duration(args.since) if args.since else None
    query = Query(
        since=since,
        entity_id=args.entity,
        rule=args.rule,
        severity=args.severity,
        zone=args.zone,
        limit=args.limit,
        newest_first=args.action != "timeline",
    )

    if args.action == "observations":
        return _observations(args, store, query)
    return _events(args, store, query)


def _events(args: argparse.Namespace, store: SqliteStore, query: Query) -> int:
    rows = store.events(query)
    if args.json:
        print(json.dumps([_event_dict(row) for row in rows], indent=2))
        return 0
    if not rows:
        print("No events matched." + _hint(args))
        return 0
    label = "Timeline" if args.action == "timeline" else "Events"
    print(
        f"{label} ({len(rows)} shown, newest {'last' if query.newest_first is False else 'first'})\n"
    )
    for row in rows:
        print(f"  {row.describe()}")
        if row.entity_id and args.action != "timeline":
            print(f"  {'':21s}{row.entity_id}  [{row.rule}]")
    return 0


def _observations(args: argparse.Namespace, store: SqliteStore, query: Query) -> int:
    rows = store.observations(query)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "timestamp": row.timestamp,
                        "entity_id": row.entity_id,
                        "identity": row.identity,
                        "entity_type": row.entity_type,
                        "motion": row.motion,
                        "speed": row.speed,
                        "posture": row.posture,
                        "zones": row.zones,
                        "activities": row.activities,
                    }
                    for row in rows
                ],
                indent=2,
            )
        )
        return 0
    if not rows:
        print("No observations matched." + _hint(args))
        return 0
    print(f"Observations ({len(rows)} shown)\n")
    print(
        f"  {'TIME':10s} {'ENTITY':14s} {'TYPE':9s} {'MOTION':11s} {'POSTURE':10s} WHERE / DOING"
    )
    for row in rows:
        where = " ".join(p for p in (row.zones, row.activities) if p)
        print(
            f"  {row.when.strftime('%H:%M:%S'):10s} {row.entity_id:14s} "
            f"{row.entity_type:9s} {row.motion or '-':11s} {row.posture or '-':10s} {where}"
        )
    return 0


def _stats(args: argparse.Namespace, store: SqliteStore) -> int:
    counts = store.counts()
    if args.json:
        print(json.dumps({**counts, "path": str(store.path)}, indent=2))
        return 0
    print(f"Store: {store.path} (schema v{store.schema_version})\n")
    print(f"  events        {counts.get('events', 0):>12,}")
    print(f"  observations  {counts.get('observations', 0):>12,}")
    if "span_s" in counts:
        hours = counts["span_s"] / 3600.0
        print(f"  covering      {hours:>12.1f} hours")
    print(f"  on disk       {counts.get('bytes', 0) / 1e6:>12.1f} MB")
    if counts.get("observations"):
        per_row = counts.get("bytes", 0) / max(1, counts["observations"] + counts["events"])
        print(f"  per row       {per_row:>12.0f} bytes")
    return 0


def _prune(args: argparse.Namespace, store: SqliteStore) -> int:
    if not args.older_than:
        raise VantageError(
            "prune needs a horizon: vantage history prune --older-than 30d. "
            "Refusing to guess, because the wrong guess deletes data."
        )
    cutoff = time.time() - parse_duration(args.older_than)
    removed = store.prune(cutoff)
    total = sum(removed.values())
    if args.json:
        print(json.dumps({"removed": removed, "cutoff": cutoff}, indent=2))
        return 0
    print(f"Removed {total:,} rows older than {args.older_than}:")
    for table, count in sorted(removed.items()):
        print(f"  {table:14s} {count:>10,}")
    if total:
        # VACUUM is not run automatically: it rewrites the entire file and holds
        # a lock for as long as that takes.
        print("\n  Space is reclaimed on the next VACUUM; SQLite reuses the pages meanwhile.")
    return 0


def _event_dict(row) -> dict:
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "camera_id": row.camera_id,
        "rule": row.rule,
        "severity": row.severity,
        "summary": row.summary,
        "entity_id": row.entity_id,
        "identity": row.identity,
        "related_id": row.related_id,
        "zone": row.zone,
        "evidence": row.evidence,
    }


def _hint(args: argparse.Namespace) -> str:
    filters = [
        name
        for name, value in (
            ("--since", args.since),
            ("--entity", args.entity),
            ("--rule", args.rule),
            ("--severity", args.severity),
            ("--zone", args.zone),
        )
        if value
    ]
    if filters:
        return f" (filters applied: {', '.join(filters)})"
    return " The store is empty; runs only write one with --store."
