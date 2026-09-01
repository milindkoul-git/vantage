"""Run the whole pipeline over real video and report what it did.

Exploratory, and deliberately not a build gate. `vantage track eval`, `activity
eval` and `spatial eval` are the gates: they score against seeded ground truth
and exit non-zero on a regression. Ordinary footage has no labels, so nothing
here scores accuracy - what it measures is everything that is checkable *without*
labels, which turns out to be most of what goes wrong.

    python tools/clip_report.py samples/*.webm --frames 600 --out /tmp/report

What it counts, and why each one is worth counting:

* **Throughput and stage health** - the parts a scenario cannot exercise: real
  containers, real resolutions, real decode failures.
* **Distinct entities against concurrent ones.** A fixed camera watching five
  people for a minute that reports two hundred identities has not seen two
  hundred people; it has lost and re-acquired the same few. Everything keyed on a
  stable identity - incidents, associations, the analytics baseline - inherits
  that.
* **What each activity was asserted about.** Grouping activity rows by the
  detected class is what surfaced `potted plant_2 is running`: 73% of everything
  the engine reported was about an object that cannot walk.
* **Which rules fired.** On footage where nothing happens, every event is a false
  positive, and that is a usable measurement without a single label.

The README records what a five-clip run of this found and what it changed. The
clips themselves are not in the repository - they are Creative Commons files from
Wikimedia Commons, listed there by title and licence.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from pathlib import Path

from vantage.app import run_ingestion
from vantage.config.schema import (
    ActivityConfig,
    DashboardConfig,
    DetectionConfig,
    DisplayConfig,
    IngestConfig,
    PoseConfig,
    RelationshipsConfig,
    SourceConfig,
    StateConfig,
    StorageConfig,
    TrackingConfig,
    VantageConfig,
)


def analyse_store(path: Path) -> dict:
    """What the run actually recorded, read back from the database."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT entity_id, entity_type, posture, motion, activities FROM observations"
        ).fetchall()
        events = connection.execute("SELECT rule, severity FROM events").fetchall()
    finally:
        connection.close()

    per_entity: collections.Counter[str] = collections.Counter()
    labels: collections.Counter[str] = collections.Counter()
    postures: collections.Counter[str] = collections.Counter()
    motions: collections.Counter[str] = collections.Counter()
    activities: collections.Counter[str] = collections.Counter()
    for entity_id, label, posture, motion, acts in rows:
        per_entity[entity_id] += 1
        labels[label] += 1
        postures[posture or "none"] += 1
        motions[motion or "none"] += 1
        for act in (acts or "").split(","):
            if act.strip():
                activities[act.strip()] += 1

    lifetimes = sorted(per_entity.values())
    return {
        "observations": len(rows),
        "distinct_entities": len(per_entity),
        "obs_per_entity_median": lifetimes[len(lifetimes) // 2] if lifetimes else 0,
        "entities_seen_once": sum(1 for n in lifetimes if n == 1),
        "labels": dict(labels.most_common(6)),
        "postures": dict(postures.most_common(6)),
        "motions": dict(motions.most_common(6)),
        "activities": dict(activities.most_common(8)),
        "events_by_rule": dict(collections.Counter(rule for rule, _ in events).most_common()),
        "events_by_severity": dict(collections.Counter(sev for _, sev in events).most_common()),
    }


def bench(clip: Path, frames: int, model: str, out_dir: Path) -> dict:
    store_path = out_dir / f"{clip.stem}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(store_path) + suffix).unlink(missing_ok=True)

    config = VantageConfig(
        source=SourceConfig(uri=str(clip), id=clip.stem),
        ingest=IngestConfig(max_frames=frames),
        detection=DetectionConfig(enabled=True, model=model),
        tracking=TrackingConfig(enabled=True),
        state=StateConfig(enabled=True),
        pose=PoseConfig(enabled=True),
        activity=ActivityConfig(enabled=True),
        storage=StorageConfig(enabled=True, path=str(store_path)),
        dashboard=DashboardConfig(enabled=False),
        display=DisplayConfig(enabled=False),
        relationships=RelationshipsConfig(enabled=True),
    )

    started = time.perf_counter()
    result = run_ingestion(config)
    wall = time.perf_counter() - started

    return {
        "clip": clip.name,
        "wall_s": round(wall, 1),
        "frames": result.frames,
        "mean_fps": round(result.mean_fps, 1),
        "dropped": result.dropped,
        "reason": result.reason,
        "detection": result.detection_summary,
        "tracking": result.tracking_summary,
        "pose": result.pose_summary,
        "state": result.state_summary,
        "activity": result.activity_summary,
        "spatial": result.spatial_summary,
        "events_raised": result.events_raised,
        "events": result.events_summary,
        "incidents": result.incidents_summary,
        "relationships": result.relationships_summary,
        "adaptive": result.adaptive,
        "stages": {
            name: {"calls": s["calls"], "failures": s["failures"], "disabled": s["disabled"]}
            for name, s in result.stage_health.items()
        },
        "store": analyse_store(store_path),
        "summary": result.summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--model", default="yolox-tiny")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for name in args.clips:
        clip = Path(name)
        print(f"\n=== {clip.name} ===", flush=True)
        try:
            report = bench(clip, args.frames, args.model, out_dir)
        except Exception as exc:  # exploratory harness: one bad clip must not end the run
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            reports.append({"clip": clip.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(report["summary"], flush=True)
        reports.append(report)

    (out_dir / "report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
