"""Independently verify and export a PMBS manifest evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def stats(rows):
    succeeded = [row for row in rows if row["success"]]
    successes = len(succeeded)
    count = len(rows)
    return {
        "scenes": count,
        "successes": successes,
        "success_rate": successes / count if count else None,
        "success_pct": 100.0 * successes / count if count else None,
        "avg_pushes": mean([row["pushes"] for row in rows]),
        "avg_pushes_successful": mean([row["pushes"] for row in succeeded]),
        "avg_execution_time_s": mean([row["execution_time_s"] for row in rows]),
        "avg_execution_time_successful_s": mean([row["execution_time_s"] for row in succeeded]),
        "avg_planning_time_s": mean([row["planning_time_s"] for row in rows]),
        "avg_total_time_s": mean([row["total_time_s"] for row in rows]),
        "total_pushes": sum(row["pushes"] for row in rows),
        "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
    }


def expected_summary(rows, expected):
    tiers = defaultdict(list)
    for row in rows:
        tiers[row["tier"]].append(row)
    return {
        "state": "complete" if len(rows) == expected else "running",
        "expected_scenes": expected,
        "completed_scenes": len(rows),
        "remaining_scenes": expected - len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "overall": stats(rows),
        "by_tier": {tier: stats(group) for tier, group in sorted(tiers.items())},
    }


def compare(actual, expected, path="summary"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path}: expected object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path}: missing {key}")
            compare(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, float):
        if actual is None or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{path}: {actual!r} != {expected!r}")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    manifest_path = Path(protocol["manifest"]).resolve()
    if sha256(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Manifest hash differs from frozen protocol")
    manifest = json.loads(manifest_path.read_text())
    all_scenes = manifest["scenes"]
    start, stop = protocol["selected_start"], protocol["selected_stop"]
    selected = all_scenes[start:stop]
    expected_count = len(selected)

    paths = sorted((output / "scenes").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if not args.allow_partial and len(rows) != expected_count:
        raise ValueError(f"Incomplete: {len(rows)}/{expected_count} scenes")
    if len(rows) > expected_count:
        raise ValueError(f"Too many scene records: {len(rows)}/{expected_count}")

    seen_indices = set()
    seen_hashes = set()
    for path, row in zip(paths, rows):
        index = row["index"]
        if path.name != f"{index:04d}.json":
            raise ValueError(f"Filename/index mismatch: {path}")
        if index in seen_indices or not start <= index < stop:
            raise ValueError(f"Duplicate or out-of-range index: {index}")
        seen_indices.add(index)
        entry = all_scenes[index]
        scene_path = Path(entry["path"]).resolve()
        actual_hash = sha256(scene_path)
        if actual_hash != entry["sha256"] or row["scene_sha256"] != actual_hash:
            raise ValueError(f"Scene hash mismatch at index {index}")
        if row["scene"] != str(scene_path) or row["tier"] != entry["tier"]:
            raise ValueError(f"Scene identity mismatch at index {index}")
        if actual_hash in seen_hashes:
            raise ValueError(f"Duplicate scene content at index {index}")
        seen_hashes.add(actual_hash)

        action_trace = [item for item in row["trace"] if item["type"] in ("push", "grasp")]
        trace_pushes = sum(item["type"] == "push" for item in action_trace)
        trace_grasps = sum(item["type"] == "grasp" for item in action_trace)
        if (row["pushes"], row["grasps"], row["actions"]) != (
            trace_pushes, trace_grasps, len(action_trace)
        ):
            raise ValueError(f"Trace/action count mismatch at index {index}")
        if row["actions"] > protocol["max_actions_per_scene"]:
            raise ValueError(f"Action cap exceeded at index {index}")
        for field in ("planning_time_s", "execution_time_s", "setup_time_s", "total_time_s"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid {field} at index {index}")
        if row["total_time_s"] + 1e-9 < row["execution_time_s"]:
            raise ValueError(f"Total time shorter than execution time at index {index}")
        if row["success"] != (row["reason"] == "success"):
            raise ValueError(f"Success/reason mismatch at index {index}")
        if row["success"] and not any(
            item["type"] == "grasp" and item.get("success") for item in action_trace
        ):
            raise ValueError(f"Successful scene has no successful grasp at index {index}")

    expected_indices = set(range(start, stop))
    missing = sorted(expected_indices - seen_indices)
    if not args.allow_partial and missing:
        raise ValueError(f"Missing scene indices: {missing[:10]}")

    summary = json.loads((output / "summary.json").read_text())
    recomputed = expected_summary(rows, expected_count)
    compare(summary, recomputed)
    if not args.allow_partial and summary["state"] != "complete":
        raise ValueError("Final summary is not complete")

    csv_path = output / "results.csv"
    fields = [
        "index", "scene", "scene_sha256", "tier", "success", "reason",
        "pushes", "grasps", "actions", "planning_time_s",
        "execution_time_s", "setup_time_s", "total_time_s",
    ]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    verification = {
        "valid": True,
        "complete": len(rows) == expected_count,
        "verified_scenes": len(rows),
        "expected_scenes": expected_count,
        "missing_indices": missing,
        "manifest": str(manifest_path),
        "manifest_sha256": protocol["manifest_sha256"],
        "summary_sha256": sha256(output / "summary.json"),
        "results_csv_sha256": sha256(csv_path),
        "verified_unix": time.time(),
    }
    atomic_json(output / "verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
