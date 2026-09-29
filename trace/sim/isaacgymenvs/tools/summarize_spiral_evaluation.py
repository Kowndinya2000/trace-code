"""Validate and summarize sharded Cartesian-baseline evaluation outputs."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


PROTOCOLS = {
    "spiral": "closed-loop-spiral-sim-v1",
    "straightline": "closed-loop-target-oriented-straightline-sim-v2",
}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def outcome(rows):
    reasons = Counter(row["reason"] for row in rows)
    successes = reasons["success"]
    return {
        "scenes": len(rows),
        "successes": successes,
        "success_pct": 100.0 * successes / len(rows) if rows else math.nan,
        "reasons": dict(sorted(reasons.items())),
    }


def write_csv(path, rows):
    fields = [
        "manifest_index", "scene_file", "sha256", "tier", "success", "reason",
        "terminal_step", "terminal_q", "max_q", "initial_q", "travelled_m",
        "sim_solve_time_s", "batch_wall_time_to_terminal_s", "grasp_evaluations",
        "physics_ticks", "waypoint_clips", "terminal_radius_m",
        "terminal_orbit_progress_rad", "terminal_path_progress_m",
        "straightline_direction_x", "straightline_direction_y", "radial_entry_steps",
        "terminal_eef_x", "terminal_eef_y",
        "terminal_target_x", "terminal_target_y",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_report(path, summary, failures, rows):
    baseline = summary["baseline"]
    overall = summary["overall"]
    timing = summary["timing"]
    sim_all = timing["sim_solve_time_s_all"]
    sim_ok = timing["sim_solve_time_s_success"]
    sim_fail = timing["sim_solve_time_s_failure"]
    total_wall = timing["sum_batch_setup_wall_s"] + timing["sum_batch_rollout_wall_s"]
    simultaneous = [
        row for row in failures
        if row.get("terminal_q") is not None
        and row["terminal_q"] > summary["params"]["grasp_threshold"]
    ]
    target_text = (
        "the current simulator ground-truth target pose at every decision"
        if baseline == "spiral"
        else "the current simulator ground-truth target pose at every decision"
    )
    parameter_rows = [
        ("Grasp-classifier threshold", f"> {summary['params']['grasp_threshold']:.2f}"),
        ("Timeout", f"{summary['params']['timeout_s']:.0f} s"),
        ("Maximum control steps", str(summary['params']['max_steps'])),
        ("Measured-motion fault limit", f"{summary['params']['max_travel_m']:.1f} m"),
    ]
    if baseline == "spiral":
        parameter_rows += [
            ("Arc step", f"{1000*summary['params']['arc_step_m']:.1f} mm"),
            ("Radial step", f"{1000*summary['params']['radial_step_m']:.1f} mm"),
            ("Minimum radius", f"{100*summary['params']['min_radius_m']:.1f} cm"),
            ("Final orbits", f"{summary['params']['orbit_loops']:g}"),
        ]
        if summary["params"].get("start_radius_m") is not None:
            parameter_rows += [
                ("Bounded start radius",
                 f"{100*summary['params']['start_radius_m']:.1f} cm"),
                ("Entry", "radial, in arc-step increments"),
            ]
    else:
        parameter_rows += [
            ("Straight-line distance", f"{100*summary['params']['distance_m']:.1f} cm"),
            ("Command step", f"{100*summary['params']['step_m']:.1f} cm"),
            ("Heading", "recomputed current EEF-to-target vector after every command"),
        ]
    reason_meanings = {
        "clutter_oow": "A clutter object's collision mesh left the canonical workspace.",
        "target_oow": "The target object's collision mesh left the canonical workspace.",
        "scene_oow_transient": "A collision mesh crossed the boundary during a command and was back inside at the endpoint.",
        "out_of_view": "The grasp network reported that the target was out of view.",
        "invalid_state": "The simulator or grasp signal became invalid.",
        "motion_stall": "The Cartesian command did not reach its tolerance before the replay limit.",
        "timeout": "Active simulator time reached the timeout.",
        "travel_limit": "Measured planar motion reached its safety limit.",
        "step_limit": "The controller reached its configured step limit.",
        "spiral_complete": "The spiral completed before the grasp threshold was reached.",
        "straightline_complete": "The full straight-line distance completed before the grasp threshold was reached.",
    }

    def timing_row(label, values):
        if values is None:
            return f"| {label} | — | — | — | — | — |"
        return (f"| {label} | {values['mean']:.3f} s | {values['median']:.3f} s | "
                f"{values['p90']:.3f} s | {values['p95']:.3f} s | {values['max']:.3f} s |")

    lines = [
        f"# {baseline.capitalize()} baseline: {overall['scenes']}-scene simulation evaluation",
        "",
        f"**Success: {overall['successes']}/{overall['scenes']} "
        f"({overall['success_pct']:.2f}%).**",
        "",
        f"The rollout uses {target_text} and the stock rendered "
        "depth/segmentation x16 grasp classifier. There is no occlusion injection, "
        "target re-sensing, or retract behavior.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {value} |" for name, value in parameter_rows]
    lines += [
        "",
        "## Outcomes",
        "",
        "| Split | Scenes | Successes | Rate |",
        "|---|---:|---:|---:|",
        f"| All | {overall['scenes']} | {overall['successes']} | {overall['success_pct']:.2f}% |",
    ]
    for tier, result in summary["by_tier"].items():
        lines.append(f"| {tier} | {result['scenes']} | {result['successes']} | "
                     f"{result['success_pct']:.2f}% |")
    lines += [
        "",
        "## Failure types",
        "",
        "| Type | Count | Meaning |",
        "|---|---:|---|",
    ]
    failure_reasons = {key: value for key, value in overall["reasons"].items()
                       if key != "success"}
    if failure_reasons:
        for reason, count in failure_reasons.items():
            lines.append(f"| `{reason}` | {count} | {reason_meanings.get(reason, 'Terminal failure.')} |")
        if any(reason in failure_reasons for reason in
               ("clutter_oow", "target_oow", "scene_oow_transient")):
            lines += [
                "",
                "Workspace containment takes precedence over a simultaneous classifier success; "
                f"{len(simultaneous)} workspace-exit cases had terminal Q above the threshold.",
            ]
    else:
        lines += ["| None | 0 | Every scene reached the grasp threshold. |"]
    tier_failure_text = ", ".join(
        f"{result['scenes'] - result['successes']}/{result['scenes']} {tier}"
        for tier, result in summary["by_tier"].items()
    )
    lines += [
        "",
        f"Failure counts by difficulty: {tier_failure_text}.",
        "",
        "## Solving times",
        "",
        "| Outcome | Mean sim time | Median | P90 | P95 | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
        timing_row("All", sim_all),
        timing_row("Success", sim_ok),
        timing_row("Failure", sim_fail),
        "",
        f"The median terminal decision was {timing['terminal_steps_all']['median']:.0f}; "
        f"the maximum was {timing['terminal_steps_all']['max']:.0f}. The "
        f"{timing['batch_count']} GPU "
        f"shards used {timing['sum_batch_rollout_wall_s']:.1f} s of cumulative rollout "
        f"wall time and {total_wall:.1f} s including simulator setup. Per-scene batch "
        "wall times are recorded, but simulator time is the batch-independent solving-time measure.",
        "",
        "## Failure cases",
        "",
    ]
    for reason in sorted({row["reason"] for row in failures}):
        group = [row for row in failures if row["reason"] == reason]
        ids = ", ".join(f"`{row['scene_file']}`" for row in group)
        lines += [f"### {reason} ({len(group)})", "", ids, ""]
    lines += [
        "## Artifacts",
        "",
        "- `summary.json`: aggregate outcome and timing distributions",
        "- `per_scene.csv` / `per_scene.json`: all scene outcomes and solving times",
        "- `failures.csv` / `failures.json`: failure-only records",
        "- `batch_*/trace.npz`: per-decision Q, EEF, target, waypoint, path progress, travel, and physics timing",
        "- `batch_*/provenance.json`: manifest, configuration, model, and source hashes",
        "",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=511)
    args = parser.parse_args()
    root = args.root.resolve()
    result_paths = sorted(root.glob("batch_*/result.json"))
    if not result_paths:
        raise SystemExit(f"No batch_*/result.json files under {root}")

    batches = [json.loads(path.read_text()) for path in result_paths]
    baseline = batches[0].get("baseline", "spiral")
    protocol = PROTOCOLS.get(baseline)
    if protocol is None or any(
            not batch.get("complete") or batch.get("protocol") != protocol
            or batch.get("baseline", "spiral") != baseline for batch in batches):
        raise SystemExit("Incomplete or incompatible batch")
    params = batches[0]["params"]
    if any(batch["params"] != params for batch in batches[1:]):
        raise SystemExit("Baseline parameters differ across batches")

    provenance = [json.loads((path.parent / "provenance.json").read_text())
                  for path in result_paths]
    manifest_hash = provenance[0]["manifest_sha256"]
    source_hashes = provenance[0]["source_sha256"]
    if any(p["manifest_sha256"] != manifest_hash for p in provenance[1:]):
        raise SystemExit("Manifest differs across batches")
    if any(p["source_sha256"] != source_hashes for p in provenance[1:]):
        raise SystemExit("Evaluation source changed across batches")

    rows = sorted((row for batch in batches for row in batch["rows"]),
                  key=lambda row: row["manifest_index"])
    indices = [row["manifest_index"] for row in rows]
    expected = list(range(args.expected_scenes))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        duplicate_count = len(indices) - len(set(indices))
        raise SystemExit(f"Manifest coverage mismatch: missing={missing} duplicates={duplicate_count}")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise SystemExit("Duplicate scene hashes in aggregate")

    failures = [row for row in rows if not row["success"]]
    tiers = defaultdict(list)
    for row in rows:
        tiers[row["tier"]].append(row)
    by_reason = defaultdict(list)
    for row in failures:
        by_reason[row["reason"]].append(row)

    summary = {
        "complete": True,
        "protocol": protocol,
        "baseline": baseline,
        "full_511_scene_set": args.expected_scenes == 511,
        "manifest": provenance[0]["manifest"],
        "manifest_sha256": manifest_hash,
        "params": params,
        "occlusion_model": False,
        "target_pose_source": batches[0]["target_pose_source"],
        "grasp_signal": "rendered depth+segmentation through simulator stock x16 grasp classifier",
        "timeout_clock": "per-scene active simulator physics time",
        "overall": outcome(rows),
        "by_tier": {tier: outcome(group) for tier, group in sorted(tiers.items())},
        "failure_types": {reason: outcome(group) for reason, group in sorted(by_reason.items())},
        "timing": {
            "sim_solve_time_s_all": stats([row["sim_solve_time_s"] for row in rows]),
            "sim_solve_time_s_success": stats([row["sim_solve_time_s"] for row in rows
                                                if row["success"]]),
            "sim_solve_time_s_failure": stats([row["sim_solve_time_s"] for row in failures]),
            "terminal_steps_all": stats([row["terminal_step"] for row in rows]),
            "terminal_steps_success": stats([row["terminal_step"] for row in rows
                                              if row["success"]]),
            "travelled_m_all": stats([row["travelled_m"] for row in rows]),
            "sum_batch_setup_wall_s": sum(batch["setup_wall_s"] for batch in batches),
            "sum_batch_rollout_wall_s": sum(batch["rollout_wall_s"] for batch in batches),
            "batch_count": len(batches),
            "wall_time_note": "Per-scene wall values are time-to-terminal inside a GPU batch; simulator time is batch-independent.",
        },
        "batch_results": [str(path.parent) for path in result_paths],
        "artifacts": {
            "report": str(root / "REPORT.md"),
            "per_scene_json": str(root / "per_scene.json"),
            "per_scene_csv": str(root / "per_scene.csv"),
            "failures_json": str(root / "failures.json"),
            "failures_csv": str(root / "failures.csv"),
        },
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (root / "per_scene.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    write_csv(root / "per_scene.csv", rows)
    write_csv(root / "failures.csv", failures)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(root / "REPORT.md", summary, failures, rows)
    (root / "complete.json").write_text(json.dumps({
        "complete": True,
        "protocol": protocol,
        "baseline": baseline,
        "scenes": len(rows),
        "successes": summary["overall"]["successes"],
        "success_pct": summary["overall"]["success_pct"],
    }, indent=2) + "\n")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
