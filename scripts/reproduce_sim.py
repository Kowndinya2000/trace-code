#!/usr/bin/env python3
"""Reproduce the simulation tables of the paper.

    python scripts/reproduce_sim.py main         # plan-conditioned policies + privileged teacher
    python scripts/reproduce_sim.py baselines    # PMBS, serial MCTS, spiral, straight line
    python scripts/reproduce_sim.py dagger       # DAgger rounds and the matched-label controls
    python scripts/reproduce_sim.py ablations    # plan horizon and recurrent memory
    python scripts/reproduce_sim.py all --gpu 0

Every table is evaluated on the same 511 development scenes with one perturbation draw per
evaluation seed, arm occlusion from the robot's own links, 10% detection dropout and the
teacher-relative execution budget. Results are written to $TRACE_RUNS/<table>/RESULTS.md
next to the raw per-scene records, so a number in the paper can be traced to an episode.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace.common import paths  # noqa: E402

paths.ensure_runtime()

SIM = paths.SIM

# The execution budget every learned policy is held to: the plan's length plus twenty decisions,
# and its nominal travel plus 15 cm. Exhausting either counts as a failure.
BUDGET = dict(protocol="teacher-relative-action-budget-v1", extra_steps=20, extra_length_m=.15,
              primitive_length_m=.04, risk_start=.70, risk_temperature=.10, max_imitation_weight=3.,
              termination="first of step or planar TCP-travel cap",
              success="simulator graspability only; budget exhaustion is timeout")
# Observation conditions shared by every row of the table.
CONDITION = dict(occlusion_mode="link_union", occlusion_margin_m=.01, p_drop=.1, blackout_len=5,
                 blackout_schedule="horizon_uniform", actor="student",
                 stopping="oracle_graspability", diagnostic_oracle=True)
MAIN = {"trace_r3": "checkpoints/trace_r3_seed{seed}.pt",
        "trace_bc": "checkpoints/trace_bc_seed{seed}.pt"}
# The supervision table: each DAgger round, then the two controls that hold the label budget fixed.
DAGGER = {"expert_bc": "checkpoints/trace_bc_seed{seed}.pt",
          "dagger_r1": "checkpoints/trace_r1_seed{seed}.pt",
          "dagger_r2": "checkpoints/trace_r2_seed{seed}.pt",
          "dagger_r3": "checkpoints/trace_r3_seed{seed}.pt",
          "student_state_bc": "checkpoints/controls/student_state_bc_seed{seed}.pt",
          "expert_bc_full_data": "checkpoints/controls/expert_bc_full_seed{seed}.pt"}
SEEDS = (0, 1, 2)
EVAL_SEEDS = (7,)



def campaign(name: str, cases: list, gpu: int, eval_seeds=EVAL_SEEDS, count: int = 511) -> Path:
    out = paths.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    for seed in eval_seeds:
        command = [paths.PYTHON, "-u", str(SIM / "isaacgymenvs/tools/run_evaluation_campaign.py"),
                   "--manifest", str(paths.require(paths.MANIFEST_DEV)),
                   "--output", str(out / "evaluation"), "--cases", str(out / "cases.json"),
                   "--teacher-checkpoint", str(paths.require(paths.TEACHER)),
                   "--reward-recipe", "graspability_only", "--gpu", str(gpu),
                   "--count", str(count), "--batch-size", "64", "--horizon", "120",
                   "--seeds", str(seed)]
        print("  " + " ".join(command[-8:]), flush=True)
        if subprocess.call(command, cwd=SIM, env=paths.child_env()) != 0:
            raise SystemExit(f"{name}: evaluation seed {seed} failed")
    return out


def main_table(gpu: int) -> None:
    cases = seeded_cases(MAIN)
    cases.append(dict(CONDITION, name="teacher_replay", actor="replay",
                      stopping="nominal_action_budget", diagnostic_oracle=False,
                      teacher_relative_budget=None))
    cases.append(dict(CONDITION, name="online_teacher", actor="teacher",
                      stopping="expert_graspability", diagnostic_oracle=False,
                      checkpoint=str(paths.TEACHER), teacher_relative_budget=None))
    out = campaign("main_table", cases, gpu)
    summarize(out, "Plan-conditioned execution and reference policies")


def seeded_cases(stages: dict) -> list:
    """Require every reported seed: never silently report a partial table."""
    cases = []
    for stage, pattern in stages.items():
        for seed in SEEDS:
            checkpoint = paths.DATA / pattern.format(seed=seed)
            paths.require(checkpoint)
            cases.append(dict(CONDITION, name=f"{stage}_s{seed}", checkpoint=str(checkpoint),
                              teacher_relative_budget=BUDGET))
    return cases


def dagger_table(gpu: int) -> None:
    """Rounds BC..R3 and the matched-label controls, all under one budget and condition."""
    cases = seeded_cases(DAGGER)
    if not cases:
        raise SystemExit("No student checkpoints found; run scripts/download_data.py")
    summarize(campaign("dagger_table", cases, gpu), "Student-state supervision")


# The search baselines share one budget: 5 s of search per push and at most 15 pushes, scored by
# the same grasp network. Only the tree search differs between them.
MCTS = dict(time_limit="5", max_actions="15", max_scenes_per_process="4")
SEARCH_ENVS = {"parallel": "500", "serial": "2"}


def run(label: str, command: list) -> None:
    print(f"[{label}]", flush=True)
    if subprocess.call(command, cwd=SIM, env=paths.child_env()) != 0:
        raise SystemExit(f"{label} failed")


def mcts_baselines(gpu: int) -> None:
    """PMBS (batched parallel search) and the serial single-environment search."""
    manifest = str(paths.require(paths.MANIFEST_DEV))
    for search, envs in SEARCH_ENVS.items():
        out = paths.RUNS / f"baseline_mcts_{search}"
        out.mkdir(parents=True, exist_ok=True)
        run(f"mcts_{search}", [paths.PYTHON, "-u", str(SIM / "pmbs_baseline/evaluate_manifest.py"),
                               "--manifest", manifest, "--output_dir", str(out), "--search", search,
                               "--num_envs", envs, "--time_limit", MCTS["time_limit"],
                               "--max_actions", MCTS["max_actions"],
                               "--max_scenes_per_process", MCTS["max_scenes_per_process"],
                               "--headless", "--sim_device", f"cuda:{gpu}"])
        run(f"mcts_{search}: verify", [paths.PYTHON, "-u", str(SIM / "pmbs_baseline/verify_results.py"),
                                       "--output_dir", str(out)])


def cartesian_baselines(gpu: int, batch: int = 64) -> None:
    """Spiral and straight-line controllers: fully observed, no re-sensing, parameter defaults."""
    manifest = paths.require(paths.MANIFEST_DEV)
    scenes = len(json.loads(manifest.read_text())["scenes"])
    for baseline in ("spiral", "straightline"):
        root = paths.RUNS / f"baseline_{baseline}"
        root.mkdir(parents=True, exist_ok=True)
        for offset in range(0, scenes, batch):
            out = root / f"batch_{offset:04d}"       # the summarizer globs batch_*/result.json
            if (out / "result.json").exists():
                continue
            spec = root / f"batch_{offset:04d}.spec.json"
            spec.write_text(json.dumps(dict(baseline=baseline, manifest=str(manifest), output=str(out),
                                            offset=offset, batch_size=min(batch, scenes - offset),
                                            seed=7), indent=2) + "\n")
            run(f"{baseline} batch_{offset:04d}",
                [paths.PYTHON, "-u", str(SIM / "isaacgymenvs/tools/evaluate_spiral.py"),
                 "task=MoreEvaluation", "test=True", "headless=True", "force_render=False",
                 "wandb_activate=False", f"sim_device=cuda:{gpu}", f"rl_device=cuda:{gpu}",
                 f"graphics_device_id={gpu}", f"+baseline_spec={spec}"])
        run(f"{baseline}: summarize",
            [paths.PYTHON, "-u", str(SIM / "isaacgymenvs/tools/summarize_spiral_evaluation.py"),
             str(root), "--expected-scenes", str(scenes)])


def baselines(gpu: int) -> None:
    mcts_baselines(gpu)
    cartesian_baselines(gpu)


def ablations(gpu: int, refit: bool = False) -> None:
    """Drive the ablation job graph: prepare, run, summarize.

    Without --refit the shipped ablation checkpoints are evaluated. With it every variant is
    retrained from the label sets first, which needs `download_data.py --only collections`.
    """
    tools = SIM / "cluster/plan_memory_ablations"
    out = paths.RUNS / "ablations"
    prepare = [paths.PYTHON, "-u", str(tools / "prepare.py"), "--data", str(paths.DATA), "--out", str(out)]
    if not refit:
        prepare.append("--use-bundle-fits")
    steps = [("prepare", prepare),
             ("run", [paths.PYTHON, "-u", str(tools / "run_pool.py"), "--out", str(out),
                      "--gpus", str(gpu), "--per-gpu", "2"]),
             ("summarize", [paths.PYTHON, "-u", str(tools / "summarize.py"), "--out", str(out)])]
    for label, command in steps:
        print(f"[ablations: {label}]", flush=True)
        if subprocess.call(command, cwd=SIM, env=paths.child_env()) != 0:
            raise SystemExit(f"ablations: {label} failed")
    print(f"\ntables and per-scene records: {out}")


def summarize(out: Path, title: str) -> None:
    """The campaign reporter writes SUMMARY.md beside the records and prints the group table."""
    command = [paths.PYTHON, str(SIM / "isaacgymenvs/tools/report_retrieval_campaign.py"),
               str(out / "evaluation")]
    print(f"\n== {title}", flush=True)
    if subprocess.call(command, cwd=SIM, env=paths.child_env()) == 0:
        print(f"\nper-scene records and SUMMARY.md: {out / 'evaluation'}")
    else:
        raise SystemExit(f"{title}: reporting failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=["main", "baselines", "dagger", "ablations", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--refit", action="store_true",
                    help="retrain each ablation variant instead of evaluating the shipped fit")
    args = ap.parse_args()
    paths.RUNS.mkdir(parents=True, exist_ok=True)
    if args.table in ("main", "all"):
        main_table(args.gpu)
    if args.table in ("baselines", "all"):
        baselines(args.gpu)
    if args.table in ("dagger", "all"):
        dagger_table(args.gpu)
    if args.table in ("ablations", "all"):
        ablations(args.gpu, args.refit)


if __name__ == "__main__":
    main()
