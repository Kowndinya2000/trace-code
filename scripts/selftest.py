#!/usr/bin/env python3
"""Check that an installation can actually run the method.

    python scripts/selftest.py            # imports, data layout, then two scenes end to end
    python scripts/selftest.py --dry-run  # imports and data layout only, no GPU needed

It solves two development scenes with the released student and prints their outcome. Two minutes
on one GPU. A pass means Isaac Gym, the renderer, the checkpoints and the scene files all line up.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace.common import paths  # noqa: E402

paths.ensure_runtime()

SIM = paths.SIM


def check_layout() -> list:
    problems = []
    for label, path in (("scene manifest", paths.MANIFEST_DEV), ("teacher", paths.TEACHER),
                        ("student", paths.STUDENT), ("assets", paths.ASSETS),
                        ("grasp models", paths.GRASP_MODELS)):
        state = "ok" if Path(path).exists() else "MISSING"
        print(f"  {label:<14} {state:<8} {path}")
        if state == "MISSING":
            problems.append(label)
    return problems


def check_imports() -> list:
    problems = []
    for module in ("isaacgym", "torch", "cv2", "hydra", "yaml"):
        try:
            __import__(module)
            print(f"  {module:<14} ok")
        except Exception as error:                      # noqa: BLE001 - report, do not raise
            print(f"  {module:<14} FAILED   {error}")
            problems.append(module)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=2)
    args = ap.parse_args()

    print("imports")
    problems = check_imports()
    print("data")
    problems += check_layout()
    if problems:
        raise SystemExit(f"\nself-test failed: {', '.join(problems)}\n"
                         "see docs/INSTALL.md; data comes from scripts/download_data.py")
    if args.dry_run:
        print("\ndry run ok")
        return

    root = paths.RUNS / "selftest"
    root.mkdir(parents=True, exist_ok=True)
    # The evaluator freezes its inputs and deliberately refuses to reuse a campaign after
    # code or checkpoints change. Give every self-test its own directory so rerunning this
    # check after an update preserves the old record without tripping that integrity guard.
    out = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    case = dict(name="selftest", checkpoint=str(paths.STUDENT), actor="student",
                occlusion_mode="link_union", occlusion_margin_m=.01, p_drop=.1, blackout_len=5,
                blackout_schedule="horizon_uniform", stopping="oracle_graspability",
                diagnostic_oracle=True,
                teacher_relative_budget=dict(protocol="teacher-relative-action-budget-v1",
                                             extra_steps=20, extra_length_m=.15, primitive_length_m=.04,
                                             risk_start=.70, risk_temperature=.10, max_imitation_weight=3.,
                                             termination="first of step or planar TCP-travel cap",
                                             success="simulator graspability only; budget exhaustion is timeout"))
    (out / "cases.json").write_text(json.dumps([case], indent=2) + "\n")
    command = [paths.PYTHON, "-u", str(SIM / "isaacgymenvs/tools/run_evaluation_campaign.py"),
               "--manifest", str(paths.MANIFEST_DEV), "--output", str(out / "evaluation"),
               "--cases", str(out / "cases.json"), "--teacher-checkpoint", str(paths.TEACHER),
               "--reward-recipe", "graspability_only", "--gpu", str(args.gpu),
               "--count", str(args.scenes), "--batch-size", str(args.scenes),
               "--horizon", "120", "--seeds", "7"]
    print(f"\nsolving {args.scenes} scenes")
    if subprocess.call(command, cwd=SIM, env=paths.child_env()) != 0:
        raise SystemExit("self-test failed: the evaluation did not complete")

    rows = []
    for record in sorted((out / "evaluation").glob("**/selftest.json")):
        rows += json.loads(record.read_text())["rows"]
    if not rows:
        raise SystemExit("self-test failed: no episodes were recorded")
    for row in rows:
        travel = row.get("terminal_travel_m")
        detail = f"{travel:.3f} m" if isinstance(travel, (int, float)) else ""
        print(f"  {row['scene']['sha256'][:8]}  {row['reason']:<18} "
              f"{row['terminal_step']:>3} decisions  {detail}")
    print(f"\nself-test passed: {sum(r['success'] for r in rows)}/{len(rows)} solved")


if __name__ == "__main__":
    main()
