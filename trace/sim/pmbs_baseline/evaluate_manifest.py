"""Evaluate the vendored PMBS policy on an ordered scene manifest.

This keeps the published PMBS policy/search/environment semantics while adding a
resumable, machine-readable evaluation harness.  One result JSON is committed
after every scene and summary.json is regenerated from committed results.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# PMBS uses working-directory-relative action and asset paths.  Anchor those to
# this copied module, never to the upstream PMBS checkout.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import cv2
import numpy as np
from isaacgym import gymutil
import torch

from constants import (
    GRASP_Q_GRASP_THRESHOLD,
    GRASP_Q_PUSH_THRESHOLD,
    NUM_ROTATION,
    PIXEL_SIZE,
    TARGET_LOWER,
    TARGET_UPPER,
    WORKSPACE_LIMITS,
)
from environment import Environment
from mcts_parallel_new.nodes import PushSearchNode
from mcts_parallel_new.push import PushState
from mcts_parallel_new.search import MonteCarloTreeSearch
from mcts_utils import MCTSHelper

SEARCH_MODULES = {"parallel": "mcts_parallel_new", "serial": "mcts_serial"}


def select_search(mode: str) -> str:
    """Bind the search implementation used by evaluate_scene.

    "parallel" is PMBS itself.  "serial" is the single-simulation-environment
    MCTS that PMBS' parallel search is measured against; it shares the scene
    manifest, grasp networks, action sampler, push primitive and action cap, so
    the only difference is how the tree is searched.
    """
    global PushSearchNode, PushState, MonteCarloTreeSearch
    package = SEARCH_MODULES[mode]
    PushSearchNode = importlib.import_module(f"{package}.nodes").PushSearchNode
    PushState = importlib.import_module(f"{package}.push").PushState
    MonteCarloTreeSearch = importlib.import_module(f"{package}.search").MonteCarloTreeSearch
    return package


DEFAULT_MANIFEST = Path(
    os.environ.get("TRACE_DATA", BASE_DIR.parent.parent.parent / "data")
) / "scenes/development.json"
DEFAULT_GRASP_MODEL = BASE_DIR.parent / "isaacgymenvs/logs_grasp/snapshot-post-020000.reinforcement.pth"
DEFAULT_GRASP_EVAL_MODEL = BASE_DIR.parent / "isaacgymenvs/logs_grasp/grasp_model-89.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
        {"name": "--num_envs", "type": int, "default": 500},
        {"name": "--search", "type": str, "default": "parallel"},
        {"name": "--manifest", "type": str, "default": str(DEFAULT_MANIFEST)},
        {"name": "--output_dir", "type": str, "required": True},
        {"name": "--time_limit", "type": float, "default": 5.0},
        {"name": "--max_actions", "type": int, "default": 15},
        {"name": "--start_index", "type": int, "default": 0},
        {"name": "--limit", "type": int, "default": 0},
        {"name": "--max_scenes_per_process", "type": int, "default": 0},
        {"name": "--seed", "type": int, "default": 1234},
        {"name": "--grasp_model", "type": str, "default": str(DEFAULT_GRASP_MODEL)},
        {"name": "--grasp_eval_model", "type": str, "default": str(DEFAULT_GRASP_EVAL_MODEL)},
    ]
    return gymutil.parse_arguments(
        description="PMBS evaluation on an ordered JSON manifest",
        custom_parameters=parameters,
        headless=True,
    )


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"Manifest has no scenes: {path}")
    seen = set()
    for index, entry in enumerate(scenes):
        scene_path = Path(entry["path"]).resolve()
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
        expected = entry.get("sha256")
        actual = sha256(scene_path)
        if expected and actual != expected:
            raise ValueError(f"Scene hash mismatch at index {index}: {scene_path}")
        if actual in seen:
            raise ValueError(f"Duplicate scene content at index {index}: {scene_path}")
        seen.add(actual)
    return manifest, scenes


def scalar_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def evaluate_scene(args, helper: MCTSHelper, entry: dict, index: int) -> dict:
    scene_path = Path(entry["path"]).resolve()
    random.seed(args.seed + index)
    np.random.seed(args.seed + index)
    torch.manual_seed(args.seed + index)

    args.test_case = str(scene_path)
    setup_start = time.perf_counter()
    env = Environment(args)
    setup_time = time.perf_counter() - setup_start
    helper.set_env(env)
    env_ids_all = torch.arange(env.num_envs, device=helper.device)
    env_ids_main = torch.arange(1, device=helper.device)
    trace = []
    pushes = 0
    grasps = 0
    planning_time = 0.0
    success = False
    reason = "action_limit"
    execution_start = time.perf_counter()

    try:
        env.reset_idx(env_ids_all)
        for action_index in range(args.max_actions):
            color_images, depth_images, _ = env.render_camera(
                env_ids_main, color=True, depth=True, segm=False
            )
            color_image, depth_image = color_images[0], depth_images[0]
            target_pixels = int(np.count_nonzero(cv2.inRange(
                cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV), TARGET_LOWER, TARGET_UPPER
            )))
            if target_pixels < 10:
                reason = "target_missing"
                break

            grasp_q, best_pixel, _ = helper.get_grasp_q(
                color_image, depth_image, post_checking=True
            )
            _, focal_depths, _ = env.render_camera(
                env_ids_main, color=False, depth=True, segm=False, focal_target=True
            )
            classifier_q = helper.grasp_eval(focal_depths[0])

            if grasp_q > GRASP_Q_GRASP_THRESHOLD:
                rotation = float(np.deg2rad(best_pixel[0] * (360.0 / NUM_ROTATION)) + np.pi / 2)
                position = [
                    float(best_pixel[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]),
                    float(best_pixel[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]),
                    0.01,
                ]
                grasp_start = time.perf_counter()
                grasp_success = scalar_bool(env.grasp_idx(env_ids_main, [position], [rotation]))
                grasps += 1
                trace.append({
                    "action_index": action_index,
                    "type": "grasp",
                    "grasp_q": float(grasp_q),
                    "classifier_q": float(classifier_q),
                    "position": position,
                    "rotation_rad": rotation,
                    "execution_time_s": time.perf_counter() - grasp_start,
                    "success": grasp_success,
                })
                if grasp_success:
                    success = True
                    reason = "success"
                    break
                continue

            plan_start = time.perf_counter()
            state_ok, object_states, _ = env.save_object_states(env_ids_main)
            if not state_ok:
                reason = "state_not_static"
                break
            initial_q = grasp_q if (
                classifier_q > GRASP_Q_PUSH_THRESHOLD
                and grasp_q <= GRASP_Q_GRASP_THRESHOLD
            ) else classifier_q
            initial_state = PushState("root", object_states[0], float(initial_q), 0, helper)
            helper.simulation_recorder["root"] = (
                object_states[0], color_image, depth_image, float(classifier_q)
            )
            root = PushSearchNode(initial_state)
            serial = args.search == "serial"
            search = (
                MonteCarloTreeSearch(root, args.time_limit)
                if serial
                else MonteCarloTreeSearch(root, env.num_envs - 1, args.time_limit)
            )
            if classifier_q > GRASP_Q_PUSH_THRESHOLD and grasp_q <= GRASP_Q_GRASP_THRESHOLD:
                PushState.grasp_method = "dqn"
                PushState.max_level = 1
            best_node = (
                search.best_action(eval=True) if serial
                else search.best_action_parallel(eval=True)
            )
            plan_elapsed = time.perf_counter() - plan_start
            planning_time += plan_elapsed
            if best_node is None or best_node.prev_move is None:
                reason = "no_push"
                break

            start = best_node.prev_move.pos0
            end = best_node.prev_move.pos1
            push_start = [[
                float(start[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]),
                float(start[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]),
                0.01,
            ]]
            push_end = [[
                float(end[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]),
                float(end[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]),
                0.01,
            ]]
            push_exec_start = time.perf_counter()
            push_ok = scalar_bool(env.push_idx(env_ids_main, push_start, push_end))
            push_exec_time = time.perf_counter() - push_exec_start
            pushes += 1
            trace.append({
                "action_index": action_index,
                "type": "push",
                "grasp_q": float(grasp_q),
                "classifier_q": float(classifier_q),
                "planning_time_s": plan_elapsed,
                "execution_time_s": push_exec_time,
                "start": push_start[0],
                "end": push_end[0],
                "motion_ok": push_ok,
            })
            helper.reset()
            del initial_state, root, search, best_node
            gc.collect()
            if not push_ok:
                reason = "push_execution_failed"
                break
    except Exception as error:
        reason = "error"
        trace.append({
            "type": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
    finally:
        execution_time = time.perf_counter() - execution_start
        helper.reset()
        env.close()
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "index": index,
        "scene": str(scene_path),
        "scene_sha256": entry.get("sha256") or sha256(scene_path),
        "tier": entry.get("tier", "unknown"),
        "success": success,
        "reason": reason,
        "pushes": pushes,
        "grasps": grasps,
        "actions": pushes + grasps,
        "planning_time_s": planning_time,
        "execution_time_s": execution_time,
        "setup_time_s": setup_time,
        "total_time_s": setup_time + execution_time,
        "trace": trace,
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(results: list[dict], expected: int) -> dict:
    complete = len(results)
    successes = sum(bool(row["success"]) for row in results)

    def stats(rows: list[dict]) -> dict:
        n = len(rows)
        s = sum(bool(row["success"]) for row in rows)
        succeeded = [row for row in rows if row["success"]]
        return {
            "scenes": n,
            "successes": s,
            "success_rate": s / n if n else None,
            "success_pct": 100.0 * s / n if n else None,
            "avg_pushes": mean([row["pushes"] for row in rows]),
            "avg_pushes_successful": mean([row["pushes"] for row in succeeded]),
            "avg_execution_time_s": mean([row["execution_time_s"] for row in rows]),
            "avg_execution_time_successful_s": mean([row["execution_time_s"] for row in succeeded]),
            "avg_planning_time_s": mean([row["planning_time_s"] for row in rows]),
            "avg_total_time_s": mean([row["total_time_s"] for row in rows]),
            "total_pushes": sum(row["pushes"] for row in rows),
            "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
        }

    tiers = defaultdict(list)
    for result in results:
        tiers[result["tier"]].append(result)
    return {
        "state": "complete" if complete == expected else "running",
        "expected_scenes": expected,
        "completed_scenes": complete,
        "remaining_scenes": expected - complete,
        "successes": successes,
        "overall": stats(results),
        "by_tier": {tier: stats(rows) for tier, rows in sorted(tiers.items())},
        "updated_unix": time.time(),
    }


def main() -> None:
    args = parse_args()
    if args.search not in SEARCH_MODULES:
        raise ValueError(f"--search must be one of {sorted(SEARCH_MODULES)}: {args.search}")
    search_package = select_search(args.search)
    if args.search == "serial" and args.num_envs < 2:
        # env 0 is the real scene, env 1 is the single simulation env the serial
        # search steps through (MCTSHelper.other_id).
        raise ValueError("--search serial needs --num_envs 2 or more")
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, all_scenes = load_manifest(manifest_path)
    stop = len(all_scenes) if args.limit <= 0 else min(len(all_scenes), args.start_index + args.limit)
    selected = list(enumerate(all_scenes))[args.start_index:stop]
    if not selected:
        raise ValueError("Selected scene range is empty")

    grasp_model = Path(args.grasp_model).resolve()
    grasp_eval_model = Path(args.grasp_eval_model).resolve()
    protocol = {
        "protocol": f"pmbs-{args.search}-mcts-manifest-v1",
        "simulation_only": True,
        "source": "parallel MCTS baseline of Huang, Boularias and Yu (IROS 2022); see NOTICE",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_role": manifest.get("role"),
        "manifest_scenes": len(all_scenes),
        "selected_start": args.start_index,
        "selected_stop": stop,
        "num_envs": args.num_envs,
        "search": args.search,
        "search_module": search_package,
        "mcts_time_limit_s_per_push": args.time_limit,
        "max_actions_per_scene": args.max_actions,
        "seed": args.seed,
        "grasp_model": str(grasp_model),
        "grasp_model_sha256": sha256(grasp_model),
        "grasp_eval_model": str(grasp_eval_model),
        "grasp_eval_model_sha256": sha256(grasp_eval_model),
        "success_definition": "PMBS simulated grasp-and-lift succeeds within the action cap",
        "execution_time_definition": "wall clock after simulator reset through terminal outcome; excludes simulator/model setup",
        "push_definition": "one executed 10 cm PMBS push primitive",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        # Campaigns written before --search existed are parallel PMBS runs.
        legacy_defaults = {"search": "parallel"}
        for key in (
            "manifest_sha256", "selected_start", "selected_stop", "num_envs",
            "search", "mcts_time_limit_s_per_push", "max_actions_per_scene", "seed",
            "grasp_model_sha256", "grasp_eval_model_sha256",
        ):
            if existing.get(key, legacy_defaults.get(key)) != protocol.get(key):
                raise ValueError(f"Cannot resume: protocol field changed: {key}")
    else:
        atomic_json(protocol_path, protocol)

    scene_dir = output / "scenes"
    scene_dir.mkdir(exist_ok=True)
    helper = MCTSHelper(str(grasp_model), str(grasp_eval_model), args.seed)
    processed_this_process = 0
    recycle_required = False
    try:
        for index, entry in selected:
            result_path = scene_dir / f"{index:04d}.json"
            if result_path.exists():
                existing = json.loads(result_path.read_text())
                if existing.get("scene_sha256") != entry.get("sha256"):
                    raise ValueError(f"Cannot resume: result/manifest mismatch at {index}")
                continue
            print(f"PMBS scene {index + 1}/{stop}: {entry['path']}", flush=True)
            result = evaluate_scene(args, helper, entry, index)
            if result["reason"] == "error":
                failure_path = output / "failures" / f"{index:04d}-{int(time.time())}.json"
                atomic_json(failure_path, result)
                raise RuntimeError(
                    f"PMBS scene {index} raised an exception; saved {failure_path}; "
                    "leaving the scene uncommitted for automatic retry"
                )
            atomic_json(result_path, result)
            processed_this_process += 1
            rows = [json.loads(path.read_text()) for path in sorted(scene_dir.glob("*.json"))]
            atomic_json(output / "summary.json", summarize(rows, len(selected)))
            print(
                f"scene={index} success={result['success']} pushes={result['pushes']} "
                f"execution_s={result['execution_time_s']:.3f} reason={result['reason']}",
                flush=True,
            )
            if (
                args.max_scenes_per_process > 0
                and processed_this_process >= args.max_scenes_per_process
                and any(not (scene_dir / f"{i:04d}.json").exists() for i, _ in selected)
            ):
                recycle_required = True
                print(
                    f"Recycling after {processed_this_process} scenes to release Isaac Gym GPU memory",
                    flush=True,
                )
                break
    finally:
        helper.close_pool()

    rows = [json.loads(path.read_text()) for path in sorted(scene_dir.glob("*.json"))]
    final_summary = summarize(rows, len(selected))
    atomic_json(output / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    if recycle_required:
        # EX_TEMPFAIL makes systemd's Restart=on-failure start a clean process,
        # which resumes from the per-scene JSON commits.
        raise SystemExit(75)


if __name__ == "__main__":
    main()
