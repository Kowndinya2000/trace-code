"""Evaluate simple Cartesian baselines in MoreEvaluation.

Example:
  python tools/evaluate_spiral.py task=MoreEvaluation test=True headless=True \
    force_render=False +spiral_spec=/absolute/path/to/spec.json

The simulator is fully observed. There is deliberately no occlusion model,
retract, or re-sense branch. Graspability is recomputed from rendered depth and
segmentation images at every completed waypoint.
"""
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # noqa: F401  # must precede torch
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from isaacgym.torch_utils import quat_apply
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.tasks.utils.constants import GRASP_Q_PUSH_THRESHOLD
from isaacgymenvs.open_loop.straightline_controller import (
    DEFAULT_DISTANCE_M as DEFAULT_STRAIGHTLINE_DISTANCE_M,
    DEFAULT_STEP_M as DEFAULT_STRAIGHTLINE_STEP_M,
    StraightLine,
)
from isaacgymenvs.open_loop.spiral_controller import (
    CircularSpiral, DEFAULT_ARC_STEP_M, DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TRAVEL_M, DEFAULT_MIN_RADIUS_M, DEFAULT_ORBIT_LOOPS,
    DEFAULT_RADIAL_STEP_M, DEFAULT_START_RADIUS_M, DEFAULT_TIMEOUT_S,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PROTOCOLS = {
    "spiral": "closed-loop-spiral-sim-v1",
    "straightline": "closed-loop-target-oriented-straightline-sim-v2",
}
SPIRAL_PARAM_DEFAULTS = {
    "max_steps": DEFAULT_MAX_STEPS,
    "timeout_s": DEFAULT_TIMEOUT_S,
    "max_travel_m": DEFAULT_MAX_TRAVEL_M,
    "arc_step_m": DEFAULT_ARC_STEP_M,
    "radial_step_m": DEFAULT_RADIAL_STEP_M,
    "min_radius_m": DEFAULT_MIN_RADIUS_M,
    "start_radius_m": DEFAULT_START_RADIUS_M,
    "orbit_loops": DEFAULT_ORBIT_LOOPS,
    "grasp_threshold": GRASP_Q_PUSH_THRESHOLD,
}
STRAIGHTLINE_PARAM_DEFAULTS = {
    "max_steps": int(math.ceil(DEFAULT_STRAIGHTLINE_DISTANCE_M /
                                DEFAULT_STRAIGHTLINE_STEP_M)),
    "timeout_s": DEFAULT_TIMEOUT_S,
    # This remains a fault guard on measured robot motion. The baseline's
    # commanded path length is controlled independently by distance_m.
    "max_travel_m": DEFAULT_MAX_TRAVEL_M,
    "distance_m": DEFAULT_STRAIGHTLINE_DISTANCE_M,
    "step_m": DEFAULT_STRAIGHTLINE_STEP_M,
    "grasp_threshold": GRASP_Q_PUSH_THRESHOLD,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def resolve_scene_path(path, manifest_path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = (manifest_path.parent / path, REPO_ROOT / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def read_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    rows = []
    seen = set()
    for source in payload.get("scenes", []):
        row = dict(source)
        scene_path = resolve_scene_path(row["path"], path)
        digest = sha256(scene_path)
        if digest != row["sha256"]:
            raise ValueError(f"Scene hash mismatch: {scene_path}")
        if digest in seen:
            raise ValueError(f"Duplicate scene content: {scene_path}")
        seen.add(digest)
        row["path"] = str(scene_path)
        rows.append(row)
    if not rows:
        raise ValueError("Empty scene manifest")
    payload["scenes"] = rows
    return payload


def validated_params(spec, baseline):
    params = dict(SPIRAL_PARAM_DEFAULTS if baseline == "spiral"
                  else STRAIGHTLINE_PARAM_DEFAULTS)
    params.update(spec.get("params", {}))
    for name in ("max_steps",):
        params[name] = int(params[name])
        if params[name] <= 0:
            raise ValueError(f"{name} must be positive")
    names = ["timeout_s", "max_travel_m", "grasp_threshold"]
    names += (["arc_step_m", "radial_step_m", "min_radius_m", "orbit_loops"]
              if baseline == "spiral" else ["distance_m", "step_m"])
    for name in names:
        params[name] = float(params[name])
        if not math.isfinite(params[name]) or params[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if params["grasp_threshold"] > 1:
        raise ValueError("grasp_threshold must be at most one")
    if baseline == "straightline":
        required_steps = int(math.ceil(params["distance_m"] / params["step_m"] - 1e-12))
        if params["max_steps"] < required_steps:
            raise ValueError("max_steps cannot complete the requested straight-line distance")
    else:
        start_radius = params.get("start_radius_m")
        if start_radius is not None:
            params["start_radius_m"] = float(start_radius)
            if (not math.isfinite(params["start_radius_m"]) or
                    params["start_radius_m"] < params["min_radius_m"]):
                raise ValueError("start_radius_m must be finite and at least min_radius_m")
    return params


def mesh_oow_by_object(env):
    """Return (env, object) endpoint containment using collision vertices."""
    state, vertices = env.block_state, env.mesh_vertices
    quat = state[:, :, None, 3:7].expand(*vertices.shape[:-1], 4)
    xyz = quat_apply(quat.reshape(-1, 4), vertices.reshape(-1, 3)).reshape_as(vertices)
    xyz = xyz + state[:, :, None, :3]
    return ((xyz[..., 0] < env.ws_x[0]) | (xyz[..., 0] > env.ws_x[1]) |
            (xyz[..., 1] < env.ws_y[0]) | (xyz[..., 1] > env.ws_y[1])).any(dim=-1)


def terminal_oow_reasons(env, aggregate_oow):
    by_object = cpu(mesh_oow_by_object(env))
    reason = np.full(env.num_envs, "scene_oow_transient", dtype="U32")
    reason[by_object[:, 1:].any(axis=1)] = "clutter_oow"
    reason[by_object[:, 0]] = "target_oow"
    reason[~aggregate_oow] = "running"
    return reason


def save_csv(path, rows):
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


def finish_process():
    # Isaac Gym's old bindings can segfault while destructing a successful sim.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    from isaacgymenvs.utils.utils import set_seed
    import isaacgymenvs

    spec_value = cfg.get("baseline_spec") or cfg.get("spiral_spec")
    if not spec_value:
        raise ValueError("Pass +baseline_spec=/absolute/path/to/spec.json")
    spec_path = Path(to_absolute_path(str(spec_value))).resolve()
    spec = json.loads(spec_path.read_text())
    baseline = str(spec.get("baseline", "spiral")).lower()
    if baseline not in PROTOCOLS:
        raise ValueError(f"Unknown baseline: {baseline}")
    protocol = PROTOCOLS[baseline]
    manifest_path = Path(spec["manifest"]).resolve()
    manifest = read_manifest(manifest_path)
    offset = int(spec.get("offset", 0))
    batch_size = int(spec.get("batch_size", 64))
    rows = manifest["scenes"][offset:offset + batch_size]
    if not rows:
        raise ValueError("Empty manifest batch")
    params = validated_params(spec, baseline)

    output = Path(spec["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_dir = output / "scenes"
    scene_dir.mkdir()
    for index, row in enumerate(rows):
        shutil.copyfile(row["path"], scene_dir / f"{index:06d}.txt")

    cfg.num_envs = len(rows)
    cfg.task.env.numEnvs = len(rows)
    cfg.task.env.test_cases.scene_root_dir = os.path.relpath(scene_dir.parent, ROOT)
    cfg.task.env.test_cases.difficulty_choice = scene_dir.name
    cfg.task.env.test_cases.sceneOffset = 0
    if cfg.task.name != "MoreEvaluation":
        raise ValueError("Baseline evaluation requires task=MoreEvaluation")
    if not cfg.test or not cfg.headless:
        raise ValueError("Baseline evaluation requires test=True headless=True")

    seed = int(spec.get("seed", 7))
    torch.cuda.set_device(torch.device(cfg.rl_device))
    set_seed(seed, False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(spec.get("cpu_threads", 4)))

    setup_t0 = time.perf_counter()
    env = isaacgymenvs.make(
        seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs, cfg.sim_device,
        cfg.rl_device, cfg.graphics_device_id, cfg.headless, cfg.multi_gpu,
        cfg.capture_video, cfg.force_render, cfg,
    )
    env.enable_physics_recording(False)
    if env.num_envs != len(rows):
        raise RuntimeError("Scene/slot count mismatch")

    permutations = np.tile(np.arange(env.num_objects - 1), (env.num_envs, 1))
    initial = env.pristine_block_state.clone()
    initial[:, :, 7:] = 0
    env.begin_attempt(initial, permutations)
    torch.cuda.synchronize(env.device)
    setup_s = time.perf_counter() - setup_t0

    n = env.num_envs
    initial_eef = cpu(env.gripper_pos[:, :2])
    initial_target = cpu(env.block_state[:, 0, :2])
    if baseline == "spiral":
        controllers = [CircularSpiral(
            arc_step_m=params["arc_step_m"],
            radial_step_m=params["radial_step_m"],
            min_radius_m=params["min_radius_m"],
            orbit_loops=params["orbit_loops"],
            start_radius_m=params["start_radius_m"],
        ) for _ in range(n)]
        straightline_direction = np.full((n, 2), np.nan, np.float64)
    else:
        controllers = [StraightLine(
            initial_eef[i], initial_target[i],
            distance_m=params["distance_m"], step_m=params["step_m"],
        ) for i in range(n)]
        straightline_direction = np.stack(
            [controller.direction for controller in controllers])
    reason = np.full(n, "running", dtype="U32")
    terminal_step = np.full(n, -1, np.int64)
    terminal_q = np.full(n, np.nan, np.float64)
    terminal_wall = np.full(n, np.nan, np.float64)
    travelled = np.zeros(n, np.float64)
    sim_solve_time = np.zeros(n, np.float64)
    physics_ticks = np.zeros(n, np.int64)
    waypoint_clips = np.zeros(n, np.int64)
    grasp_evaluations = np.ones(n, np.int64)
    radius = np.full(n, np.nan, np.float64)
    orbit_progress = np.zeros(n, np.float64)
    path_progress = np.zeros(n, np.float64)
    radial_entry_steps = np.zeros(n, np.int64)
    initial_q = cpu(env.grasp_q_parallel_values).astype(np.float64)
    max_q = initial_q.copy()

    shape = (params["max_steps"] + 1, n)
    traces = {
        "q": np.full(shape, np.nan, np.float32),
        "radius_m": np.full(shape, np.nan, np.float32),
        "orbit_progress_rad": np.full(shape, np.nan, np.float32),
        "path_progress_m": np.full(shape, np.nan, np.float32),
        "heading_xy": np.full(shape + (2,), np.nan, np.float32),
        "radial_entry": np.zeros(shape, bool),
        "travelled_m": np.full(shape, np.nan, np.float32),
        "sim_solve_time_s": np.full(shape, np.nan, np.float32),
        "eef_xy": np.full(shape + (2,), np.nan, np.float32),
        "target_xy": np.full(shape + (2,), np.nan, np.float32),
        "waypoint_xy": np.full(shape + (2,), np.nan, np.float32),
        "command_physics_ticks": np.zeros(shape, np.int16),
        "observed": np.zeros(shape, bool),
    }
    traces["q"][0] = initial_q
    traces["eef_xy"][0] = initial_eef
    traces["target_xy"][0] = initial_target
    traces["heading_xy"][0] = straightline_direction
    traces["travelled_m"][0] = 0
    traces["path_progress_m"][0] = 0
    traces["sim_solve_time_s"][0] = 0
    traces["observed"][0] = True

    def retire(mask, why, step, q, start_wall):
        mask = np.asarray(mask, bool) & (reason == "running")
        if isinstance(why, str):
            reason[mask] = why
        else:
            reason[mask] = np.asarray(why)[mask]
        terminal_step[mask] = step
        terminal_q[mask] = q[mask]
        terminal_wall[mask] = time.perf_counter() - start_wall
        if mask.any():
            env.freeze(torch.as_tensor(np.flatnonzero(mask), device=env.device))

    rollout_t0 = time.perf_counter()
    q = initial_q.copy()
    aggregate_oow = cpu(env.physical_oow() | env.requested_initial_oow)
    invalid = cpu(env.invalid_state()) | ~np.isfinite(q)
    retire(aggregate_oow, terminal_oow_reasons(env, aggregate_oow), 0, q, rollout_t0)
    retire(invalid, "invalid_state", 0, q, rollout_t0)
    retire(q == -2.0, "out_of_view", 0, q, rollout_t0)
    retire(q > params["grasp_threshold"], "success", 0, q, rollout_t0)

    zero_action = torch.zeros(n, dtype=torch.long, device=env.device)
    dt = float(cfg.task.sim.dt)
    max_command_control_steps = int(math.ceil(env.replay_max_ticks / env.control_freq_inv)) + 1

    for command_index in range(params["max_steps"]):
        live = reason == "running"
        if not live.any():
            break
        retire(live & (sim_solve_time >= params["timeout_s"]), "timeout",
               command_index, q, rollout_t0)
        live = reason == "running"
        if not live.any():
            break

        eef_before = cpu(env.gripper_pos)
        target = cpu(env.block_state[:, 0, :2])
        waypoints = eef_before.copy()
        complete_after_command = np.zeros(n, bool)
        radial_entry_command = np.zeros(n, bool)
        for i in np.flatnonzero(live):
            if baseline == "spiral":
                baseline_step = controllers[i].next(eef_before[i, :2], target[i])
                radius[i] = baseline_step.radius_m
                orbit_progress[i] = baseline_step.orbit_progress_rad
                radial_entry_command[i] = baseline_step.phase == "radial_entry"
                radial_entry_steps[i] += int(radial_entry_command[i])
            else:
                baseline_step = controllers[i].next(eef_before[i, :2], target[i])
                path_progress[i] = baseline_step.progress_m
                straightline_direction[i] = baseline_step.direction
            unclipped = baseline_step.waypoint
            clipped = np.clip(unclipped,
                              [env.ws_x[0], env.ws_y[0]],
                              [env.ws_x[1], env.ws_y[1]])
            waypoint_clips[i] += int(not np.allclose(unclipped, clipped, atol=1e-12))
            waypoints[i, :2] = clipped
            complete_after_command[i] = baseline_step.complete

        proposed = np.linalg.norm(waypoints[:, :2] - eef_before[:, :2], axis=1)
        over_travel = live & (travelled + proposed > params["max_travel_m"] + 1e-9)
        retire(over_travel, "travel_limit", command_index, q, rollout_t0)
        live = reason == "running"
        if not live.any():
            break

        env.command_cartesian_waypoints(waypoints, live)
        pending = live.copy()
        command_ticks = np.zeros(n, np.int64)
        command_oow = np.zeros(n, bool)
        command_invalid = np.zeros(n, bool)
        for _ in range(max_command_control_steps):
            env.step(zero_action)
            ticks_now = cpu(env.replay_ticks).astype(np.int64)
            done_now = cpu(env.replay_done).astype(bool)
            command_oow |= cpu(env.step_mesh_oow)
            command_invalid |= cpu(env.invalid_state())
            newly_done = pending & done_now
            command_ticks[newly_done] = ticks_now[newly_done]
            pending &= ~done_now
            if not pending.any():
                break
        stalled = pending.copy()
        if stalled.any():
            command_ticks[stalled] = cpu(env.replay_ticks)[stalled]

        torch.cuda.synchronize(env.device)
        eef_after = cpu(env.gripper_pos)
        goal_error = np.linalg.norm(eef_after - waypoints, axis=1)
        stalled |= live & (goal_error >= env.replay_tol) & \
            (command_ticks >= env.replay_max_ticks)
        actual_travel = np.linalg.norm(eef_after[:, :2] - eef_before[:, :2], axis=1)
        travelled[live] += actual_travel[live]
        physics_ticks[live] += command_ticks[live]
        sim_solve_time[live] += command_ticks[live] * dt
        q = cpu(env.grasp_q_parallel_values).astype(np.float64)
        max_q[live] = np.maximum(max_q[live], q[live])
        grasp_evaluations[live] += 1
        step = command_index + 1
        target_after = cpu(env.block_state[:, 0, :2])
        traces["q"][step] = q
        traces["radius_m"][step] = radius
        traces["orbit_progress_rad"][step] = orbit_progress
        traces["path_progress_m"][step] = path_progress
        traces["heading_xy"][step] = straightline_direction
        traces["radial_entry"][step] = radial_entry_command
        traces["travelled_m"][step] = travelled
        traces["sim_solve_time_s"][step] = sim_solve_time
        traces["eef_xy"][step] = eef_after[:, :2]
        traces["target_xy"][step] = target_after
        traces["waypoint_xy"][step] = waypoints[:, :2]
        traces["command_physics_ticks"][step] = command_ticks
        traces["observed"][step, live] = True

        invalid = command_invalid | cpu(env.invalid_state()) | ~np.isfinite(q)
        aggregate_oow = command_oow | cpu(env.physical_oow() | env.step_mesh_oow)
        retire(aggregate_oow, terminal_oow_reasons(env, aggregate_oow),
               step, q, rollout_t0)
        retire(invalid, "invalid_state", step, q, rollout_t0)
        retire(q == -2.0, "out_of_view", step, q, rollout_t0)
        retire(stalled, "motion_stall", step, q, rollout_t0)
        retire((q > params["grasp_threshold"]), "success", step, q, rollout_t0)
        retire(complete_after_command, f"{baseline}_complete", step, q, rollout_t0)
        retire((reason == "running") & (sim_solve_time >= params["timeout_s"]),
               "timeout", step, q, rollout_t0)
        if step % 30 == 0:
            counts = Counter(reason)
            print(f"[{baseline} {offset}:{offset+n}] step={step} active={counts['running']} "
                  f"success={counts['success']}/{n}", flush=True)

    retire(reason == "running", "step_limit", params["max_steps"], q, rollout_t0)
    torch.cuda.synchronize(env.device)
    rollout_wall_s = time.perf_counter() - rollout_t0

    final_eef = cpu(env.gripper_pos[:, :2])
    final_target = cpu(env.block_state[:, 0, :2])
    result_rows = []
    for i, scene in enumerate(rows):
        result_rows.append({
            "manifest_index": offset + i,
            "scene_file": Path(scene["path"]).name,
            "sha256": scene["sha256"],
            "tier": scene.get("tier", "unspecified"),
            "success": bool(reason[i] == "success"),
            "reason": str(reason[i]),
            "terminal_step": int(terminal_step[i]),
            "terminal_q": float(terminal_q[i]) if np.isfinite(terminal_q[i]) else None,
            "max_q": float(max_q[i]) if np.isfinite(max_q[i]) else None,
            "initial_q": float(initial_q[i]) if np.isfinite(initial_q[i]) else None,
            "travelled_m": float(travelled[i]),
            "sim_solve_time_s": float(sim_solve_time[i]),
            "batch_wall_time_to_terminal_s": float(terminal_wall[i]),
            "grasp_evaluations": int(grasp_evaluations[i]),
            "physics_ticks": int(physics_ticks[i]),
            "waypoint_clips": int(waypoint_clips[i]),
            "terminal_radius_m": float(radius[i]) if np.isfinite(radius[i]) else None,
            "terminal_orbit_progress_rad": float(orbit_progress[i]),
            "terminal_path_progress_m": float(path_progress[i]),
            "straightline_direction_x": (
                float(straightline_direction[i, 0])
                if np.isfinite(straightline_direction[i, 0]) else None),
            "straightline_direction_y": (
                float(straightline_direction[i, 1])
                if np.isfinite(straightline_direction[i, 1]) else None),
            "radial_entry_steps": int(radial_entry_steps[i]),
            "initial_eef_xy": initial_eef[i].tolist(),
            "initial_target_xy": initial_target[i].tolist(),
            "terminal_eef_x": float(final_eef[i, 0]),
            "terminal_eef_y": float(final_eef[i, 1]),
            "terminal_target_x": float(final_target[i, 0]),
            "terminal_target_y": float(final_target[i, 1]),
            "scene": scene,
        })

    counts = Counter(reason)
    successes = counts["success"]
    summary = {
        "complete": True,
        "protocol": protocol,
        "baseline": baseline,
        "offset": offset,
        "scenes": n,
        "successes": successes,
        "success_pct": 100.0 * successes / n,
        "reasons": dict(sorted(counts.items())),
        "params": params,
        "occlusion_model": False,
        "target_pose_source": (
            "current simulator ground-truth block state at every decision"
            if baseline == "spiral"
            else "current simulator ground-truth block state at every decision"),
        "grasp_signal": "rendered depth+segmentation through simulator stock x16 grasp classifier",
        "timeout_clock": "per-scene active simulator physics time",
        "initialization_hold_included_in_solve_time": False,
        "setup_wall_s": setup_s,
        "rollout_wall_s": rollout_wall_s,
        "mean_sim_solve_time_s": float(sim_solve_time.mean()),
        "median_sim_solve_time_s": float(np.median(sim_solve_time)),
        "mean_batch_wall_time_to_terminal_s": float(terminal_wall.mean()),
        "rows": result_rows,
    }

    tracked = [
        "tools/evaluate_spiral.py", "tasks/utils/constants.py",
        "tasks/more_evaluation.py", "tasks/more_open_loop.py",
        "tasks/more_teacher.py", "tasks/more_robust.py", "tasks/more.py",
        "cfg/task/MoreEvaluation.yaml", "cfg/task/MoreOpenLoop.yaml",
    ]
    if baseline == "spiral":
        tracked += ["open_loop/spiral_controller.py", "open_loop/run_spiral.py"]
    else:
        tracked += ["open_loop/straightline_controller.py"]
    provenance = {
        "protocol": protocol,
        "baseline": baseline,
        "spec": spec,
        "spec_sha256": sha256(spec_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_sha256": {path: sha256(ROOT / path) for path in tracked},
        "grasp_checkpoints": {
            path: sha256(ROOT / path) for path in
            ("logs_grasp/snapshot-post-020000.reinforcement.pth",
             "logs_grasp/grasp_model-89.pth")
        },
        "configuration": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "host": platform.node(),
        "workspace": [list(env.ws_x), list(env.ws_y)],
        "scene_rows": rows,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_csv(output / "per_scene.csv", result_rows)
    np.savez_compressed(output / "trace.npz", **traces,
                        manifest_index=np.arange(offset, offset + n),
                        reason=reason, terminal_step=terminal_step,
                        straightline_direction_xy=straightline_direction)
    (output / "complete.json").write_text(json.dumps({
        "complete": True, "protocol": protocol, "baseline": baseline, "scenes": n,
        "successes": successes, "success_pct": summary["success_pct"],
    }, indent=2) + "\n")
    print(f"{baseline.upper()} EVALUATION COMPLETE: {successes}/{n} "
          f"({summary['success_pct']:.2f}%) {dict(sorted(counts.items()))} "
          f"rollout={rollout_wall_s:.1f}s", flush=True)
    finish_process()


if __name__ == "__main__":
    main()
