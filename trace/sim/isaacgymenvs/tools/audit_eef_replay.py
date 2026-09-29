"""Record or replay a physics trace using an archived trajectory's initial scene.

Run from isaacgymenvs (pmbs interpreter, LD_LIBRARY_PATH set):
  python tools/audit_eef_replay.py task=MoreOpenLoop train=MoreOpenLoopSetSCPPO \
    test=True headless=True +traj_json=/absolute/trajectory.json \
    +mode=record +out_dir=/absolute/audit
Then +mode=joint, cartesian, or cartesian_feedforward with
+traj_json=<audit>/record.json. The last mode uses recorded motor commands as
feedforward plus EEF tracking correction; cartesian is direct pose-only IK.

record regenerates the stored ACTION sequence; it does not solve a different
policy rollout. All outputs are new files; the source archive is untouched.
cartesian uses the full measured 6D trace and original timestamps, not sparse
policy-step points. Tracking error is reported even when graspability passes.
"""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa: F401 -- before torch
import tools._net_compat  # noqa: F401
import tools._ckpt_compat  # noqa: F401
import hydra
from hydra.utils import to_absolute_path
import numpy as np
import torch
from isaacgymenvs.open_loop.solve_in_twin import make_player
from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory
from isaacgymenvs.open_loop.trace_metrics import compare_physics
from isaacgymenvs.utils.utils import set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg):
    source = OpenLoopTrajectory.load(to_absolute_path(str(cfg.traj_json)))
    mode = str(cfg.get("mode", "joint"))
    if mode not in ("record", "joint", "cartesian", "cartesian_feedforward", "physics"):
        raise ValueError("Unknown EEF replay mode: " + mode)
    out = Path(to_absolute_path(str(cfg.out_dir)))
    out.mkdir(parents=True, exist_ok=True)
    scene = Path(to_absolute_path(source.scene_file))
    # Stable private numeric suite; avoid the shared mutable real2sim directory.
    suite = out / "scenes"
    suite.mkdir(exist_ok=True)
    if source.scene_text is not None or scene.resolve() != (suite / "000000.txt").resolve():
        source.write_scene(str(suite / "000000.txt"))
    pkg = Path(__file__).resolve().parents[1]
    cfg.task.env.test_cases.scene_root_dir = os.path.relpath(str(out), str(pkg))
    cfg.task.env.test_cases.difficulty_choice = "scenes"
    cfg.task.env.numEnvs = cfg.num_envs = 1
    cfg.checkpoint = to_absolute_path(source.checkpoint)
    cfg.seed = set_seed(int(source.metadata.get("seed", 42)), torch_deterministic=False)
    player, obs = make_player(cfg)
    env = player.env
    env.reset_idx(torch.arange(1, device=env.device))
    settle_steps = int(source.metadata.get("settle_steps", 30))
    settle_mm = float(env.settle(settle_steps)[0] * 1000) if settle_steps else 0.0
    obs = player.env_reset(env)
    env.clear_recording()
    env.enable_physics_recording(include_replay=True)
    if mode in ("joint", "cartesian", "cartesian_feedforward"):
        env.set_timed_replay([source], mode=mode)
    elif mode == "physics":
        env.set_replay([source.physics_path()])
    max_steps = len(source.dense) if mode == "record" else int(cfg.get("budget", 400))
    if max_steps < 1:
        raise ValueError("The audit needs at least one recorded action or replay step")
    q, oow = [], []
    for step in range(max_steps):
        action = source.dense[step]["action"] if mode == "record" else 0
        obs, _, _, _ = player.env_step(env, torch.tensor([action], device=env.device))
        q.append(float(env.grasp_q_parallel_values[0]))
        oow.append(bool(env.oow_violation()[0]))
        if mode != "record" and bool(env.replay_done[0]):
            break
        # Never let a recorded action tape reset the environment after a
        # terminal score; it must run the entire archived sequence once.
        env.reset_buf[:] = 0
    rec = env.recording_of(0)
    result = OpenLoopTrajectory(str(suite / "000000.txt"), source.checkpoint, dict(source.metadata))
    result.scene_text = (suite / "000000.txt").read_text()
    result.start_eef = source.start_eef
    result.dense = rec["steps"] if mode == "record" else source.dense
    result.waypoints = source.waypoints
    result.grasp = source.grasp if mode == "record" else None
    result.final_target = list(env.target_pose(0))
    result.set_physics(rec["physics"])
    complete = mode == "record" or bool(env.replay_done[0])
    result.metadata.update({"audit_mode": mode, "solved": complete and q[-1] > 0.9 and not any(oow),
                            "replay_done": complete,
                            "final_q": q[-1], "settle_disp_mm": settle_mm})
    result.save(str(out / (mode + ".json")))
    summary = {"mode": mode, "final_q": q[-1], "max_q": max(q), "oow": any(oow),
               "ticks": len(result.physics["samples"]), "settle_mm": settle_mm,
               "replay_done": complete}
    if source.physics is not None:
        summary.update(compare_physics(source.physics, result.physics))
    if mode == "record":
        old = np.asarray(source.dense_path())
        new = np.asarray(result.dense_path())
        summary["archived_eef_max_mm"] = float(np.linalg.norm(new-old, axis=1).max()*1000)
        (out / "cartesian_reference.json").write_text(json.dumps({
            "frame": "sim", "position_unit": "m", "time_unit": "s",
            "quaternion_order": "xyzw", "samples": result.cartesian_trace()}, indent=2) + "\n")
    (out / (mode + "_metrics.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("EEF_AUDIT " + json.dumps(summary), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
