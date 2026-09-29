"""Stage 2: solve perceived scene(s) in the digital twin, export trajectories.

Rolls out the trained PPO policy (deterministic) in MoreOpenLoop — one env
per scene file in the configured scene dir (num_envs=1 for the perceived
real scene, num_envs=N to batch-solve a suite) — until the grasp network
reports the target graspable, an object leaves the workspace (paper OOW
rule -> failure), or the step budget runs out. Solved envs are frozen so the
first episode is the only episode. Then the tiled 16-rotation GPN gives the
final grasp pose and one OpenLoopTrajectory JSON is written per scene.

Run from isaacgymenvs/ (pmbs env, LD_LIBRARY_PATH set):

  # the perceived real scene (default cfg: test-cases/real2sim -> 1 env)
  python open_loop/solve_in_twin.py task=MoreOpenLoop test=True headless=True \\
      checkpoint=runs/<exp>/nn/<ckpt>.pth +trajectory_out=open_loop/out/real2sim
  (num_envs is clamped to the number of scene files; +settle_steps=30 runs a
   settle-freeze pass first so interpenetrating perceived blocks pop apart
   BEFORE the policy sees the scene — the per-scene displacement is reported;
   +oow_rule=False keeps pushing after a block leaves the workspace, which the
   real robot would do too — the violation is still recorded in metadata)

  # batch: a whole sim suite (lower-bound study)
  python open_loop/solve_in_twin.py task=MoreOpenLoop test=True headless=True \\
      task.env.test_cases.scene_root_dir=test-cases/dataset/selected \\
      task.env.test_cases.difficulty_choice=test-128 num_envs=128 \\
      checkpoint=... +trajectory_out=open_loop/out/test-128

Output: <trajectory_out>/<scene>.json per env + summary.json.
"""
import os
import sys
sys.path.insert(0, os.getcwd())  # run from isaacgymenvs/: rlgames_utils does `from tasks import ...`

import isaacgym  # noqa: F401  (must precede torch)
import tools._net_compat  # noqa: F401  (registers token_set)
import tools._ckpt_compat  # noqa: F401  (map_location='cpu' for cross-cluster ckpts)

import json
import os

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

GRASPABLE_Q_THRESHOLD = 0.9   # same threshold as the More reward terminal


def load_scene_colors(scene_file, num_objects=11):
    """Block colors (11,3 floats in [0,1]) in scene-file order (0 = target)."""
    colors = []
    with open(scene_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 10:
                colors.append([float(parts[1]), float(parts[2]), float(parts[3])])
    assert len(colors) >= num_objects, \
        f"expected {num_objects} objects in {scene_file}, found {len(colors)}"
    return colors[:num_objects]      # More keeps only the first num_objects lines


def scene_files(task_cfg):
    """All NNNNNN.txt scene files, in the ascending numeric order More
    assigns them to envs."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_dir = os.path.join(pkg_root,
                             task_cfg["env"]["test_cases"]["scene_root_dir"],
                             task_cfg["env"]["test_cases"]["difficulty_choice"])
    names = sorted(int(n[:-4]) for n in os.listdir(scene_dir)
                   if n.endswith(".txt") and n[:-4].isdigit())
    assert names, f"no NNNNNN.txt scene file in {scene_dir}"
    return [os.path.join(scene_dir, f"{i:06d}.txt") for i in names]


def make_player(cfg):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    def create_env_thunk(**kwargs):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id, cfg.headless,
            cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kwargs,
        )

    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs:
                    RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU",
                                          "env_creator": create_env_thunk})
    runner = Runner(RLGPUAlgoObserver())
    runner.load(omegaconf_to_dict(cfg.train))
    runner.reset()
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False):
        player.init_rnn()
    return player, obses


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory
    from isaacgymenvs.open_loop import frames

    assert cfg.checkpoint, "pass checkpoint=<path to trained PPO .pth>"
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    out_dir = to_absolute_path(cfg.get("trajectory_out", "open_loop/out/real2sim"))
    os.makedirs(out_dir, exist_ok=True)

    cfg_dict = omegaconf_to_dict(cfg)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    scenes = scene_files(cfg_dict["task"])
    n = min(int(cfg.task.env.numEnvs), len(scenes))   # one env per scene, never wrap
    cfg.task.env.numEnvs = n
    cfg.num_envs = n
    scenes = scenes[:n]
    settle_steps = int(cfg.get("settle_steps", 30))
    oow_rule = bool(cfg.get("oow_rule", True))     # False: keep pushing after an OOW event (still recorded)
    print(f"[open-loop] {n} env(s); scene[0] = {scenes[0]}")

    player, obses = make_player(cfg)
    env = player.env  # MoreOpenLoop (attribute access proxied by RLGPUEnv)
    dev = env.reset_buf.device
    env.reset_idx(torch.arange(n, device=dev))
    # Phase marker for the camera overlay: the real cameras are already rolling
    # while this runs, so the video has to say what the wait is for. Marked HERE
    # rather than before launch so it separates simulator startup from the solve.
    phase_dir = cfg.get("phase_dir", None)
    if phase_dir:
        from isaacgymenvs.open_loop import recorder_io
        recorder_io.mark(str(phase_dir), "solve", f"PPO rollout, {n} scene(s)")

    settle_mm = torch.zeros(n, device=dev)
    if settle_steps > 0:
        settle_mm = 1000.0 * env.settle(settle_steps)
        print(f"[open-loop] settle-freeze ({settle_steps} steps): max block displacement "
              f"mean {settle_mm.mean():.1f} mm, max {settle_mm.max():.1f} mm"
              + (f"  <- {int((settle_mm > 20).sum())} scene(s) moved >2 cm: perception inconsistent"
                 if (settle_mm > 20).any() else ""))
    obses = player.env_reset(player.env)
    env.clear_recording()
    env.enable_physics_recording(bool(cfg.get("record_physics", True)))

    solved = torch.zeros(n, dtype=torch.bool, device=dev)
    oow = torch.zeros(n, dtype=torch.bool, device=dev)
    end_step = torch.zeros(n, dtype=torch.long, device=dev)
    max_steps = int(env.max_episode_length) - 3
    for step in range(max_steps):
        action = player.get_action(obses, is_deterministic=True)
        obses, _, _, _ = player.env_step(player.env, action)
        q = env.grasp_q_parallel_values
        live = ~env.frozen
        viol = env.oow_violation()
        oow |= live & viol
        new_oow = live & viol if oow_rule else torch.zeros_like(live)
        new_solved = live & ~new_oow & (q > GRASPABLE_Q_THRESHOLD)
        done = new_oow | new_solved
        if done.any():
            solved |= new_solved
            end_step[done] = step + 1
            env.freeze(done.nonzero(as_tuple=False).squeeze(-1))
        if env.frozen.all():
            break
    end_step[~env.frozen] = max_steps

    summary = []
    for i in range(n):
        rec = env.recording_of(i)
        traj = OpenLoopTrajectory(
            scene_file=scenes[i], checkpoint=str(cfg.checkpoint),
            metadata={"task": cfg.task_name, "seed": int(cfg.seed), "env": i,
                      "graspable_q_threshold": GRASPABLE_Q_THRESHOLD,
                      "sim_workspace_limits": frames.SIM_WORKSPACE_LIMITS.tolist(),
                      "real_workspace_limits": frames.REAL_WORKSPACE_LIMITS.tolist(),
                      "pixel_size": frames.PIXEL_SIZE})
        traj.record_start(rec["start_eef"] if rec["start_eef"] is not None
                          else env.gripper_pos[i].tolist())
        with open(scenes[i]) as scene_input:
            traj.scene_text = scene_input.read()
        for s in rec["steps"]:
            traj.record_step(s["t"], s["eef"], s["action"], s["grasp_q"],
                             obj_xy=s.get("obj_xy"))
        for w in rec["waypoints"]:
            traj.record_waypoints(w["t"], w["action"], w["wp1"], w["wp2"])
        if rec["physics"] is not None:
            traj.set_physics(rec["physics"])
        traj.final_target = list(env.target_pose(i))
        ok = bool(solved[i])
        if ok:
            g = env.compute_final_grasp(i, load_scene_colors(scenes[i], env.num_objects))
            # The x16 post-check can veto every orientation the in-loop q
            # accepted; it then returns q=0 at pixel (0, 0), which is not a
            # grasp. Leave the record empty: the executor re-senses the real
            # scene for the grasp anyway (demo38 aborted on the bogus point).
            if g is not None and g["q"] > 0:
                x_sim, y_sim = frames.pix_to_sim(g["px"], g["py"])
                traj.record_grasp(x_sim, y_sim, g["px"], g["py"], g["rotation_idx"], g["q"])
            elif g is not None:
                print(f"[open-loop] {os.path.basename(scenes[i])}: in-loop q passed but the "
                      f"x16 post-check found no viable orientation; grasp left to the real re-sense")
        traj.metadata.update({"solved": ok, "oow_violation": bool(oow[i]), "oow_rule": oow_rule,
                              "num_steps": int(end_step[i]),
                              "settle_steps": settle_steps, "settle_disp_mm": float(settle_mm[i]),
                              "final_q": float(env.grasp_q_parallel_values[i])})
        name = os.path.splitext(os.path.basename(scenes[i]))[0]
        path = traj.save(os.path.join(out_dir, f"{name}.json"))
        summary.append({"scene": name, "solved": ok, "oow": bool(oow[i]),
                        "steps": int(end_step[i]), "primitives": len(traj.waypoints),
                        "settle_disp_mm": round(float(settle_mm[i]), 1),
                        "executed_pts": len(traj.executed_path()),
                        "physics_ticks": len(traj.physics["samples"]) if traj.physics else 0,
                        "grasp_q": traj.grasp["q"] if traj.grasp else None, "file": path})
        print(f"[open-loop] {name}: {'SOLVED' if ok else 'FAIL'}"
              f"{' (OOW)' if oow[i] else ''} steps={int(end_step[i])} settle={float(settle_mm[i]):.1f}mm "
              f"primitives={len(traj.waypoints)} executed_pts={summary[-1]['executed_pts']}"
              + (f" grasp q={traj.grasp['q']:.2f} rot={traj.grasp['rotation_idx']}" if traj.grasp else ""))

    n_ok = sum(s["solved"] for s in summary)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"checkpoint": str(cfg.checkpoint), "num_scenes": n, "solved": n_ok,
                   "oow": sum(s["oow"] for s in summary), "scenes": summary}, f, indent=1)
    print(f"[open-loop] twin solve: {n_ok}/{n} solved{' under the OOW rule' if oow_rule else ' (OOW rule off)'}; "
          f"trajectories in {out_dir}")

    if phase_dir:
        # Explicit end marker. Without one this row runs until the robot moves,
        # so it also absorbs process exit AND the executor's own startup.
        from isaacgymenvs.open_loop import recorder_io as _rio
        _rio.mark(str(phase_dir), "solve-done", "")

    if bool(cfg.get("fast_exit", True)):
        # Every output is on disk and flushed by here, so skip atexit/GC/
        # destructors. Measured saving is only ~0.35 s of a 12.95 s process
        # (5.85 s setup, 6.56 s rollout+write, 0.55 s exit) -- Isaac Gym
        # teardown is NOT the expensive part, contrary to the first reading of
        # the demo5 overlay, which had the row swallowing the next process's
        # startup. Kept because it is free. +fast_exit=False to debug shutdown.
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
