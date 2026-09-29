"""Replay stored trajectories in the twin with the POLICY OUT OF THE LOOP.

The blind open-loop test that needs no robot: for each trajectory JSON,
build N replicas of its scene in one batched sim — replica 0 nominal (the
exact perceived scene), replicas 1..N-1 with perception-noise pose jitter +
friction/mass domain randomization — and drive every replica's EEF through
the SAME absolute path (default: the `executed` straight-segment path, i.e.
exactly what execute_trajectory.py sends to the UR5e via moveL). Reports:

  nominal_ok     : replaying the recorded motion reproduces the solve
                   (target graspable, no OOW) in the unperturbed twin —
                   a fidelity check of the trajectory representation
  target_err_mm  : nominal replica's final target position vs the solve's
  certificate    : fraction of perturbed replicas ending graspable w/o OOW
                   = predicted blind open-loop success probability (the
                   lockstep certificate of the method description (§3.A))

Run from isaacgymenvs/:

  python open_loop/replay_in_twin.py task=MoreOpenLoop test=True headless=True \\
      +replay_dir=open_loop/out/test-128-ep270 +replay_n=16 \\
      [+replay_mode=joint|cartesian_feedforward|cartesian|physics|executed|dense|commanded] \\
      [+noise_pos=0.003 +noise_yaw=2.0 +friction_lo=0.2 +friction_hi=0.5] \\
      [+replay_out=open_loop/out/test-128-ep270/replay.json]

No checkpoint is needed (the policy is bypassed); rl_games is not used.
Version-3 joint and cartesian_feedforward modes preserve the recorded motor
commands and physics clock. cartesian is a pose-only tracking diagnostic.
physics follows all recorded XYZ positions with the legacy arrival gate.
Timed modes report EEF tracking errors against the physics trace and do not
add the legacy extra settle step to the nominal comparison. Each timed
replica is scored at its own completion; incomplete replays cannot succeed.
"""
import os
import sys
sys.path.insert(0, os.getcwd())  # run from isaacgymenvs/: rlgames_utils does `from tasks import ...`

import isaacgym  # noqa: F401
import tools._net_compat  # noqa: F401  (registers token_set)
import tools._ckpt_compat  # noqa: F401  (map_location='cpu' for cross-cluster ckpts)

import glob
import json
import os
import shutil
import tempfile

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

Q_THRESH = 0.9


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory

    replay_dir = to_absolute_path(str(cfg.get("replay_dir", "open_loop/out/real2sim")))
    N = int(cfg.get("replay_n", 16))
    mode = str(cfg.get("replay_mode", "executed"))
    max_m = int(cfg.get("replay_max", 0))
    skip = int(cfg.get("replay_skip", 0))       # chunking: skip the first k solved trajectories
    noise_pos = float(cfg.get("noise_pos", 0.003))
    noise_yaw = float(cfg.get("noise_yaw", 2.0))
    fr = (float(cfg.get("friction_lo", 0.2)), float(cfg.get("friction_hi", 0.5)))
    budget = int(cfg.get("replay_budget", 900))
    if N < 1 or budget < 1:
        raise ValueError("replay_n and replay_budget must both be positive")
    out_json = to_absolute_path(str(cfg.get("replay_out", os.path.join(replay_dir, f"replay_{mode}.json"))))

    files = sorted(f for f in glob.glob(os.path.join(replay_dir, "*.json"))
                   if not os.path.basename(f).startswith(("summary", "replay")))
    trajs = [OpenLoopTrajectory.load(f) for f in files]
    keep = [i for i, t in enumerate(trajs) if t.metadata.get("solved")][skip:]
    if max_m:
        keep = keep[:max_m]
    files, trajs = [files[i] for i in keep], [trajs[i] for i in keep]
    M = len(trajs)
    assert M, f"no SOLVED trajectory JSON in {replay_dir}"

    # replicate scene i N times into a temp suite: env i*N+j loads scene i
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = tempfile.mkdtemp(prefix="_replay_", dir=os.path.join(pkg, "test-cases"))
    for i, t in enumerate(trajs):
        for j in range(N):
            t.write_scene(os.path.join(tmp, f"{i * N + j:06d}.txt"))
    cfg.task.env.numEnvs = M * N
    cfg.num_envs = M * N
    cfg.task.env.test_cases.scene_root_dir = "test-cases"
    cfg.task.env.test_cases.difficulty_choice = os.path.basename(tmp)

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
                            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                            cfg.headless, cfg.multi_gpu, cfg.capture_video,
                            cfg.force_render, cfg)
    dev = env.reset_buf.device
    E = M * N
    settle_steps = int(cfg.get("settle_steps", trajs[0].metadata.get("settle_steps", 30)))
    if settle_steps > 0:                     # same settle-freeze as the solve
        env.settle(settle_steps)
    perturbed = [e for e in range(E) if e % N != 0]
    if perturbed:
        env.perturb(perturbed, pos_noise=noise_pos, yaw_noise_deg=noise_yaw,
                    friction_range=fr, seed=int(cfg.seed))
    env.reset_idx(torch.arange(E, device=dev))
    timed = mode in ("joint", "cartesian", "cartesian_feedforward")
    if timed:
        env.set_timed_replay([trajs[e // N] for e in range(E)], mode=mode)
        env.enable_physics_recording(include_replay=True)
        paths = [trajs[e // N].physics_path() for e in range(E)]
    else:
        paths = [trajs[e // N].path(mode) for e in range(E)]
        env.set_replay(paths)

    zero = torch.zeros(E, dtype=torch.long, device=dev)
    finished = torch.zeros(E, dtype=torch.bool, device=dev)
    endpoint_q = torch.zeros(E, device=dev)
    endpoint_xy = torch.zeros((E, 2), device=dev)
    endpoint_steps = torch.zeros(E, dtype=torch.long, device=dev)
    violated = torch.zeros(E, dtype=torch.bool, device=dev)
    for step in range(budget):
        live = ~finished if timed else torch.ones(E, dtype=torch.bool, device=dev)
        env.step(zero)                                   # actions ignored on replay envs
        violated |= live & env.oow_violation()
        endpoint_q[live] = env.grasp_q_parallel_values[live]
        endpoint_xy[live] = env.block_state[live, 0, :2]
        endpoint_steps[live] = step + 1
        finished |= env.replay_done
        if env.replay_done.all():
            break
    if not timed:
        env.step(zero)                                   # legacy end-settle convention
        endpoint_q[:] = env.grasp_q_parallel_values
        endpoint_xy[:] = env.block_state[:, 0, :2]
        violated |= env.oow_violation()
        finished |= env.replay_done
    ok = finished & (endpoint_q > Q_THRESH) & ~violated

    rows = []
    print(f"\n=== twin replay ({mode} path, {M} trajectories x {N} replicas, "
          f"noise {noise_pos * 1000:.0f}mm/{noise_yaw}deg, friction U{fr}, "
          f"{step + 1} steps) ===")
    for i, t in enumerate(trajs):
        s, e = i * N, (i + 1) * N
        tx, ty = endpoint_xy[s].tolist()
        # A batched solver may continue simulating shorter solved scenes.
        # The last recorded tick is the reference endpoint for timed replay.
        ft = t.physics["samples"][-1]["block_state"][0] if timed else t.final_target
        ft = ft or [float("nan")] * 3
        err_mm = 1000.0 * ((tx - ft[0]) ** 2 + (ty - ft[1]) ** 2) ** 0.5
        cert = float(ok[s + 1:e].float().mean()) if N > 1 else float("nan")
        rows.append({"file": os.path.basename(files[i]), "scene": t.scene_file,
                     "nominal_ok": bool(ok[s]), "nominal_q": float(endpoint_q[s]),
                     "nominal_done": bool(finished[s]), "nominal_steps": int(endpoint_steps[s]),
                     "completed_replicas": int(finished[s:e].sum()),
                     "nominal_oow": bool(violated[s]), "target_err_mm": round(err_mm, 1),
                     "certificate": round(cert, 3),
                     "oow_frac": round(float(violated[s + 1:e].float().mean()), 3) if N > 1 else None,
                     "path_pts": len(paths[s]), "solve_steps": t.metadata.get("num_steps")})
        if timed:
            from isaacgymenvs.open_loop.trace_metrics import compare_physics
            actual = env.recording_of(s)["physics"]
            rows[-1]["tracking"] = compare_physics(t.physics, actual)
        r = rows[-1]
        print(f"  {r['file']}: nominal={'OK ' if r['nominal_ok'] else 'FAIL'} q={r['nominal_q']:.2f} "
              f"target_err={r['target_err_mm']:.1f}mm  certificate={r['certificate']:.2f} "
              f"oow={r['oow_frac']}  pts={r['path_pts']}")
    n_nom = sum(r["nominal_ok"] for r in rows)
    certs = [r["certificate"] for r in rows if r["certificate"] == r["certificate"]]
    print(f"nominal reproduction: {n_nom}/{M};  mean certificate: "
          f"{sum(certs) / len(certs):.3f}" if certs else f"nominal reproduction: {n_nom}/{M}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({"mode": mode, "n_replicas": N, "noise_pos": noise_pos, "noise_yaw": noise_yaw,
               "friction": fr, "rows": rows}, open(out_json, "w"), indent=1)
    print(f"saved: {out_json}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
