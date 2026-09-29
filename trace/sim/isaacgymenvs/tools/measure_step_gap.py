"""Measure commanded-vs-executed EEF displacement per RL step.

Each MoreTeacher action commands a fixed 2 cm primitive; with
controlFrequencyInv = 1 the arm only realises a fraction of it, which inflates
the effective horizon and breaks open-loop replay fidelity. This script runs a
checkpoint deterministically and reports, per control frequency, the realised
displacement distribution and how many RL steps a solve costs.

Usage (from isaacgymenvs/), one process per setting:
  python tools/measure_step_gap.py task=MoreTeacher test=True headless=True \
      num_envs=64 task.env.test_cases.scene_root_dir=test-cases \
      task.env.test_cases.difficulty_choice=gen-v2/test \
      task.env.robust.randomizeReset=False task.env.robust.domainRand=False \
      task.env.videoLog.enabled=False checkpoint=<ckpt> \
      task.env.controlFrequencyInv=1 +gap_steps=200 +gap_tag=cfi1
Repeat with controlFrequencyInv=3 and 5 (Isaac Gym cannot rebuild a sim
in-process, so each setting needs its own process).
"""
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401
import tools._net_compat  # noqa: F401  (registers token_set)
import tools._ckpt_compat  # noqa: F401  (map_location='cpu' for cross-cluster ckpts)
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# commanded metres per primitive; read from the task (env.pushDistanceM)
COMMANDED = 0.02   # overwritten at runtime from the env


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    steps = int(cfg.get("gap_steps", 200))
    tag = str(cfg.get("gap_tag", "gap"))
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    def thunk(**kw):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id, cfg.headless,
            cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kw)

    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver())
    runner.load(omegaconf_to_dict(cfg.train))
    runner.reset()
    player = runner.create_player()
    player.restore(cfg.checkpoint)

    env = player.env
    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False):
        player.init_rnn()

    E = env.reset_buf.shape[0]
    dev = env.reset_buf.device
    global COMMANDED
    COMMANDED = float(getattr(env, "push_total", COMMANDED))  # not a hardcoded 2cm
    print(f"commanded primitive: {COMMANDED*1000:.1f} mm")
    prev = env.gripper_pos[:, :2].clone()
    disp, tilt, jvel = [], [], []
    plan_del, dropped, accepted, eef_depth = [], 0, 0, []
    was_active = env.plan_active.clone()
    prev_seq = env.plan_delivery_seq.clone()
    done = torch.zeros(E, dtype=torch.bool, device=dev)
    solve_steps = torch.full((E,), -1, dtype=torch.long, device=dev)

    for t in range(steps):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        obses, _, _, _ = player.env_step(player.env, a)
        cur = env.gripper_pos[:, :2]
        d = (cur - prev).norm(dim=-1)
        disp.append(torch.where(done, torch.full_like(d, float("nan")), d))
        prev = cur.clone()
        # EEF tilt: angle between the tool z-axis and world -z (0 = perpendicular
        # to the table). The live controller feeds orn_err = 0 to IK, so
        # orientation is unconstrained and can drift with larger motions.
        q = env.gripper_rot                      # (E,4) xyzw
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        zax = torch.stack([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)], dim=1)
        # deviation from vertical, sign-agnostic: the URDF tool z may point
        # wrist-ward, so an aligned tool reads 0 or 180 deg. Report the angle to
        # the nearest vertical => 0 means perpendicular to the table either way.
        cosang = (zax * torch.tensor([0., 0., -1.], device=dev)).sum(dim=1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cosang))
        ang = torch.minimum(ang, 180.0 - ang)
        tilt.append(torch.where(done, torch.full_like(ang, float("nan")), ang))
        jv = env.ur5e_dof_vel[:, :6].abs().amax(dim=1)
        jvel.append(torch.where(done, torch.full_like(jv, float("nan")), jv))
        # EEF containment by DEPTH, not a binary test: the clamp holds the
        # gripper on the boundary where PD noise crosses it by microns, so a
        # zero-tolerance test reports 'outside' for contact and for a 10cm
        # escape alike. Depth distinguishes them.
        wx, wy = getattr(env, "ws_x", (0.276, 0.724)), getattr(env, "ws_y", (-0.224, 0.224))
        ex, ey = cur[:, 0], cur[:, 1]
        depth = torch.maximum(
            torch.maximum(wx[0] - ex, ex - wx[1]),
            torch.maximum(wy[0] - ey, ey - wy[1])).clamp(min=0.0)
        eef_depth.append(torch.where(done, torch.full_like(depth, float("nan")), depth))
        # per-PLAN delivery: when a plan completes, compare the realised
        # displacement from its start pose against the commanded 2 cm. This is
        # the fidelity number that matters when a plan spans several RL steps
        # (reach_gated), unlike per-RL-step displacement.
        # completions are recorded env-side inside the substep loop, so a plan
        # that starts AND finishes within one RL step is still counted
        now_active = env.plan_active
        newly = (env.plan_delivery_seq != prev_seq) & ~done
        if torch.any(newly):
            plan_del.append(env.plan_delivery_last[newly].detach().clone())
        prev_seq = env.plan_delivery_seq.clone()
        # dropped actions: an action arriving while a plan is still running is
        # ignored (pre_physics_step only accepts envs with plan_active False)
        dropped += int((was_active & ~done).sum())
        accepted += int((~was_active & ~done).sum())
        was_active = now_active.clone()

        succ = (env.grasp_q_parallel_values > 0.9) & ~done
        solve_steps[succ] = t + 1
        done |= env.reset_buf.bool()

    D = torch.stack(disp)                      # (T, E)
    v = D[~torch.isnan(D)]
    TL = torch.stack(tilt); tl = TL[~torch.isnan(TL)]
    JV = torch.stack(jvel); jv = JV[~torch.isnan(JV)]
    ED = torch.stack(eef_depth); ed = ED[~torch.isnan(ED)]
    cfi = int(env.control_freq_inv)
    q = lambda p: float(v.quantile(p))
    solved = solve_steps[solve_steps > 0]
    out = {
        "tag": tag, "control_freq_inv": cfi, "envs": E, "rl_steps": steps,
        "commanded_mm": COMMANDED * 1000,
        "executed_mm_mean": float(v.mean()) * 1000,
        "executed_mm_median": q(0.5) * 1000,
        "executed_mm_p90": q(0.9) * 1000,
        "efficiency_pct": 100 * float(v.mean()) / COMMANDED,
        "efficiency_pct_median": 100 * q(0.5) / COMMANDED,
        "sim_seconds_per_rl_step": cfi / 60.0,
        "eef_tilt_deg_mean": float(tl.mean()),
        "eef_tilt_deg_p95": float(tl.quantile(0.95)),
        "eef_tilt_deg_max": float(tl.max()),
        "joint_vel_max_rad_s": float(jv.max()),
        "joint_vel_p95_rad_s": float(jv.quantile(0.95)),
        "eef_outside_depth_max_mm": float(ed.max()) * 1000,
        "eef_outside_depth_p95_mm": float(ed.quantile(0.95)) * 1000,
        "eef_outside_1mm_pct": 100 * float((ed > 0.001).float().mean()),
        "eef_on_boundary_pct": 100 * float((ed > 0.0).float().mean()),
        # mean, not median: once the solvers terminate, the recorded plans are
        # dominated by stuck envs and the median collapses toward 0
        "plan_delivery_mm_mean": None if not plan_del else float(torch.cat(plan_del).mean()) * 1000,
        "plan_delivery_pct": None if not plan_del else 100 * float(torch.cat(plan_del).mean()) / COMMANDED,
        "plan_delivery_mm_median": None if not plan_del else float(torch.cat(plan_del).median()) * 1000,
        "n_plans_completed": int(sum(x.numel() for x in plan_del)),
        "dropped_action_pct": 100.0 * dropped / max(1, dropped + accepted),
        "n_solved": int((solve_steps > 0).sum()),
        "median_solve_steps": None if not len(solved) else int(solved.median()),
    }
    if out["n_plans_completed"] == 0:
        print("WARNING: no completed plans sampled — per-plan delivery is n/a")
    print("\nNOTE: median/p90 are computed only over steps before an env "
          "terminated; when most envs solve early the surviving sample is "
          "dominated by stuck envs and the median becomes bimodal. Prefer "
          "efficiency_pct (mean) and plan_delivery_pct for comparisons.")
    print("\n=== commanded vs executed EEF displacement ===")
    for k, val in out.items():
        print(f"  {k:26} {val}")
    os.makedirs("tools/qa_out", exist_ok=True)
    path = f"tools/qa_out/step_gap_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
