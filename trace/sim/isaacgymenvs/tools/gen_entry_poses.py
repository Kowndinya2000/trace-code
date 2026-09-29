"""Generate joint configurations for the K=8 workspace entry anchors.

Drives one env per anchor with damped-least-squares IK to the anchor xy at
the home cruise height, then records the converged UR5e joint angles.
Output feeds MoreRobust's random-entry reset (robust.entryAnchors).

Usage (from isaacgymenvs/):
  python tools/gen_entry_poses.py task=MoreTeacher test=True headless=True num_envs=8
Writes cfg/entry_poses.json.
"""
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401
import hydra
from omegaconf import DictConfig
import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# workspace rect (sim frame, matches WS_X/WS_Y in more_robust.py)
X0, X1, Y0, Y1 = 0.276, 0.724, -0.224, 0.224
ANCHORS = [  # 4 corners + 4 edge midpoints; index 0 = classic home side (W mid)
    (X0, 0.5 * (Y0 + Y1)), (X1, 0.5 * (Y0 + Y1)),
    (0.5 * (X0 + X1), Y0), (0.5 * (X0 + X1), Y1),
    (X0, Y0), (X0, Y1), (X1, Y0), (X1, Y1),
]


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    cfg.task.env.numEnvs = len(ANCHORS)
    cfg.num_envs = len(ANCHORS)
    cfg.task.env.robust.randomizeReset = False
    cfg.task.env.videoLog.enabled = False
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env = isaacgymenvs.make(
        cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
        cfg.sim_device, cfg.rl_device, cfg.graphics_device_id, cfg.headless,
        cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)
    env.reset()

    from isaacgym import gymtorch
    E = len(ANCHORS)
    dev = env.device
    gidx = env.gripper_idxs.long()
    rb_states = gymtorch.wrap_tensor(env.gym.acquire_rigid_body_state_tensor(env.sim))
    env.rb_states = rb_states
    env.refresh_env_tensors()
    z = float(env.rb_states[gidx, 2].mean())
    tgt = torch.tensor([[a[0], a[1], z] for a in ANCHORS], device=dev)

    for it in range(600):
        env.refresh_env_tensors()
        eef = env.rb_states[gidx, :3]
        err = tgt - eef
        if it % 100 == 0:
            print(f"iter {it}: xy err (mm) "
                  + " ".join(f"{e:.1f}" for e in (err[:, :2].norm(dim=1) * 1000).tolist()))
        dpose = torch.zeros(E, 6, 1, device=dev)
        dpose[:, :3, 0] = err
        u = env.control_ik(dpose)
        env.ur5e_dof_targets[:, :6] = env.ur5e_dof_pos[:, :6] + u
        env.ur5e_dof_targets[:, 6:] = 0.0
        env.gym.set_dof_position_target_tensor(
            env.sim, gymtorch.unwrap_tensor(env.ur5e_dof_targets))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

    env.refresh_env_tensors()
    eef = env.rb_states[gidx, :3]
    err_mm = ((tgt[:, :2] - eef[:, :2]).norm(dim=1) * 1000).tolist()
    out = {"cruise_z": z, "anchors": []}
    for i, a in enumerate(ANCHORS):
        reach = err_mm[i] < 5.0
        out["anchors"].append({
            "xy": [a[0], a[1]], "reachable": bool(reach),
            "err_mm": round(err_mm[i], 2),
            "joints": [round(float(j), 6) for j in env.ur5e_dof_pos[i, :6].tolist()],
        })
        print(f"anchor {i} {a}: err {err_mm[i]:.1f} mm  "
              f"{'OK' if reach else 'UNREACHABLE'}")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "cfg", "entry_poses.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
