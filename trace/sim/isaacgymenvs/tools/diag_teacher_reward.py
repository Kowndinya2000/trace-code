"""Random-action sanity check of the MoreTeacher reward (signs/scales/terminals).
Run: python tools/diag_teacher_reward.py task=MoreTeacher num_envs=8 headless=True"""
import os
import sys
sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401
import hydra
from omegaconf import DictConfig
import torch

from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed)
    env = isaacgymenvs.make(
        cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
        cfg.sim_device, cfg.rl_device, cfg.graphics_device_id, cfg.headless,
        cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)
    env.reset()
    n = cfg.task.env.numEnvs
    for step in range(60):
        a = torch.randint(0, 16, (n,), device=env.device)
        env.step(a)
        if step % 10 == 0 or env.reset_buf.any():
            phi, bdist = env._phi_terms()
            r = env.rew_buf
            print(f"step {step:3d} | r mean {r.mean():+.4f} min {r.min():+.4f} "
                  f"max {r.max():+.4f} | Phi mean {phi.mean():+.3f} "
                  f"[{phi.min():+.3f},{phi.max():+.3f}] | "
                  f"minBdist {bdist.min():+.4f} | resets {int(env.reset_buf.sum())} "
                  f"| succ {int(env.successes.sum())}")


if __name__ == "__main__":
    main()
