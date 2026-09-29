"""Diagnostic rollout: log per-step grasp-Q stats / resets to find why all
envs 'succeed' at ~112 steps in strict eval. Usage: same overrides as
tools/eval_strict.py."""
import os
import sys
sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

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

    env = player.env
    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)
    n = cfg.task.env.numEnvs
    env.reset_idx(torch.arange(n, device=env.reset_buf.device))

    WS = torch.tensor([[0.176, 0.724], [-0.224, 0.224]], device=env.reset_buf.device)

    def oow():
        xy = env.block_state[:, :, :2]
        return ((xy[..., 0] < WS[0, 0]) | (xy[..., 0] > WS[0, 1]) |
                (xy[..., 1] < WS[1, 0]) | (xy[..., 1] > WS[1, 1]))

    init_out = oow()
    seen_viol = torch.zeros(n, dtype=torch.bool, device=env.reset_buf.device)

    print("env0 blocks at t=0 (compare with scene file):")
    for bi in range(env.block_state.shape[1]):
        print(f"  b{bi}: ({env.block_state[0, bi, 0]:.3f}, {env.block_state[0, bi, 1]:.3f}, {env.block_state[0, bi, 2]:.3f})")

    for step in range(130):
        action = player.get_action(obses, is_deterministic=True)
        obses, _, _, _ = player.env_step(player.env, action)
        q = env.grasp_q_parallel_values
        viol = oow() & ~init_out
        new_envs = viol.any(dim=1) & ~seen_viol
        if new_envs.any():
            ids = new_envs.nonzero(as_tuple=False).squeeze(-1)[:4]
            for e in ids.tolist():
                b = viol[e].nonzero(as_tuple=False).squeeze(-1).tolist()
                pos = [f"b{bi}:({env.block_state[e, bi, 0]:.3f},{env.block_state[e, bi, 1]:.3f},{env.block_state[e, bi, 2]:.3f})" for bi in b]
                print(f"  VIOL step {step} env {e}: {pos}")
            seen_viol |= new_envs
        if step % 10 == 0 or (env.reset_buf > 0).sum() > 0:
            g0 = env.gripper_pos[0]
            print(f"step {step:3d} | resets {int((env.reset_buf>0).sum()):3d} "
                  f"| succ {int((env.successes>0).sum()):3d} "
                  f"| Q min {q.min():+.2f} med {q.median():+.2f} max {q.max():+.2f} "
                  f"| Q>0.9 {int((q>0.9).sum()):3d} | Q==-2 {int((q==-2).sum()):3d} "
                  f"| eef0 ({g0[0]:.3f},{g0[1]:.3f},{g0[2]:.3f}) "
                  f"| prog0 {int(env.progress_buf[0])}")
        if step == 74:
            print("env0 all blocks at step 74:")
            for bi in range(env.block_state.shape[1]):
                print(f"  b{bi}: ({env.block_state[0, bi, 0]:.3f}, {env.block_state[0, bi, 1]:.3f}, {env.block_state[0, bi, 2]:.3f})")


if __name__ == "__main__":
    main()
