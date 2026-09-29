"""STRICT single-episode evaluation — one scene, one attempt, one verdict.

Motivation (Aug 2026 audit): eval_test.py + MoreTest report "success rate"
from a `success_sign` that (a) latches across trials (never cleared in
reset_idx) and (b) counts within-trial retries after auto-reset. Both
inflate the number. This harness records each env's FIRST episode outcome
only: success iff the target became graspable before the first
termination (timeout / out-of-view). No retries, no latching.

Also writes a per-scene CSV (scene id -> outcome, steps) so failures can be
inspected individually — the input the lockstep certifier work needs.

Usage (from isaacgymenvs/, pmbs env, LD_LIBRARY_PATH set):
  python tools/eval_strict.py task=MoreTest test=True headless=True num_envs=128 \
      checkpoint=runs/<...>/nn/<ckpt>.pth [+strict_out=tools/qa_out/strict_ep915.csv]
"""
import os
import sys
sys.path.insert(0, os.getcwd())  # run from isaacgymenvs/: rlgames_utils does `from tasks import ...`

import isaacgym  # noqa: F401
import tools._net_compat  # noqa: F401  (registers token_set)
import tools._ckpt_compat  # noqa: F401  (map_location='cpu' for cross-cluster ckpts)

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import torch
import numpy as np

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    assert cfg.checkpoint, "pass checkpoint=<path>"
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    out_csv = to_absolute_path(cfg.get("strict_out", "tools/qa_out/strict_eval.csv"))

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
    if getattr(player, "is_rnn", False):
        player.init_rnn()

    n = cfg.task.env.numEnvs
    device = env.reset_buf.device
    env.reset_idx(torch.arange(n, device=device))

    recorded = torch.zeros(n, dtype=torch.bool, device=device)
    ep_return = torch.zeros(n, device=device)      # undiscounted episode return
    outcome = torch.zeros(n, dtype=torch.bool, device=device)   # success flag
    end_step = torch.zeros(n, dtype=torch.long, device=device)

    # Out-of-Workspace rule from the IROS26 paper: if ANY object leaves the
    # 55x45 cm workspace at any point, the episode is a failure regardless
    # of later grasp success. (MoreTest/eval_test never enforced this.)
    WS = torch.tensor([[0.176, 0.724], [-0.224, 0.224]], device=device)

    def oow_mask():
        xy = env.block_state[:, :, :2]                      # (n, blocks, 2)
        return ((xy[..., 0] < WS[0, 0]) | (xy[..., 0] > WS[0, 1]) |
                (xy[..., 1] < WS[1, 0]) | (xy[..., 1] > WS[1, 1]))

    initially_out = oow_mask()          # boundary-padded dummies are exempt
    violated = torch.zeros(n, dtype=torch.bool, device=device)
    first_viol_step = torch.full((n,), -1, dtype=torch.long, device=device)

    max_steps = int(env.max_episode_length) + 5
    for step in range(max_steps):
        action = player.get_action(obses, is_deterministic=True)
        obses, _, _, _ = player.env_step(player.env, action)
        ep_return += env.rew_buf.detach() * (~recorded).float()   # freeze at termination
        new_viol = ((oow_mask() & ~initially_out).any(dim=1)) & (~recorded) & (~violated)
        first_viol_step[new_viol] = step + 1
        violated |= new_viol
        # After compute_reward: reset_buf marks envs terminating THIS step;
        # env.successes==1 exactly for the graspable ones (set before the
        # out-of-view / timeout reset branches in compute_more_reward_jit).
        done_now = (env.reset_buf > 0) & (~recorded)
        if done_now.any():
            outcome[done_now] = env.successes[done_now] > 0
            end_step[done_now] = step + 1
            recorded |= done_now
        if recorded.all():
            break

    if getattr(env, "vid_enabled", False) and getattr(env, "_vid_frames", None):
        env._flush_video()              # partial window at eval end

    # anything never terminated within budget counts as failure at max_steps
    end_step[~recorded] = max_steps
    n_done = int(recorded.sum())
    raw_succ = int(outcome.sum())
    strict = outcome & (~violated)
    n_succ = int(strict.sum())
    succ_steps = end_step[strict].float()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("env,graspable,oow_violation,first_viol_step,success,end_step\n")
        for i in range(n):
            f.write(f"{i},{int(outcome[i])},{int(violated[i])},{int(first_viol_step[i])},"
                    f"{int(strict[i])},{int(end_step[i])}\n")

    print("\n=== STRICT single-episode evaluation ===")
    print(f"checkpoint : {cfg.checkpoint}")
    print(f"scenes     : {n} (terminated: {n_done})")
    print(f"graspable  : {raw_succ}/{n} = {100.0*raw_succ/n:.2f}%  (no OOW rule)")
    print(f"OOW violat.: {int(violated.sum())}/{n} episodes pushed an object "
          f"out of the 55x45cm workspace")
    if violated.any():
        v = first_viol_step[violated].float()
        early = int((first_viol_step[violated] <= 5).sum())
        print(f"first violation step: mean {v.mean():.1f}, median {v.median():.0f}, "
              f"min {int(v.min())}, max {int(v.max())} "
              f"({early} within the first 5 steps = settling explosions)")
    print(f"RETURN     : mean {float(ep_return.mean()):.3f}  "
          f"succ {float(ep_return[strict].mean()) if n_succ else float('nan'):.3f}  "
          f"fail {float(ep_return[~strict].mean()) if n_succ < n else float('nan'):.3f}")
    print(f"SUCCESS    : {n_succ}/{n} = {100.0*n_succ/n:.2f}%  "
          f"(graspable AND no violation — paper metric)")
    if n_succ:
        print(f"steps (succ): mean {succ_steps.mean():.1f}, "
              f"median {succ_steps.median():.0f}, max {int(succ_steps.max())}")
    fails = [str(i) for i in range(n) if not strict[i]]
    print(f"failed envs: {', '.join(fails) if fails else 'none'}")
    print(f"per-scene  : {out_csv}")


if __name__ == "__main__":
    main()
