"""Behavioral diagnostics for MoreTeacher checkpoints (why is a run stuck?).

Deterministic first-episode rollout on the configured test scenes, capturing
per-step Phi terms (g, c, c3, arc, B), actions, and EEF/clutter motion.
Answers: does the policy reach the pile, does any Phi term move, is there an
action-direction bias, and what terminates episodes.

Usage (from isaacgymenvs/):
  python tools/diag_teacher_compare.py task=MoreTeacher test=True headless=True \
      num_envs=64 task.env.test_cases.scene_root_dir=test-cases \
      task.env.test_cases.difficulty_choice=gen-v1/test \
      task.env.robust.randomizeReset=False checkpoint=runs/<...>.pth \
      [+diag_tag=v2ep435] [+diag_budget=450]
"""
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


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    budget = int(cfg.get("diag_budget", 450))
    tag = str(cfg.get("diag_tag", "diag"))
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
    n_act = 16

    tgt0 = env.block_state[:, 0, :2].clone()
    clut0 = env.block_state[:, 1:, :2].clone()
    eef_prev = env.gripper_pos[:, :2].clone()
    eef0 = eef_prev.clone()
    switches = torch.zeros(E, device=dev)
    a_prev = torch.full((E,), -1, dtype=torch.long, device=dev)

    done = torch.zeros(E, dtype=torch.bool, device=dev)
    term = torch.zeros(E, dtype=torch.long, device=dev)   # 1 succ 2 oow 3 oov 4 timeout
    steps = torch.zeros(E, dtype=torch.long, device=dev)
    act_hist = torch.zeros(E, n_act, device=dev)
    path_len = torch.zeros(E, device=dev)
    min_d_tgt = torch.full((E,), 1e9, device=dev)
    first = {}    # Phi terms at t0 (post first compute)
    last = {}     # latest Phi terms per env (frozen at done)

    for step in range(budget):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        live = ~done
        act_hist[live, a[live]] += 1
        switches += (live & (a_prev >= 0) & (a != a_prev)).float()
        a_prev = torch.where(live, a, a_prev)
        obses, _, _, _ = player.env_step(player.env, a)

        eef = env.gripper_pos[:, :2]
        path_len += torch.where(live, (eef - eef_prev).norm(dim=-1),
                                torch.zeros_like(path_len))
        d_tgt = (eef - env.block_state[:, 0, :2]).norm(dim=-1)
        min_d_tgt = torch.where(live, torch.minimum(min_d_tgt, d_tgt), min_d_tgt)
        eef_prev = eef.clone()

        dt = env.diag_terms
        if step == 0:
            first = {k: v.clone() for k, v in dt.items()}
            for k in dt:
                last[k] = dt[k].clone()
        else:
            for k in dt:
                last[k] = torch.where(live, dt[k], last[k])

        q = env.grasp_q_parallel_values
        succ = (q > 0.9) & live
        oow = (env.reset_buf.bool()) & live & ~succ & (q != -2.0) & \
              (env.progress_buf < env.max_episode_length - 1)
        oov = (q == -2.0) & live & ~succ
        tout = (env.progress_buf >= env.max_episode_length - 1) & live & ~succ & ~oov
        for code, m in ((1, succ), (2, oow), (3, oov), (4, tout)):
            term[m] = code
            steps[m] = step + 1
        done |= env.reset_buf.bool() & live
        if bool(done.all()):
            break
    term[~done] = 4
    steps[~done] = budget

    tgt_disp = (env.block_state[:, 0, :2] - tgt0).norm(dim=-1)
    clut_disp = (env.block_state[:, 1:, :2] - clut0).norm(dim=-1).sum(dim=-1)

    probs = act_hist / act_hist.sum(dim=1, keepdim=True).clamp(min=1)
    ent = -(probs * (probs + 1e-9).log()).sum(dim=1)
    top_share = probs.max(dim=1).values
    global_hist = act_hist.sum(dim=0) / act_hist.sum().clamp(min=1)

    names = {1: "success", 2: "oow", 3: "out_of_view", 4: "timeout"}
    print(f"\n=== teacher diagnostics [{tag}] {E} scenes, ckpt {os.path.basename(cfg.checkpoint)} ===")
    counts = {n: int((term == c).sum()) for c, n in names.items()}
    print(f"terminations: {counts}")
    print(f"steps: mean {steps.float().mean():.0f}  median {steps.float().median():.0f}")
    print(f"action entropy/episode: mean {ent.mean():.2f} (max {torch.log(torch.tensor(16.)):.2f}); "
          f"top-action share: mean {top_share.mean():.2f}")
    print("global action histogram: " +
          " ".join(f"{i}:{p:.02f}" for i, p in enumerate(global_hist.tolist()) if p > 0.02))
    print(f"EEF: path len mean {path_len.mean():.2f} m; min dist to target mean "
          f"{min_d_tgt.mean():.3f} m  (<0.15m reached-pile: {int((min_d_tgt < 0.15).sum())}/{E})")
    print(f"displacement: target mean {tgt_disp.mean()*100:.1f} cm; clutter-sum mean {clut_disp.mean()*100:.1f} cm")
    # oscillation: low net/path ratio + high switch rate = dithering in place
    net = (env.gripper_pos[:, :2] - eef0).norm(dim=-1)
    ratio = net / path_len.clamp(min=1e-6)
    sw = switches / steps.float().clamp(min=1)
    print(f"oscillation: net-disp/path-len mean {ratio.mean():.2f} "
          f"(1=straight, ~0=dither); action-switch rate mean {sw.mean():.2f}")
    print("Phi terms (t0 -> end, mean):")
    for k in ("g", "c", "c3", "arc", "B"):
        f0, l0 = first[k].mean(), last[k].mean()
        print(f"  {k:3}: {f0:7.3f} -> {l0:7.3f}   (delta {l0 - f0:+.3f})")
    if E <= 8:
        for e in range(E):
            print(f"  env{e}: {names[int(term[e])]:11} steps {int(steps[e]):3} "
                  f"gQ_end {float(last['g'][e]):.3f}  net/path {float(ratio[e]):.2f} "
                  f"switch {float(sw[e]):.2f}")


if __name__ == "__main__":
    main()
