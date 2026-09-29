"""Thorough grasp-network A/B across camera configs on MID-EPISODE states.

Answers: does the 320px/0.64m wide camera (2mm/px preserved) give the same
grasp-Q as the original 224px/0.448m camera on the states that matter —
partially solved scenes, displaced targets, near-edge cases?

Modes (chunk-per-process; Isaac Gym cannot rebuild cameras in-process):
  record : roll a competent checkpoint on the first N test scenes at the
           production camera, snapshot full block states every SNAP steps
           -> tools/qa_out/ab_states.pt
  measure: for a given camera config, teleport every snapshot back in and
           re-compute Q through the exact training render path
           -> tools/qa_out/ab_q_<tag>.pt

Usage (from isaacgymenvs/):
  python tools/gn_ab_test.py ... +ab_mode=record checkpoint=<v1 ckpt> num_envs=64
  python tools/gn_ab_test.py ... +ab_mode=measure +ab_tag=w320 num_envs=64 \
      task.env.videoLog.cameraRes=320 task.env.videoLog.cameraFov=0.036829
  python tools/gn_ab_test.py ... +ab_mode=measure +ab_tag=n224 num_envs=64 \
      task.env.videoLog.cameraRes=224 task.env.videoLog.cameraFov=0.0
  (always add: task=MoreTeacher test=True headless=True
   task.env.test_cases.scene_root_dir=test-cases
   task.env.test_cases.difficulty_choice=gen-v1/test
   task.env.robust.randomizeReset=False task.env.robust.domainRand=False
   task.env.videoLog.enabled=False)
Then compare with +ab_mode=report.
"""
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

SNAP = 15
QA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tools", "qa_out")


def q_now(env):
    """Exact replica of the training-time render->GN path (more.py post_physics)."""
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)
    try:
        cam_depth = torch.stack(env.cam_depth_tensors, dim=0)
        cam_segm = torch.stack(env.cam_segm_tensors, dim=0)
        depth_clean = torch.where(torch.isneginf(cam_depth),
                                  torch.zeros_like(cam_depth), cam_depth)
        # central 224px reference window — must mirror more.py post_physics
        H = depth_clean.shape[1]
        c0 = max((H - 224) // 2, 0)
        central = depth_clean[:, c0:c0 + 224, c0:c0 + 224]
        depth_shifted = depth_clean - central.amin(dim=(1, 2), keepdim=True)
        q = env.mcts_helper.grasp_prob_B(depth_shifted, cam_segm, tile_length=112)
    finally:
        env.gym.end_access_image_tensors(env.sim)
    return q.clone()


def make_env_and_player(cfg, need_player):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    def thunk(**kw):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id, cfg.headless,
            cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kw)

    if not need_player:
        return thunk(), None
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver())
    runner.load(omegaconf_to_dict(cfg.train))
    runner.reset()
    player = runner.create_player()
    player.restore(cfg.checkpoint)
    if getattr(player, "is_rnn", False):
        pass  # init after first reset
    return player.env, player


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    mode = str(cfg.get("ab_mode"))
    os.makedirs(QA, exist_ok=True)

    if mode == "report":
        st = torch.load(os.path.join(QA, "ab_states.pt"))
        qa = torch.load(os.path.join(QA, "ab_q_w320.pt")).flatten()
        qb = torch.load(os.path.join(QA, "ab_q_n224.pt")).flatten()
        tgt = st["states"][:, :, 0, 0:2].reshape(-1, 2)          # (T*E, 2)
        n = min(len(qa), len(qb))
        qa, qb, tgt = qa[:n], qb[:n], tgt[:n]
        d = (qa - qb).abs()
        cx, cy = 0.6, 0.05
        edge = ((tgt[:, 0] - cx).abs() > 0.20) | ((tgt[:, 1] - cy).abs() > 0.17)
        thr = ((qa > 0.9) != (qb > 0.9))
        print(f"pairs: {n}   mean|dQ| {d.mean():.4f}  p95 {d.quantile(0.95):.4f} "
              f"max {d.max():.4f}")
        print(f"threshold(0.9) disagreements: {int(thr.sum())} "
              f"({100*thr.float().mean():.2f}%)")
        print(f"near-edge subset ({int(edge.sum())} states): mean|dQ| "
              f"{d[edge].mean():.4f}  max {d[edge].max():.4f}  "
              f"thr-disagree {int((thr & edge).sum())}")
        print(f"central subset: mean|dQ| {d[~edge].mean():.4f}  "
              f"max {d[~edge].max():.4f}  thr-disagree {int((thr & ~edge).sum())}")
        hi = torch.argsort(d, descending=True)[:10]
        for i in hi.tolist():
            print(f"  worst: state {i}  q320={qa[i]:.3f} q224={qb[i]:.3f} "
                  f"tgt=({tgt[i,0]:.2f},{tgt[i,1]:.2f})")
        return

    cfg.task.env.robust.randomizeReset = False
    cfg.task.env.robust.domainRand = False
    set_np_formatting()
    cfg.seed = set_seed(42, False)

    if mode == "record":
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        env, player = make_env_and_player(cfg, True)
        obses = player.env_reset(player.env)
        player.get_batch_size(obses, 1)
        if getattr(player, "is_rnn", False):
            player.init_rnn()
        E = env.reset_buf.shape[0]
        snaps = []
        for step in range(450):
            a = player.get_action(obses, is_deterministic=True)
            a = a.squeeze(-1) if a.dim() > 1 else a
            obses, _, _, _ = player.env_step(player.env, a)
            if step % SNAP == 0:
                snaps.append(env.block_state.clone().cpu())   # (E, O, 13)
        torch.save({"states": torch.stack(snaps)},            # (T, E, O, 13)
                   os.path.join(QA, "ab_states.pt"))
        print(f"recorded {len(snaps)} rounds x {E} envs")
        return

    # measure
    tag = str(cfg.get("ab_tag"))
    env, _ = make_env_and_player(cfg, False)
    env.reset()
    st = torch.load(os.path.join(QA, "ab_states.pt"))["states"].to(env.device)
    T, E = st.shape[0], st.shape[1]
    ids = torch.arange(E, device=env.device)
    qs = []
    for t in range(T):
        env.default_block_state[:, :, :] = st[t]
        env.default_block_state[:, :, 7:] = 0.0               # zero velocities
        env.reset_idx(ids, "ab")
        for _ in range(2):                                    # brief settle
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
        qs.append(q_now(env).cpu())
        if t % 10 == 0:
            print(f"round {t}/{T}: meanQ {qs[-1].mean():.3f} maxQ {qs[-1].max():.3f}")
    torch.save(torch.stack(qs), os.path.join(QA, f"ab_q_{tag}.pt"))
    print(f"saved ab_q_{tag}.pt")


if __name__ == "__main__":
    main()
