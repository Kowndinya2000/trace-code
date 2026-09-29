"""Render a 2x2 rollout video with the 55x45cm workspace boundary drawn on
each tile, each tile frozen at its own first termination, and a red frame
flag at the step the training OOW rule (footprint corner outside) fired.
Pixel mapping is fitted at t=0 from the target's segm centroid vs its world
xy (handles axis swap/sign of the top-down camera automatically).

Usage (from isaacgymenvs/):
  python tools/render_boundary_video.py task=MoreTeacher test=True headless=True \
      num_envs=4 task.env.test_cases.scene_root_dir=test-cases \
      task.env.test_cases.difficulty_choice=_oow_pick task.env.robust.randomizeReset=False \
      task.env.robust.domainRand=False task.env.videoLog.enabled=True \
      checkpoint=<ckpt> +out_mp4=/path/out.mp4
"""
import os, sys, subprocess
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa
import tools._net_compat  # noqa: F401  (registers token_set)
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import numpy as np, torch, cv2
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv
    out = to_absolute_path(str(cfg.get("out_mp4", "boundary.mp4")))
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    cfg.task.env.videoLog.enabled = True
    cfg.task.env.videoLog.firstStart = 10 ** 9      # we grab frames ourselves
    set_np_formatting(); cfg.seed = set_seed(42, False)

    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
                                 cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                                 cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                 cfg.force_render, cfg, **kw)
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver()); runner.load(omegaconf_to_dict(cfg.train)); runner.reset()
    player = runner.create_player(); player.restore(cfg.checkpoint)
    env = player.env; obses = player.env_reset(env); player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()
    _v = cfg.get("vid_envs", [0, 1, 2, 3])
    cand = [int(x) for x in (_v if not isinstance(_v, str) else _v.split(","))]
    E = len(cand); dev = env.reset_buf.device

    def grab():
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        col = [env.cam_color_tensors[i].clone().cpu().numpy()[..., :3] for i in cand]
        seg = [env.cam_segm_tensors[i].clone().cpu().numpy() for i in cand]
        env.gym.end_access_image_tensors(env.sim)
        return col, seg

    # fit world->pixel mapping from target centroids at t=0
    col0, seg0 = grab()
    H = col0[0].shape[0]; ppm = 500.0                     # 2mm/px
    tgt = env.block_state[cand, 0, :2].cpu().numpy()
    cents = []
    for s in seg0:
        ys, xs = np.nonzero(s == 255); cents.append((xs.mean(), ys.mean()))
    cents = np.array(cents)
    ok = ~np.isnan(cents).any(1)
    assert ok.any(), "target not visible in any env at t=0"
    cents, tgt = cents[ok], tgt[ok]
    best = None
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                w = tgt[:, ::-1] if swap else tgt
                px = np.stack([sx * w[:, 0], sy * w[:, 1]], 1) * ppm
                off = (cents - px).mean(0); err = np.abs(cents - px - off).mean()
                if best is None or err < best[0]: best = (err, swap, sx, sy, off)
    err, swap, sx, sy, off = best
    print(f"pixel mapping fit: swap={swap} sx={sx} sy={sy} offset={off} mean err {err:.2f}px")

    def to_px(x, y):
        w = (y, x) if swap else (x, y)
        return int(round(sx * w[0] * ppm + off[0])), int(round(sy * w[1] * ppm + off[1]))

    WS_X, WS_Y = env.ws_x, env.ws_y
    corners = [to_px(WS_X[0], WS_Y[0]), to_px(WS_X[1], WS_Y[0]),
               to_px(WS_X[1], WS_Y[1]), to_px(WS_X[0], WS_Y[1])]
    poly = np.array(corners, np.int32).reshape(-1, 1, 2)

    frames = [[] for _ in range(E)]; done = [False] * E; viol = [-1] * E; outcome = ["timeout"] * E
    def annotate(img, e, step, flag):
        im = np.ascontiguousarray(img.copy())
        cv2.polylines(im, [poly], True, (255, 255, 0), 1)
        cv2.putText(im, f"env{cand[e]} t={step}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if flag: cv2.rectangle(im, (0, 0), (H - 1, H - 1), (255, 0, 0), 3)
        return im
    for e in range(E): frames[e].append(annotate(col0[e], e, 0, False))
    for step in range(450):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        obses, _, _, _ = player.env_step(env, a)
        col, _ = grab()
        q = env.grasp_q_parallel_values
        for e in range(E):
            if done[e]: continue
            term = bool(env.reset_buf[cand[e]])
            succ = bool(q[cand[e]] > 0.9)
            if term: outcome[e] = "success" if succ else ("timeout" if step >= env.max_episode_length - 2 else "OOW-corner")
            flag = term and not succ and step < env.max_episode_length - 2
            if flag and viol[e] < 0: viol[e] = step + 1
            frames[e].append(annotate(col[e], e, step + 1, flag))
            if term:
                done[e] = True
                for _ in range(20): frames[e].append(frames[e][-1])   # hold 1s
        if all(done): break
    print("outcomes: " + ", ".join(f"env{cand[e]}={outcome[e]}@{viol[e] if viol[e]>0 else len(frames[e])-21}" for e in range(E)))
    keep = [e for e in range(E) if outcome[e] == "OOW-corner"][:4]
    if len(keep) < 4: keep += [e for e in range(E) if e not in keep][:4 - len(keep)]
    frames = [frames[e] for e in keep]
    print("tiles (TL,TR,BL,BR): " + ", ".join(f"env{cand[e]} {outcome[e]}" for e in keep))
    T = max(len(f) for f in frames)
    for f in frames:
        while len(f) < T: f.append(f[-1])
    while len(frames) < 4: frames.append([np.zeros_like(frames[0][0])] * T)
    raw = out[:-4] + "_raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 20, (2 * H, 2 * H))
    for t in range(T):
        top = np.concatenate([frames[0][t], frames[1][t]], 1)
        bot = np.concatenate([frames[2][t], frames[3][t]], 1)
        vw.write(cv2.cvtColor(np.concatenate([top, bot], 0), cv2.COLOR_RGB2BGR))
    vw.release()
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
    print(f"termination steps (training OOW flag): {viol}; wrote {out}")


if __name__ == "__main__":
    main()
