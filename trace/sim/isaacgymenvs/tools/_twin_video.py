"""Closed-loop vs open-loop replay, same scene, one process, one video.

env0 = policy in the loop (closed-loop reference)
env1 = replay of the DENSE path      (every recorded EEF pose)
env2 = replay of the EXECUTED path   (Douglas-Peucker simplification -> moveL)
env3 = replay of the COMMANDED path  (wp1/wp2 the policy asked for)

set_replay(None) keeps an env policy-driven, so all four run in lockstep on
identical scenes and can be tiled frame-by-frame. Also dumps per-step metrics
(EEF xy, target xy, grasp q) so the divergence can be measured, not just seen.
"""
import json, os, shutil, sys
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa: F401
import tools._net_compat  # noqa: F401  (registers token_set)
import tools._ckpt_compat  # noqa: F401
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import numpy as np, torch, cv2
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory

MODES = ["closed-loop (policy)", "replay: dense", "replay: executed", "replay: commanded"]
COLS = [(110, 200, 110), (200, 160, 90), (90, 90, 235), (170, 110, 200)]


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    traj_json = to_absolute_path(str(cfg.traj_json))
    out_mp4 = to_absolute_path(str(cfg.get("out_mp4", "open_loop/out/twin_compare.mp4")))
    budget = int(cfg.get("budget", 400))
    traj = OpenLoopTrajectory.load(traj_json)
    scene_src = to_absolute_path(traj.scene_file)

    pkg = os.path.dirname(os.path.abspath(__file__)).replace("/tools", "")
    tmp = os.path.join(pkg, "test-cases", "_twinvid_tmp")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for i in range(4):
        shutil.copy(scene_src, os.path.join(tmp, "%06d.txt" % i))
    cfg.task.env.test_cases.scene_root_dir = "test-cases"
    cfg.task.env.test_cases.difficulty_choice = "_twinvid_tmp"
    cfg.task.env.numEnvs = 4
    cfg.task.env.videoLog.enabled = True
    cfg.task.env.videoLog.firstStart = 10 ** 9
    set_np_formatting(); cfg.seed = set_seed(42, False)

    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, 4, cfg.sim_device,
                                 cfg.rl_device, cfg.graphics_device_id, cfg.headless,
                                 cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kw)
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver()); runner.load(omegaconf_to_dict(cfg.train)); runner.reset()
    player = runner.create_player(); player.restore(to_absolute_path(cfg.checkpoint))
    env = player.env
    obses = player.env_reset(env); player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()

    env.settle(int(cfg.get("settle_steps", 30)))
    # settle() steps physics and re-resets, so the obs fetched before it are
    # stale; acting on them sends env0 down a different (still valid) solution
    # branch than the trajectory being replayed. solve_in_twin.py re-reads here.
    obses = player.env_reset(env)
    env.set_replay([None, traj.path("dense"), traj.path("executed"), traj.path("commanded")])
    print("path lengths: dense=%d executed=%d commanded=%d" %
          (len(traj.path("dense")), len(traj.path("executed")), len(traj.path("commanded"))))

    H = env.cam_color_tensors[0].shape[0]
    frames, log = [], []

    def grab():
        env.gym.fetch_results(env.sim, True); env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim); env.gym.start_access_image_tensors(env.sim)
        imgs = [env.cam_color_tensors[i].clone().cpu().numpy()[..., :3] for i in range(4)]
        env.gym.end_access_image_tensors(env.sim)
        return imgs

    for step in range(budget):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        obses, _, _, _ = player.env_step(env, a)
        q = env.grasp_q_parallel_values.detach().cpu().numpy()
        eef = env.gripper_pos[:, :2].detach().cpu().numpy()
        tgt = env.blocks_rect_rotated[0, :, 0, :].detach().cpu().numpy()
        log.append({"step": step, "q": q.tolist(), "eef": eef.tolist(), "tgt": tgt.tolist(),
                    "replay_done": env.replay_done.detach().cpu().tolist()})
        imgs = grab()
        tiles = []
        for i, im in enumerate(imgs):
            t = np.ascontiguousarray(im.copy())
            cv2.rectangle(t, (0, 0), (H - 1, 22), COLS[i], -1)
            cv2.putText(t, MODES[i], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(t, "q=%.2f" % q[i], (5, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (60, 255, 60) if q[i] > 0.9 else (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        top = np.concatenate(tiles[:2], 1); bot = np.concatenate(tiles[2:], 1)
        frames.append(np.concatenate([top, bot], 0))
        if bool(env.replay_done[1:].all()) and step > 20:
            break

    raw = out_mp4[:-4] + "_raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 20, (2 * H, 2 * H))
    for f in frames: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    import subprocess, imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4])
    json.dump(log, open(out_mp4[:-4] + "_metrics.json", "w"))
    qf = env.grasp_q_parallel_values.detach().cpu().numpy()
    print("FINAL q: closed=%.3f dense=%.3f executed=%.3f commanded=%.3f" % tuple(qf))
    print("wrote", out_mp4, "and", out_mp4[:-4] + "_metrics.json", "frames:", len(frames))


if __name__ == "__main__":
    main()
