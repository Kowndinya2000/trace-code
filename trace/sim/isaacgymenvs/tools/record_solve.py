"""Record a twin solve as FOUR separate high-resolution annotated videos.

One video per execution mode, not a tiled contact sheet:
  closed-loop (policy)  - the policy stepping the twin
  replay: dense         - every recorded EEF pose replayed blind
  replay: executed      - the Douglas-Peucker waypoints the robot will moveL
  replay: commanded     - the wp1/wp2 the policy actually asked for

Frames come from the env.videoLog.vizRes camera (default 3840 px over the same
0.64 m the policy sees, i.e. 0.17 mm/px). Square, so 4K means 3840x3840, not
3840x2160. That camera is visualisation-only --
the policy and the grasp network keep the 320 px / fov 0.036829 pair, because
changing resolution without the matching fov rescales mm/px and silently
corrupts grasp Q.

The clock burned into the frames is POLICY STEPPING TIME and nothing else.
Read it for what it is:
  * it excludes the frame grabs and the encode (both outside the timed region),
    which is the point -- rendering must not inflate a reported solve time;
  * it also excludes simulator startup and settle(), so this number is not
    the complete digital-twin stage of a run;
  * four envs step in lockstep here (the policy plus three replays), so it is
    not a single-scene solve time either -- it is what those four cost together.
The pipeline's own twin timing comes from the phase log, not from this file.

  python tools/record_solve.py task=MoreOpenLoop train=MoreOpenLoopSetPPO \
    test=True headless=True checkpoint=<ckpt> +traj_json=<solve.json> \
    +out_dir=open_loop/out/video          # +viz_res=2560 for a smaller render
"""
import json, os, shutil, sys, time, tempfile
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
from isaacgymenvs.open_loop import trajectory as traj_mod
from isaacgymenvs.open_loop import frames

# (file key, display name, sub-caption)
# Sentence case throughout, matching the real-robot overlays; UPPERCASE is
# reserved for status badges. API names (moveL) and acronyms keep their form.
MODES = [("closed-loop", "Simulation rollout", "PPO policy, non-prehensile pushing"),
         ("dense", "Replay: dense", "Every recorded EEF pose, replayed blind in the twin"),
         ("executed", "Replay: executed", "moveL waypoints - what the real robot runs"),
         ("commanded", "Replay: commanded", "The wp1/wp2 targets the policy asked for")]
PHYSICS_MODES = [MODES[0],
                ("joint", "Replay: recorded motor commands", "Original joint targets and physics-tick timing"),
                ("cartesian_feedforward", "Replay: full EEF trace + feedforward", "Original motor commands with fresh EEF tracking correction"),
                ("cartesian", "Replay: full EEF trace, pose only", "Every physics-tick pose through IK; measured tracking may lag")]
TAIL_HOLD = 8      # frames kept after graspability is reached, then cut: the
                   # closed-loop env auto-resets on success and would otherwise
                   # keep running with the target already retrieved
# Google News palette (BGR), the same one the real-robot overlays use, so the
# two video sets read as one deliverable. Medium variants mark graphics, light
# variants carry text, #202124 is every panel ground.
C_PANEL = (36, 33, 32)        # #202124 black
C_TEXT = (244, 243, 241)      # #F1F3F4 light grey
C_DIM = (166, 160, 154)       # #9AA0A6 grey
C_OOW = (4, 188, 251)         # #FBBC04 yellow  (workspace boundary)
C_OK = (214, 234, 206)        # #CEEAD6 light green
C_OK_MARK = (83, 168, 52)     # #34A853 medium green
# one accent per mode, from the four Google hues
ACCENT = [(244, 133, 66),     # #4285F4 medium blue   - policy
          (83, 168, 52),      # #34A853 medium green  - dense
          (53, 67, 234),      # #EA4335 medium red    - executed
          (4, 188, 251)]      # #FBBC04 yellow        - commanded


def draw_overlay(img, mode_i, step, q, t_sim, n_steps, scene_name, solved, target_cls,
                 rate_txt, speed_txt, compute_txt):
    """Annotate one viz frame in place. img is BGR, square, viz_res px.

    Layout is computed from measured text widths so nothing can collide:
      band    left  what the view is             right  rates, then compute cost
      header  left  mode name + what it is        right  twin time + RL env step
      footer  row1  graspability meter            right  GRASPABLE badge
              row2  target class  centre playback speed  right  scene label
    """
    h = img.shape[0]
    s = h / float(frames.CANVAS_SIZE)          # viz px per canvas px
    u = s / 2.0                                # text/geometry unit
    f = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, int(u))

    # --- workspace boundary -------------------------------------------------
    (wx0, wx1), (wy0, wy1) = frames.SIM_WORKSPACE_LIMITS[0], frames.SIM_WORKSPACE_LIMITS[1]
    p = [frames.sim_to_canvas_pix(a, b) for a, b in
         ((wx0, wy0), (wx1, wy0), (wx1, wy1), (wx0, wy1))]
    pts = np.array([[int(round(c * s)), int(round(r * s))] for r, c in p], np.int32)
    cv2.polylines(img, [pts], True, C_OOW, max(1, int(s)), cv2.LINE_AA)
    cv2.putText(img, "Workspace  44.8 x 44.8 cm", (pts[0][0] + int(8 * u), pts[0][1] + int(24 * u)),
                f, 0.48 * u, C_OOW, th, cv2.LINE_AA)
    cv2.putText(img, "Exiting the boundary is treated as failure and episode terminates",
                (pts[0][0] + int(8 * u), pts[0][1] + int(44 * u)), f, 0.33 * u, C_OOW, th, cv2.LINE_AA)

    # standing caption: what the viewer is actually looking at


    # context captions live in the empty band between header and workspace box,
    # so the footer stays two rows and does not crowd the scene
    yc = pts[0][1] - int(12 * u)
    left = "Isaac Gym digital twin  |  orthographic view"
    sc = 0.34 * u
    avail = h - int(50 * u)
    while sc > 0.12 * u:
        lw = cv2.getTextSize(left, f, sc, th)[0][0]
        rw = cv2.getTextSize(rate_txt, f, sc, th)[0][0]
        if lw + rw <= avail:
            break
        sc *= 0.94
    cv2.putText(img, left, (int(16 * u), yc), f, sc, C_DIM, th, cv2.LINE_AA)
    cv2.putText(img, rate_txt, (h - rw - int(18 * u), yc), f, sc, C_DIM, th, cv2.LINE_AA)
    # second standing line: COMPUTE cost, static. The ticking clock above is
    # simulated time so the twin plays at the same 1x the robot videos do;
    # what the twin cost to run is a different quantity and must not share a
    # clock with it.
    if compute_txt:
        cw = cv2.getTextSize(compute_txt, f, sc, th)[0][0]
        cv2.putText(img, compute_txt, (h - cw - int(18 * u), yc - int(22 * u)),
                    f, sc, C_DIM, th, cv2.LINE_AA)

    # --- header -------------------------------------------------------------
    hb = int(52 * u)
    cv2.rectangle(img, (0, 0), (h, hb), C_PANEL, -1)
    cv2.rectangle(img, (0, 0), (int(10 * u), hb), ACCENT[mode_i], -1)
    _, name, desc = MODES[mode_i]
    cv2.putText(img, name, (int(22 * u), int(24 * u)), f, 0.54 * u, C_TEXT, th, cv2.LINE_AA)
    cv2.putText(img, desc, (int(22 * u), int(44 * u)), f, 0.38 * u, C_DIM, th, cv2.LINE_AA)
    r1 = f"Twin time  {t_sim:6.2f} s"
    r2 = f"RL env step {step}/{n_steps}"
    for txt, y, col, sc in ((r1, 24, C_TEXT, 0.48), (r2, 44, C_DIM, 0.42)):
        w = cv2.getTextSize(txt, f, sc * u, th)[0][0]
        cv2.putText(img, txt, (h - w - int(18 * u), int(y * u)), f, sc * u, col, th, cv2.LINE_AA)

    # --- footer -------------------------------------------------------------
    fh = int(76 * u)
    fb = h - fh
    cv2.rectangle(img, (0, fb), (h, h), C_PANEL, -1)
    qcol = C_OK if q >= 0.9 else (252, 227, 210) if q >= 0.4 else C_DIM

    # row 1: graspability meter
    y1 = fb + int(26 * u)
    lbl = f"Graspability {q:5.3f}"
    cv2.putText(img, lbl, (int(16 * u), y1), f, 0.48 * u, qcol, th, cv2.LINE_AA)
    lw = cv2.getTextSize(lbl, f, 0.48 * u, th)[0][0]
    badge = "GRASPABLE" if q >= 0.9 else ""
    bw = cv2.getTextSize("GRASPABLE", f, 0.52 * u, th)[0][0]
    bx0 = int(16 * u) + lw + int(24 * u)
    bx1 = h - bw - int(34 * u)
    if bx1 > bx0:
        cv2.rectangle(img, (bx0, y1 - int(14 * u)), (bx1, y1 - int(2 * u)), (67, 64, 60), -1)
        cv2.rectangle(img, (bx0, y1 - int(14 * u)),
                      (bx0 + int((bx1 - bx0) * min(1.0, q / 1.2)), y1 - int(2 * u)), qcol, -1)
        thr = bx0 + int((bx1 - bx0) * (0.9 / 1.2))
        cv2.line(img, (thr, y1 - int(18 * u)), (thr, y1 + int(2 * u)), C_TEXT, th)
        cv2.putText(img, "0.9", (thr - int(10 * u), y1 + int(16 * u)), f, 0.34 * u, C_DIM, th, cv2.LINE_AA)
    if badge:
        cv2.putText(img, badge, (h - bw - int(18 * u), y1), f, 0.52 * u, C_OK, th, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (h - 1, h - 1), C_OK_MARK, max(2, int(2 * s)))   # green frame = graspable

    # row 2: target on the left, wall-clock on the right
    y2 = fb + int(60 * u)
    cv2.circle(img, (int(24 * u), y2 - int(5 * u)), int(7 * u), (244, 133, 66), -1)  # sim target blue (BGR)
    cv2.putText(img, f"Target: {target_cls}", (int(40 * u), y2), f, 0.44 * u, C_TEXT, th, cv2.LINE_AA)
    sw = cv2.getTextSize(scene_name, f, 0.44 * u, th)[0][0]
    cv2.putText(img, scene_name, (h - sw - int(18 * u), y2), f, 0.44 * u, C_DIM, th, cv2.LINE_AA)
    # centred between them: the gap in footer row 2 is the only spot that costs
    # the workspace view nothing
    pw = cv2.getTextSize(speed_txt, f, 0.42 * u, th)[0][0]
    cv2.putText(img, speed_txt, ((h - pw) // 2, y2), f, 0.42 * u, C_DIM, th, cv2.LINE_AA)

    return img


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    global MODES
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    traj_json = to_absolute_path(str(cfg.traj_json))
    out_dir = to_absolute_path(str(cfg.get("out_dir", "open_loop/out/video")))
    # 4K square. The viz camera is visualisation-only -- the policy and the
    # grasp network keep the 320 px / fov 0.036829 pair, and changing THAT
    # rescales mm/px and silently corrupts grasp Q. Frame capture stays outside
    # the timed region, so sim_solve_seconds is unaffected.
    viz_res = int(cfg.get("viz_res", 3840))
    budget = int(cfg.get("budget", 400))
    os.makedirs(out_dir, exist_ok=True)
    traj = traj_mod.OpenLoopTrajectory.load(traj_json)
    use_physics = traj.physics is not None and not bool(cfg.get("legacy_replays", False))
    if use_physics:
        MODES = PHYSICS_MODES
    scene_src = to_absolute_path(traj.scene_file)
    scene_name = os.path.splitext(os.path.basename(scene_src))[0]
    if traj.scene_text is not None:
        target_cls = traj.scene_text.splitlines()[0].split()[0].replace(".urdf", "")
    else:
        with open(scene_src) as _f:                  # line 0 is the target
            target_cls = _f.readline().split()[0].replace(".urdf", "")
    # human-facing label: "000000" means nothing to an audience
    scene_label = str(cfg.get("scene_label", "")) or f"Test case {int(scene_name) + 1}"
    _dt = float(cfg.task.sim.dt)
    _cfi = int(cfg.task.env.controlFrequencyInv)
    _push_cm = float(cfg.task.env.pushDistanceM) * 100.0
    _ctrl_hz = 1.0 / (_dt * _cfi)
    fps_override = float(cfg.get("fps", 0))     # 0 = the control rate

    # Optional: the phase log of the run this scene came from, so the caption
    # can quote what the twin ACTUALLY cost end to end instead of only the
    # rollout re-measured here.
    twin_stage = None
    _pj = cfg.get("phases_json", None)
    if _pj and os.path.exists(to_absolute_path(str(_pj))):
        with open(to_absolute_path(str(_pj))) as _f:
            _ph = {e["phase"]: e["t"] for e in json.load(_f).get("events", [])}
        if {"twin", "solve-done"} <= set(_ph):
            twin_stage = _ph["solve-done"] - _ph["twin"]
    rate_txt = (f"Physics {1.0 / _dt:.0f} Hz   |   RL control {1.0 / (_dt * _cfi):.0f} Hz "
                f"({_cfi} steps/env step)   |   push action length {_push_cm:.0f} cm")

    pkg = os.path.dirname(os.path.abspath(__file__)).replace("/tools", "")
    tmp = tempfile.mkdtemp(prefix="_recsolve_", dir=os.path.join(pkg, "test-cases"))
    for i in range(4):
        traj.write_scene(os.path.join(tmp, "%06d.txt" % i))
    cfg.task.env.test_cases.scene_root_dir = "test-cases"
    cfg.task.env.test_cases.difficulty_choice = os.path.basename(tmp)
    cfg.task.env.numEnvs = 4
    cfg.task.env.videoLog.enabled = True
    cfg.task.env.videoLog.firstStart = 10 ** 9
    cfg.task.env.videoLog.vizRes = viz_res          # the high-res viz camera
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
    assert len(env.cam_viz_tensors) == 4, "viz camera missing — is videoLog.vizRes set?"
    obses = player.env_reset(env); player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()

    env.settle(int(cfg.get("settle_steps", 30)))
    # Re-read observations AFTER settling, exactly as solve_in_twin.py does.
    # settle() steps physics and re-resets the scene, so the pre-settle obs are
    # stale; acting on them makes the closed-loop tile take a different first
    # push and solve the scene a different way than the trajectory we replay.
    obses = player.env_reset(env)
    if use_physics:
        env.set_timed_replay([None, traj, traj, traj],
                             mode=["joint", "joint", "cartesian_feedforward", "cartesian"])
    else:
        env.set_replay([None, traj.path("dense"), traj.path("executed"), traj.path("commanded")])
    n_steps = max(len(traj.path("dense")), 1)

    spool = os.path.join(out_dir, "_frames")
    shutil.rmtree(spool, ignore_errors=True); os.makedirs(spool)
    buf = [[] for _ in range(4)]          # metadata only; pixels go to disk
    t_solve = 0.0                      # policy/replay stepping ONLY
    t_capture = 0.0
    solved_at = [None] * 4
    t_graspable = [None] * 4
    for step in range(budget):
        t0 = time.perf_counter()
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        obses, _, _, _ = player.env_step(env, a)
        q = env.grasp_q_parallel_values.detach().cpu().numpy()
        if use_physics and q[0] >= 0.9:
            env.freeze(torch.tensor([0], device=env.device))
        t_solve += time.perf_counter() - t0

        t1 = time.perf_counter()
        env.gym.fetch_results(env.sim, True); env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim); env.gym.start_access_image_tensors(env.sim)
        imgs = [env.cam_viz_tensors[i].clone().cpu().numpy()[..., :3] for i in range(4)]
        env.gym.end_access_image_tensors(env.sim)
        for i, im in enumerate(imgs):
            if q[i] >= 0.9 and solved_at[i] is None:
                solved_at[i] = step
                t_graspable[i] = t_solve          # wall-clock at first graspable
            cv2.imwrite(os.path.join(spool, f"{i}_{step:05d}.png"),
                        np.ascontiguousarray(im[..., ::-1]))
            buf[i].append((step, float(q[i]), t_solve))
        t_capture += time.perf_counter() - t1

        if bool(env.replay_done[1:].all()) and step > 20:
            break

    tg = t_graspable[0] if t_graspable[0] is not None else t_solve
    print(f"[record] stepped {len(buf[0])} frames | time to first graspable {tg:.2f} s "
          f"| total stepping {t_solve:.2f} s | capture overhead {t_capture:.2f} s (excluded)")
    print("[record] NOTE: that clock is policy stepping for 4 tiled envs; it excludes "
          "simulator startup and settle, so it is not the twin stage of a pipeline run.")

    # Playback is 1x of SIMULATED time: one frame per RL env step, encoded at
    # the control rate, so the twin's motion runs at the speed it would run at
    # on a robot. That is what makes this directly comparable to the D415/D455
    # videos, which are 1x of wall-clock -- and the push leg on the real arm is
    # itself speed-matched to the twin's EEF. Pegging playback to COMPUTE time
    # instead made the twin play at a speed nothing else in the demo shares.
    fps = fps_override or _ctrl_hz
    _speed = fps / _ctrl_hz
    speed_txt = ("Speed 1x  (real time)" if abs(_speed - 1.0) < 0.02
                 else f"Speed {_speed:.2f}x")
    # compute cost, reported as static text rather than as a clock
    parts = [f"Policy rollout {t_solve:.2f} s compute"]
    if twin_stage is not None:
        parts.append(f"Digital twin stage {twin_stage:.2f} s "
                     f"(simulator load + settle + rollout)")
    compute_txt = "   |   ".join(parts)
    print(f"[record] encoding at {fps:.2f} fps -> {speed_txt};  {compute_txt}")

    import subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    written = []
    for i, (key, _nm, _d) in enumerate(MODES):
        slug = scene_label.lower().replace(" ", "-")
        raw = os.path.join(out_dir, f"{slug}_{key}_raw.mp4")
        out = raw.replace("_raw.mp4", ".mp4")
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (viz_res, viz_res))
        rows = buf[i]
        if solved_at[i] is not None:
            # Cut at the RESET, not at a fixed offset: the closed-loop env
            # auto-resets on success, and q collapsing back to ~0 with the
            # blocks rearranged is that reset, not part of the solve.
            cut = solved_at[i]
            for st, qv, _t in rows:
                if st > solved_at[i]:
                    if qv < 0.5:
                        break
                    cut = st
            cut = min(cut, solved_at[i] + TAIL_HOLD)
            rows = [r for r in rows if r[0] <= cut]
            rows += [rows[-1]] * max(0, TAIL_HOLD - (cut - solved_at[i]))   # hold on success
        for step, q, t in rows:
            im = cv2.imread(os.path.join(spool, f"{i}_{step:05d}.png"))
            vw.write(draw_overlay(im, i, step, q, step / _ctrl_hz, len(rows) - 1, scene_label,
                                  solved_at[i] is not None and step >= solved_at[i], target_cls,
                                  rate_txt, speed_txt, compute_txt))
        vw.release()
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", raw,
                        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        os.remove(raw); written.append(out)
        print(f"[record] {out}")

    meta = {"scene": scene_name, "scene_label": scene_label, "target_class": target_cls,
            "physics_hz": round(1.0 / _dt, 1), "control_hz": round(1.0 / (_dt * _cfi), 1),
            "control_frequency_inv": _cfi, "push_primitive_m": round(_push_cm / 100.0, 3), "trajectory": traj_json, "viz_res": viz_res, "fps": round(fps, 3), "playback_speed": round(_speed, 3),
            "playback_clock": "simulated time", "twin_stage_seconds": twin_stage,
            # named for what it is: policy stepping for the FOUR tiled envs, to
            # first graspable. NOT the twin stage of a pipeline run.
            "policy_stepping_seconds_to_graspable": round(tg, 3),
            "sim_solve_seconds": round(tg, 3),          # back-compat alias
            "num_envs_stepped": 4,
            "excludes": ["simulator startup", "settle", "frame capture", "encode"],
            "sim_total_stepping_seconds": round(t_solve, 3),
            "per_mode_time_to_graspable": {MODES[i][0]: (round(t_graspable[i], 3)
                                                         if t_graspable[i] is not None else None)
                                           for i in range(4)},
            "capture_overhead_seconds": round(t_capture, 3),
            "frames": len(buf[0]), "videos": written,
            "solved": bool(traj.metadata.get("solved")),
            "num_steps": traj.metadata.get("num_steps"),
            "final_q": traj.metadata.get("final_q")}
    tpath = os.path.join(out_dir, f"{scene_label.lower().replace(' ', '-')}_sim_timing.json")
    with open(tpath, "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(spool, ignore_errors=True)
    print(f"[record] timing -> {tpath}")


if __name__ == "__main__":
    main()
