"""Stage 2a: collect (o^S, z^tau, p^T, V_T) tuples for student distillation.

Two phases per scene batch, per the method description

  A. NOMINAL plan. Roll the privileged teacher in the unperturbed twin E^0 and
     record its EEF waypoints and actions. This is the prior z^tau the student
     is conditioned on -- exactly what the real robot would carry in from the
     one-shot perception pass.

  B. PERTURBED rollout. perturb() the scene into E^xi, then act with --actor
     (teacher for warm-start, student for DAgger) while the TEACHER is queried
     at every visited state for its full 16-way distribution and value.

The teacher is recurrent, so its hidden state is advanced along the ACTOR's
trajectory rather than its own -- otherwise the label at a student-visited state
would be conditioned on a history the student never experienced (review T1).

  python tools/collect_student_data.py task=MoreOpenLoop train=MoreOpenLoopSetPPO \
    test=True headless=True num_envs=32 checkpoint=<teacher.pth> \
    +scenes=gen-v2/train +out=student_data/shard0 +episodes=2
"""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa: F401
import tools._net_compat  # noqa: F401
import tools._ckpt_compat  # noqa: F401
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import numpy as np, torch
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop.arm_occlusion import SimulatorArmOcclusion, visibility_settings
from isaacgymenvs.open_loop.evaluation_core import aligned_student_obs


def teacher_forward(player, obs):
    """Full distribution + value at these states, advancing the teacher GRU."""
    o = player._preproc_obs(obs)
    inp = {"is_train": False, "prev_actions": None, "obs": o, "rnn_states": player.states}
    with torch.no_grad():
        res = player.model(inp)
    player.states = res["rnn_states"]
    logits = res["logits"]
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.detach(), res["values"].detach()


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    out_dir = to_absolute_path(str(cfg.get("out", "student_data/shard0")))
    os.makedirs(out_dir, exist_ok=True)
    episodes = int(cfg.get("episodes", 1))
    max_steps = int(cfg.get("max_steps", 120))
    actor = str(cfg.get("actor", "teacher"))
    pos_noise = float(cfg.get("pos_noise", 0.003))
    yaw_noise = float(cfg.get("yaw_noise_deg", 2.0))

    set_np_formatting(); cfg.seed = set_seed(cfg.seed, False)
    n = int(cfg.task.env.numEnvs)
    cfg.task.env.videoLog.enabled = True          # colour tensors not needed, but
    cfg.task.env.videoLog.firstStart = 10 ** 9    # segm/depth are always created

    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, n, cfg.sim_device,
                                 cfg.rl_device, cfg.graphics_device_id, cfg.headless,
                                 cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kw)
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver()); runner.load(omegaconf_to_dict(cfg.train)); runner.reset()
    player = runner.create_player(); player.restore(to_absolute_path(cfg.checkpoint))
    env = player.env
    assert actor in ("teacher", "student"), actor
    snet, s_head, P_vec = None, None, None
    if actor == "student":
        # DAgger: the STUDENT drives, the teacher is still queried at every state
        # it visits, so the labels cover the student's own mistakes -- the whole
        # point BC misses. The teacher GRU advances along this trajectory too.
        import torch as _t
        from isaacgymenvs.learning.student_net import StudentNet
        sp = to_absolute_path(str(cfg.student))
        if os.path.isdir(sp):
            for c in ("student.pt", "student_best.pth"):
                if os.path.exists(os.path.join(sp, c)): sp = os.path.join(sp, c); break
        _ck = _t.load(sp, map_location="cpu"); _sd = _ck.get("model", _ck)
        n_out = int(_sd["pi.weight"].shape[0])
        s_head = "residual_xy" if n_out == 2 else "categorical"
        snet = StudentNet(n_actions=n_out).to(cfg.rl_device); snet.load_state_dict(_sd); snet.eval()
        if s_head == "residual_xy":
            import math as _m
            tot = float(cfg.task.env.pushDistanceM); _s = tot/2.0; _ds = _s/_m.sqrt(2)
            P = np.zeros((16, 2), np.float32)
            P[:4] = 2*np.array([[0,_s],[_s,0],[0,-_s],[-_s,0]], np.float32)
            dg = np.array([[_ds,_ds],[-_ds,_ds],[_ds,-_ds],[-_ds,-_ds]], np.float32)
            E,W,N_,S_ = (np.array(x,np.float32) for x in ([_s,0],[-_s,0],[0,_s],[0,-_s]))
            fg,sg = [E,W,E,W],[N_,N_,S_,S_]
            for _a in range(12):
                _d,_m2 = _a//3, _a%3
                P[4+_a] = 2*dg[_d] if _m2==0 else (fg[_d]+sg[_d] if _m2==1 else sg[_d]+fg[_d])
            P_vec = _t.tensor(P, device=cfg.rl_device)
        print(f"[collect] DAgger: actor=student ({s_head}) from {sp}")

    shards, t_start = 0, time.time()
    # settle() re-baselines pristine_block_state to the CURRENT block state, so
    # calling it per episode made each episode's "pristine" the previous
    # episode's scattered end state: clutter drifted apart monotonically across
    # collection (mean pairwise object distance 0.089 m -> 0.178 m over 40
    # episodes, and the already-solved fraction 0.08 -> 0.96). It must run ONCE,
    # before the loop, which is all its docstring ever claimed to do -- pop apart
    # interpenetrating perceived blocks at startup.
    _obses = player.env_reset(env); player.get_batch_size(_obses, 1)
    env.settle(int(cfg.get("settle_steps", 30)))

    for ep in range(episodes):
        obses = player.env_reset(env); player.get_batch_size(obses, 1)
        if getattr(player, "is_rnn", False):
            player.init_rnn()
        obses = player.env_reset(env)

        # ---- phase A: nominal plan in E^0 -------------------------------------
        plan_xy = [[] for _ in range(n)]
        plan_act = [[] for _ in range(n)]
        plan_obj = [[] for _ in range(n)]      # twin-predicted object layout
        for t in range(max_steps):
            a = player.get_action(obses, is_deterministic=True)
            a = a.squeeze(-1) if a.dim() > 1 else a
            eef = env.gripper_pos[:, :2].detach().cpu().numpy()
            av = a.detach().cpu().numpy().reshape(-1)
            cen = env.blocks_rect_rotated[:, :, 0, :].detach().cpu().numpy()   # (O,E,2)
            for i in range(n):
                plan_xy[i].append(tuple(eef[i])); plan_act[i].append(int(av[i]))
                plan_obj[i].append(cen[:, i, :].copy())
            obses, _, _, _ = player.env_step(env, a)
            if bool((env.grasp_q_parallel_values >= 0.9).all()):
                break

        # ---- phase B: perturbed rollout in E^xi, teacher queried throughout ----
        env.perturb(np.arange(n), pos_noise=pos_noise, yaw_noise_deg=yaw_noise, seed=ep)
        env.reset_idx(torch.arange(n, device=env.device))
        obses = player.env_reset(env)
        if getattr(player, "is_rnn", False):
            player.init_rnn()

        vec_b, log_b, val_b, meta_b, live_b = [], [], [], [], []
        # A trajectory ENDS when its own target becomes graspable. Previously the
        # loop ran a fixed max_steps for every env and only broke when ALL envs
        # were solved simultaneously, so ~70% of stored pairs were post-solve
        # states the policy never has to act in. Standard DAgger labels the
        # states a rollout actually visits before termination -- nothing after.
        live = np.ones(n, np.float32)
        s_h = None            # student GRU state, per episode
        prev_a = np.full(n, -1, np.int64)
        stale = np.zeros((n, so.N_TOKENS), np.float32)
        vis_frac = np.zeros(n, np.float32)
        rng = np.random.default_rng(1234 + ep)
        occlusion = SimulatorArmOcclusion(env)
        visibility_config = visibility_settings(cfg, max_steps)
        p_drop = visibility_config['p_drop']
        bl_len = visibility_config['blackout_len']
        blackout_steps = []
        for i in range(n):                       # one sampled blackout window per env
            if bl_len > 0:
                t0 = int(rng.integers(0, max(1, max_steps - bl_len)))
                blackout_steps.append(set(range(t0, t0 + bl_len)))
            else:
                blackout_steps.append(set())
        for t in range(max_steps):
            logits, values = teacher_forward(player, obses)      # label AT this state
            a = torch.argmax(logits, dim=-1)

            # token-space student view: no render needed, occlusion is modelled
            centers = env.blocks_rect_rotated[:, :, 0, :].detach().cpu().numpy()   # (O,E,2)
            eef_np = env.gripper_pos[:, :2].detach().cpu().numpy()
            tobs = obses["obs"].detach().cpu().numpy() if isinstance(obses, dict) \
                else obses.detach().cpu().numpy()
            rows = []
            geometric_visibility = occlusion.visibility(visibility_config)
            permutations = env.t_obs_perm.detach().cpu().numpy()
            for i in range(n):
                vis = geometric_visibility[i]
                vis = so.apply_random_occlusion(vis, rng, p_drop=p_drop,
                                                blackout=(t in blackout_steps[i]))
                stale[i] = so.step_staleness(stale[i], vis)
                rows.append(aligned_student_obs(so,tobs[i],centers[:,i,:],vis,stale[i],
                                                permutations[i],prev_a[i],plan_xy[i],t,
                                                plan_act[i],plan_obj[i]))
                vis_frac[i] = float(vis.mean())
            vec_b.append(np.stack(rows))
            log_b.append(logits.cpu().numpy())
            val_b.append(values.cpu().numpy().reshape(n))
            live_b.append(live.copy())          # 1 while this env is still running
            meta_b.append(np.array([[t, vis_frac[i], float(env.grasp_q_parallel_values[i])]
                                    for i in range(n)], np.float32))
            if snet is not None:                    # DAgger: the student acts
                import torch as _t
                _x = _t.tensor(np.stack(rows), dtype=_t.float32,
                               device=cfg.rl_device).unsqueeze(0)
                with _t.no_grad():
                    _out, _, s_h = snet(_x, s_h)
                a = (_t.cdist(_out[0], P_vec).argmin(-1) if s_head == "residual_xy"
                     else _out[0].argmax(-1)).to(env.device)
            prev_a = a.detach().cpu().numpy().reshape(-1)
            obses, _, _, _ = player.env_step(env, a)
            # retire every env whose target just became graspable
            solved_now = (env.grasp_q_parallel_values >= 0.9).detach().cpu().numpy()
            live = live * (~solved_now).astype(np.float32)
            if live.sum() == 0:
                break

        path = os.path.join(out_dir, f"ep{ep:04d}.npz")
        np.savez_compressed(path,
                            obs=np.stack(vec_b).astype(np.float32),     # (T,N,OBS_DIM)
                            teacher_logits=np.stack(log_b).astype(np.float32),
                            teacher_value=np.stack(val_b).astype(np.float32),
                            meta=np.stack(meta_b).astype(np.float32),
                            visibility_config=np.array(json.dumps(visibility_config,sort_keys=True)),
                            valid=np.stack(live_b).astype(np.float32))   # (T,N) pre-termination
        shards += 1
        _v = np.stack(live_b)
        print(f"[collect] ep{ep}: {len(vec_b)} steps x {n} envs -> {path} "
              f"(visible mean {np.stack(meta_b)[..., 1].mean():.2f}, "
              f"valid {100*_v.mean():.0f}%, mean traj len {_v.sum(0).mean():.1f})")

    print(f"[collect] {shards} shards in {time.time()-t_start:.1f} s -> {out_dir}")


if __name__ == "__main__":
    main()
