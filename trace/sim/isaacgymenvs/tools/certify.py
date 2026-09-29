"""Lockstep robustness certifier prototype (the method description (§3.A)).

For each of M scenes, builds N replicas in one batched sim: replica 0 is the
NOMINAL twin (zero pose noise); replicas 1..N-1 sample perception pose noise
+ per-env friction/mass DR. All replicas of a scene execute ONE shared action
stream chosen from the nominal replica's observations — exactly the fixed
sequence an open-loop robot replay would execute. The certificate of a scene
is the fraction of perturbed replicas that end graspable without pushing
anything out of the workspace: a pre-execution prediction of open-loop
success probability.

Mechanics: scenes are replicated into a temp suite dir (env i*N+j loads
scene i); MoreRobust cert hooks do the rest (noNoiseEnvStride=N keeps
leaders nominal, suppressResets freezes terminals, sceneSymmetry off).

Usage (from isaacgymenvs/):
  python tools/certify.py task=MoreRobust test=True headless=True \
      checkpoint=runs/<...>.pth '+cert_scenes=test-cases/dataset/selected/test-128' \
      +cert_m=8 +cert_n=32 [+cert_budget=250] [+cert_out=tools/qa_out/cert.json]
  (num_envs is set to M*N automatically)
"""
import json
import os
import shutil
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

Q_THRESH = 0.9


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    M = int(cfg.get("cert_m", 8))
    N = int(cfg.get("cert_n", 32))
    budget = int(cfg.get("cert_budget", 250))
    src_dir = to_absolute_path(str(cfg.get("cert_scenes")))
    out_json = to_absolute_path(str(cfg.get("cert_out", "tools/qa_out/cert.json")))
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # replicate the first M scenes N times into a temp suite
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(pkg, "test-cases", "_cert_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))[:M]
    for i, f in enumerate(files):
        for j in range(N):
            shutil.copy(os.path.join(src_dir, f),
                        os.path.join(tmp, f"{i * N + j:06d}.txt"))

    cfg.task.env.numEnvs = M * N
    cfg.num_envs = M * N
    cfg.task.env.test_cases.scene_root_dir = "test-cases"
    cfg.task.env.test_cases.difficulty_choice = "_cert_tmp"
    rb = cfg.task.env.robust
    rb.randomizeReset = True
    rb.sceneSymmetry = False
    rb.homePanNoiseDeg = 0.0
    rb.noNoiseEnvStride = N
    rb.suppressResets = True

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
    dev = env.reset_buf.device
    E = M * N
    leader = (torch.arange(E, device=dev) // N) * N

    latched = torch.zeros(E, dtype=torch.bool, device=dev)
    violated = torch.zeros(E, dtype=torch.bool, device=dev)
    action_streams = [[] for _ in range(M)]
    done_step = torch.full((M,), budget, dtype=torch.long, device=dev)

    for step in range(budget):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        a_b = a[leader]                       # broadcast leader actions
        for g in range(M):
            if int(done_step[g]) == budget:
                action_streams[g].append(int(a_b[g * N]))
        obses, _, _, _ = player.env_step(player.env, a_b)
        q = env.grasp_q_parallel_values
        violated |= (env._oow_of(env.block_state[:, :, :2]) & ~env.oow_exempt).any(dim=1)
        latched |= (q > Q_THRESH)
        lead_done = (latched | violated)[leader[::N] + 0]
        for g in range(M):
            if int(done_step[g]) == budget and bool(lead_done[g * 1]):
                done_step[g] = step + 1
        if bool(lead_done.all()):
            break

    rows = []
    print(f"\n=== lockstep certificates ({M} scenes x {N} replicas, "
          f"noise {float(rb.poseNoisePos)*1000:.0f}mm/{float(rb.poseNoiseYawDeg)}deg) ===")
    for g in range(M):
        s, e = g * N, (g + 1) * N
        ok = (latched[s:e] & ~violated[s:e])
        nominal = bool(ok[0])
        cert = float(ok[1:].float().mean())
        rows.append({"scene": files[g], "nominal_success": nominal,
                     "certificate": round(cert, 3),
                     "viol_frac": round(float(violated[s + 1:e].float().mean()), 3),
                     "steps": int(done_step[g]),
                     "actions": action_streams[g]})
        print(f"  {files[g]}: nominal={'OK ' if nominal else 'FAIL'} "
              f"certificate={cert:.2f}  viol={rows[-1]['viol_frac']:.2f} "
              f"steps={int(done_step[g])}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rows, open(out_json, "w"), indent=1)
    print(f"saved: {out_json}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
