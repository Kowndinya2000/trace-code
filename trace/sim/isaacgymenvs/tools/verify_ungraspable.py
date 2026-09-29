"""Verify no dataset scene starts graspable: run the grasp classifier on the
t=0 state of every scene (chunked envs, authored poses, no randomization).
Scenes with initial grasp-Q above threshold are trivially solvable -> flagged.

  python tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
      '+vg_scenes=test-cases/gen-v1/train' [+vg_chunk=512] [+vg_thresh=0.9]
"""
import json
import os
import shutil
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
    src = to_absolute_path(str(cfg.get("vg_scenes")))
    chunk = int(cfg.get("vg_chunk", 512))
    start = int(cfg.get("vg_start", -1))   # >=0: process ONE chunk (new sim
    # per process — Isaac Gym cannot recreate sims in-process)
    thresh = float(cfg.get("vg_thresh", 0.9))
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(f for f in os.listdir(src) if f.endswith(".txt"))
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.task.env.test_cases.scene_root_dir = "test-cases"
    cfg.task.env.test_cases.difficulty_choice = "_vg_tmp"
    cfg.task.env.robust.randomizeReset = False
    cfg.task.env.robust.domainRand = False

    results = {}
    ranges = ([start] if start >= 0 else list(range(0, len(files), chunk)))
    for k0 in ranges:
        sub = files[k0:k0 + chunk]
        tmp = os.path.join(pkg, "test-cases", "_vg_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for i, f in enumerate(sub):
            shutil.copy(os.path.join(src, f), os.path.join(tmp, f"{i:06d}.txt"))
        cfg.task.env.numEnvs = len(sub)
        cfg.num_envs = len(sub)
        env = isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.test, len(sub), cfg.sim_device,
            cfg.rl_device, cfg.graphics_device_id, cfg.headless, cfg.multi_gpu,
            cfg.capture_video, cfg.force_render, cfg)
        env.reset()
        a = torch.zeros(len(sub), dtype=torch.long, device=env.device)
        env.step(a)                      # one step -> grasp_prob_B populated
        q = env.grasp_q_parallel_values.clone().cpu()
        for f, qi in zip(sub, q.tolist()):
            results[f] = round(qi, 3)
        del env
        torch.cuda.empty_cache()
        print(f"chunk {k0 // chunk + 1}: {len(sub)} scenes, "
              f"graspable@t0: {int((q > thresh).sum())}, qmax {float(q.max()):.3f}")
        shutil.rmtree(tmp, ignore_errors=True)

    bad = {f: v for f, v in results.items() if v > thresh}
    tag = f"_{start}" if start >= 0 else ""
    out = os.path.join(pkg, "tools", "qa_out",
                       f"graspable_t0_{os.path.basename(src)}{tag}.json")
    json.dump({"threshold": thresh, "bad": bad, "all": results},
              open(out, "w"), indent=1)
    qs = list(results.values())
    print(f"\n=== {os.path.basename(src)}: {len(files)} scenes ===")
    print(f"graspable at t=0 (q>{thresh}): {len(bad)}")
    print(f"q distribution: median {sorted(qs)[len(qs)//2]:.3f}, "
          f"max {max(qs):.3f}")
    if bad:
        print("flagged:", ", ".join(sorted(bad)[:20]),
              "..." if len(bad) > 20 else "")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
