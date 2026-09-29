"""Push a finished dataset suite to wandb as ONE run: every settled sim-render
sheet per tier (preview_sim*.png) + per-tier layout metrics from meta.json.

Usage: python tools/push_suite_wandb.py test-cases/gen-v2 gen-v2-SUITE
"""
import glob, json, os, sys
import wandb

root, name = sys.argv[1], sys.argv[2]
run = wandb.init(project="push-ret-dataset", name=name, job_type="suite")
tiers = sorted(d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)) and glob.glob(os.path.join(root, d, "*.txt")))
summary = {}
for t in tiers:
    d = os.path.join(root, t)
    n = len(glob.glob(os.path.join(d, "*.txt")))
    summary[f"{t}/n_scenes"] = n
    sheets = sorted(glob.glob(os.path.join(d, "preview_sim*.png")))
    for i, sh in enumerate(sheets):
        wandb.log({f"{t}/sim_sheet_{i:02d}": wandb.Image(sh, caption=f"{t} settled sim renders, sheet {i} ({n} scenes in tier)")})
    meta = os.path.join(d, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        rows = m if isinstance(m, list) else m.get("scenes", [])
        for key in ("t4c_cm", "t3nn_mm", "void_frac", "max_pocket_mm", "blocked16"):
            vals = [r[key] for r in rows if isinstance(r, dict) and key in r]
            if vals:
                wandb.log({f"{t}/{key}": wandb.Histogram(vals)})
                summary[f"{t}/{key}_median"] = sorted(vals)[len(vals) // 2]
run.summary.update(summary)
print(json.dumps(summary, indent=1))
print(run.url)
run.finish()
