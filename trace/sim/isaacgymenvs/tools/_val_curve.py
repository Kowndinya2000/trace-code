"""Validation learning curve: evaluate new checkpoints on gen-v2/val and log to wandb.

Selection happens on val (177 scenes) so gen-v2/test_final (328) stays untouched
until one checkpoint per run is chosen. Runs as its own process so it never
perturbs or slows training; writes a CSV as well as a companion wandb run.
Usage: python tools/_val_curve.py <run-name> [every_nth_ckpt] [poll_seconds]
"""
import csv, os, re, subprocess, sys, time
from pathlib import Path

RUN = sys.argv[1]
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 10      # eval every Nth checkpoint
POLL = int(sys.argv[3]) if len(sys.argv) > 3 else 900
GPU = os.environ.get("VAL_GPU", "0")
ROOT = str(Path(__file__).resolve().parents[2])
CSV = f"{ROOT}/tools/qa_out/val_curve_{RUN}.csv"
# train config must match the checkpoint's network exactly. SC checkpoints have
# embed.0.weight (64,11) and cannot load under the non-SC builder.
if "setsc" in RUN:   TRAIN_CFG = "train=MoreTeacherSetSCPPO"
elif "set" in RUN:   TRAIN_CFG = "train=MoreTeacherSetPPO"
else:                TRAIN_CFG = None

os.chdir(ROOT)
try:
    import wandb
    wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "trace"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    id=f"{RUN}-val", name=f"{RUN}-val", group=RUN,
                    job_type="val-curve", resume="allow", reinit=True)
except Exception as e:                                      # never let logging kill the daemon
    print("wandb unavailable:", e); wb = None

done = set()
if os.path.exists(CSV):
    done = {int(r["epoch"]) for r in csv.DictReader(open(CSV))}
else:
    with open(CSV, "w") as f:
        f.write("epoch,frames,val_success_pct,val_oow,graspable_pct,median_steps,val_return,val_return_succ,ckpt\n")

def epoch_of(p):
    m = re.search(r"_ep_(\d+)_", os.path.basename(p))
    return int(m.group(1)) if m else -1

while True:
    cks = sorted((p for p in os.popen(f"ls {ROOT}/runs/{RUN}/nn/*_ep_*.pth 2>/dev/null").read().split()),
                 key=epoch_of)
    todo = [p for p in cks if epoch_of(p) >= 0 and epoch_of(p) not in done
            and (epoch_of(p) // 15) % EVERY == 0]
    for ck in todo:
        ep = epoch_of(ck)
        cmd = ["python", "-u", "tools/eval_strict.py", "task=MoreTeacher", "test=True", "headless=True",
               "num_envs=177", "task.env.test_cases.scene_root_dir=test-cases",
               "task.env.test_cases.difficulty_choice=gen-v3/val",
               "task.env.robust.randomizeReset=False", "task.env.robust.domainRand=False",
               "task.env.videoLog.enabled=False", "wandb_activate=False",
               "task.env.episodeLength=450",          # eval at the published horizon
               f"sim_device=cuda:{GPU}", f"rl_device=cuda:{GPU}", f"graphics_device_id={GPU}",
               f"checkpoint={ck}", f"+strict_out=tools/qa_out/val_{RUN}_ep{ep}.csv"]
        if TRAIN_CFG:
            cmd.insert(3, TRAIN_CFG)
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        def grab(pat, cast=float, default=-1):
            m = re.search(pat, out)
            return cast(m.group(1)) if m else default
        succ = grab(r"SUCCESS\s*:\s*\d+/\d+ = ([\d.]+)%")
        grasp = grab(r"graspable\s*:\s*\d+/\d+ = ([\d.]+)%")
        oow = grab(r"OOW violat\.:\s*(\d+)/", int)
        med = grab(r"median (\d+)", int)
        ret = grab(r"RETURN\s*:\s*mean ([-\d.]+)")
        ret_s = grab(r"RETURN.*?succ ([-\d.]+)")
        frames = ep * 1024 * 128
        with open(CSV, "a") as f:
            f.write(f"{ep},{frames},{succ},{oow},{grasp},{med},{ret},{ret_s},{os.path.basename(ck)}\n")
        if wb:
            wb.log({"val/success_pct": succ, "val/graspable_pct": grasp, "val/oow": oow,
                    "val/median_steps": med, "val/return": ret, "val/return_succ": ret_s,
                    "epoch": ep, "frames": frames})
        print(f"[{RUN}] ep {ep}: val success {succ}% oow {oow}", flush=True)
        done.add(ep)
    if POLL == 0:            # single pass: lets one GPU serve several runs in turn
        break
    time.sleep(POLL)
