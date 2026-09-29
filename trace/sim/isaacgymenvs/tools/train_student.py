"""Stage-2 BC warm-start: distil the privileged teacher into the student.

Consumes the shards from tools/collect_student_data.py and fits
    L = KL(p^T || p^S) + c * (V_T - V_S)^2
over whole sequences, because the student is recurrent and its whole point is
carrying belief across steps where it cannot see. Batching over flattened
timesteps would train it as if every step were independent, which is precisely
the ability under test.

This is warm-start ONLY. It never queries the teacher on states the student
chose, so it cannot teach recovery from the student's own mistakes -- that is
DAgger's job (collect with --actor student, aggregate, refit). Selection here is
on held-out distillation loss; final model selection must be strict rollout
success under the deployment occlusion regime, never action-matching accuracy
(Aug 26 audit lesson: matching accuracy is the metric that hid the bulldozer).

  python tools/train_student.py --data student_data/ep300_gen-v2 --out runs_student/bc0
"""
import argparse, glob, json, os, time
import numpy as np
import torch

import sys
sys.path.insert(0, os.getcwd())
from isaacgymenvs.learning.student_net import StudentNet, distillation_loss
from isaacgymenvs.open_loop import student_obs as so


def load_shards(paths):
    """Pad ragged shards to the longest sequence and return a validity mask.

    Truncating to min(T) silently destroyed the large-perturbation regimes: one
    1-step shard in P2/P3 collapsed the whole run to T=1 (1% of steps, and no
    sequences at all, so the GRU was never exercised). Early termination is
    exactly what makes those regimes interesting, so they must be kept.
    """
    obs, tl, tv = [], [], []
    for p_ in paths:
        d = np.load(p_)
        obs.append(d["obs"]); tl.append(d["teacher_logits"]); tv.append(d["teacher_value"])
    T = max(x.shape[0] for x in obs)
    D = obs[0].shape[-1]; A = tl[0].shape[-1]
    N = sum(x.shape[1] for x in obs)
    O = np.zeros((T, N, D), np.float32); L = np.zeros((T, N, A), np.float32)
    V = np.zeros((T, N), np.float32);    M = np.zeros((T, N), np.float32)
    k = 0
    for o, l, v in zip(obs, tl, tv):
        t, n = o.shape[0], o.shape[1]
        O[:t, k:k+n] = o; L[:t, k:k+n] = l; V[:t, k:k+n] = v; M[:t, k:k+n] = 1.0
        k += n
    return O, L, V, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of ep*.npz shards")
    ap.add_argument("--out", default="runs_student/bc0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-seqs", type=int, default=32, help="sequences per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=0.05,
                    help="value MSE ran 4-20x the KL at 0.5, drowning the policy head: "
                         "every categorical student collapsed to ONE action (1/16 used, "
                         "100%% of steps). 0.05 keeps the two terms comparable.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    assert paths, f"no shards in {args.data}"
    obs, tl, tv, msk = load_shards(paths)
    T, N, D = obs.shape
    assert D == so.OBS_DIM, f"obs dim {D} != student_obs.OBS_DIM {so.OBS_DIM}"
    print(f"[student] {len(paths)} shards -> T={T} N={N} obs={D}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    OB, TL, TV, MK = to(obs), to(tl), to(tv), to(msk)

    net = StudentNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    best, hist = float("inf"), []

    for ep in range(args.epochs):
        net.train()
        idx = tr_i[rng.permutation(len(tr_i))[:args.batch_seqs]]
        logits, value, _ = net(OB[:, idx])
        loss, kl, v = distillation_loss(logits, TL[:, idx], value, TV[:, idx],
                                        args.value_coef, mask=MK[:, idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        net.eval()
        with torch.no_grad():
            lo, va, _ = net(OB[:, val_i])
            vl, vk, vv = distillation_loss(lo, TL[:, val_i], va, TV[:, val_i],
                                           args.value_coef, mask=MK[:, val_i])
            m_ = MK[:, val_i]
            agree = (((lo.argmax(-1) == TL[:, val_i].argmax(-1)).float() * m_).sum()
                     / m_.sum().clamp(min=1.0))
        hist.append({"epoch": ep, "train_loss": float(loss), "train_kl": float(kl),
                     "val_loss": float(vl), "val_kl": float(vk), "val_value_mse": float(vv),
                     "val_argmax_agreement": float(agree)})
        if float(vl) < best:
            best = float(vl)
            torch.save({"model": net.state_dict(), "obs_dim": so.OBS_DIM,
                        "epoch": ep, "val_loss": best},
                       os.path.join(args.out, "student_best.pth"))
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:4d}  train {float(loss):.4f}  val {float(vl):.4f} "
                  f"(kl {float(vk):.4f}, vmse {float(vv):.4f}, agree {float(agree):.3f})")

    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    print(f"[student] best val {best:.4f} -> {args.out}/student_best.pth")
    print("[student] NOTE: argmax agreement is diagnostic only. Select on strict "
          "rollout success under the deployment occlusion regime.")


if __name__ == "__main__":
    main()
