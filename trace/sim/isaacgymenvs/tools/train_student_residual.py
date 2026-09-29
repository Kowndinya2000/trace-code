"""Design B student: regress the EEF displacement VECTOR instead of a 16-way choice.

Design A (train_student.py) copies the teacher's categorical push choice. B asks
whether a continuous waypoint correction is the better interface: same trunk,
same shards, but the policy head emits (dx, dy) and is fit by MSE to the
teacher's net commanded displacement.

Comparable metric: the predicted vector is snapped to the nearest of the 16
primitives and compared against the teacher's argmax, so `agree` means the same
thing in both trainers (chance = 1/16 = 0.0625).
"""
import argparse, glob, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.open_loop import student_obs as so


def primitive_vectors(total):
    """Net (dx, dy) per action, mirroring More.build_two_step_plan (d1 + d2)."""
    s = total / 2.0; ds = s / math.sqrt(2)
    v = np.zeros((16, 2), np.float32)
    card = np.array([[0, s], [s, 0], [0, -s], [-s, 0]], np.float32)
    v[:4] = 2 * card
    diag = np.array([[ds, ds], [-ds, ds], [ds, -ds], [-ds, -ds]], np.float32)
    E, W, N, S = (np.array(x, np.float32) for x in ([s,0],[-s,0],[0,s],[0,-s]))
    first  = [E, W, E, W]        # NE, NW, SE, SW
    second = [N, N, S, S]
    for a in range(12):
        d, mode = a // 3, a % 3
        if mode == 0:   v[4+a] = 2 * diag[d]
        elif mode == 1: v[4+a] = first[d] + second[d]
        else:           v[4+a] = second[d] + first[d]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="runs_student/res0")
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--batch-seqs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--device", default="cuda")
    ap.add_argument("--push", type=float, default=0.04, help="pushDistanceM the shards were collected at")
    ap.add_argument("--weight", default="none", choices=["none", "invlen", "corrective"],
                    help="per-state loss weighting scheme (see --help notes in the file header)")
    ap.add_argument("--corrective-tau", type=float, default=0.01,
                    help="metres; corrective weight = clip(||d_student - d_teacher|| / tau, 0, 1)")
    ap.add_argument("--seed", type=int, default=0, help="torch/numpy seed for this fit")
    ap.add_argument("--gru-hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--ablate", default="none",
                    help="observation/architecture ablation (see learning/student_ablate.py)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    assert files, a.data
    # ragged shards: pad to the longest and mask. Truncating to min(T) would drop
    # 99% of the large-perturbation regimes (one 1-step shard collapses P2/P3).
    ob, tll, tvv, vld = [], [], [], []
    for f in files:
        d = np.load(f)
        ob.append(d["obs"]); tll.append(d["teacher_logits"]); tvv.append(d["teacher_value"])
        # `valid` marks the pre-termination steps of each trajectory (the
        # collector retires an environment once its target is graspable).
        # Legacy shards without this key treat every stored step as valid.
        vld.append(d["valid"] if "valid" in d.files
                   else np.ones(d["obs"].shape[:2], np.float32))
    T = max(x.shape[0] for x in ob); Bn = sum(x.shape[1] for x in ob)
    D = ob[0].shape[-1]; A = tll[0].shape[-1]
    obs = np.zeros((T, Bn, D), np.float32); tl = np.zeros((T, Bn, A), np.float32)
    tv = np.zeros((T, Bn), np.float32);     mk = np.zeros((T, Bn), np.float32)
    k = 0
    for o, l, v, q in zip(ob, tll, tvv, vld):
        t, n = o.shape[0], o.shape[1]
        obs[:t, k:k+n] = o; tl[:t, k:k+n] = l; tv[:t, k:k+n] = v
        mk[:t, k:k+n] = q                      # pad mask AND per-env termination
        k += n
    print(f"[student] {len(files)} shards -> T={T} N={Bn} obs={D}  "
          f"valid steps {100*mk.mean():.1f}%  mean traj len {mk.sum(0).mean():.1f}")
    B = Bn
    assert D == so.OBS_DIM, f"obs dim {D} != {so.OBS_DIM}"

    P = torch.tensor(primitive_vectors(a.push), device=a.device)          # (16,2)
    dev = a.device
    obs_t = torch.tensor(obs, device=dev); tl_t = torch.tensor(tl, device=dev)
    tv_t = torch.tensor(tv, device=dev); mk_t = torch.tensor(mk, device=dev)
    tgt_a = tl_t.argmax(-1)                                                # teacher's choice
    tgt_v = P[tgt_a]                                                       # -> displacement label

    n_val = max(1, int(B * a.val_frac)); idx = np.random.permutation(B)
    vi = torch.tensor(idx[:n_val], device=dev); ti = torch.tensor(idx[n_val:], device=dev)

    # Ablation: zero the masked observation channels for BOTH train and val, so
    # the model never sees them, and optionally strip the recurrence. Masking the
    # input (rather than shrinking it) keeps every other dimension at its usual
    # index, so the same evaluator loads the checkpoint unchanged.
    from isaacgymenvs.learning.student_ablate import obs_mask, strip_recurrence, ABLATIONS
    assert a.ablate in ABLATIONS, f"unknown --ablate {a.ablate}; choices {sorted(ABLATIONS)}"
    _m = obs_mask(a.ablate, dev)
    if _m is not None:
        obs_t = obs_t * _m
    # ---- per-state loss weighting -------------------------------------------
    # `none`       every valid state weighs 1 (standard DAgger: uniform over STATES)
    # `invlen`     w = 1/|tau| -> uniform over TRAJECTORIES. Because episodes end on
    #              success, a failed rollout contributes ~120 states and a success
    #              ~15, so uniform-over-states weighs failures 5-8x. This removes
    #              that implicit bias.
    # `corrective` w rises with how far the COLLECTING policy's displacement was
    #              from the teacher's, read from a `corrective` array written by
    #              tools/codagger/annotate_corrective.py. Uses the vector distance,
    #              not argmax mismatch: 6 of 16 primitives share endpoints and the
    #              teacher averages 2.6 near-optimal actions, so argmax disagreement
    #              mostly flags harmless ties.
    if a.weight == "invlen":
        L = mk.sum(0, keepdims=True)                       # (1,B) trajectory length
        mk = mk / np.maximum(L, 1.0)
        mk = mk * (mk.size / max(mk.sum(), 1e-9)) * (float((L > 0).sum()) / max(mk.size, 1))
    elif a.weight == "corrective":
        cw = np.zeros_like(mk); k = 0
        for f, o in zip(files, ob):
            d = np.load(f); n = o.shape[1]; t = o.shape[0]
            if "corrective" not in d.files:
                raise SystemExit(f"{f} has no `corrective` array; run "
                                 f"tools/codagger/annotate_corrective.py first")
            cw[:t, k:k+n] = d["corrective"]; k += n
        w = np.clip(cw / max(a.corrective_tau, 1e-9), 0.0, 1.0)
        mk = mk * w
    if a.weight != "none":
        print(f"[weight] {a.weight}: effective weighted states "
              f"{mk.sum():.0f} (unweighted {int((mk > 0).sum())})")
    mk_t = torch.tensor(mk, device=dev)                    # rebuild after reweighting

    torch.manual_seed(a.seed); np.random.seed(a.seed)   # seed AFTER the split draw
    net = StudentNet(n_actions=2, embed=a.embed, gru_hidden=a.gru_hidden).to(dev)
    if a.ablate == "no_gru":
        net = strip_recurrence(net)
    if a.ablate != "none":
        print(f"[ablate] {a.ablate}: {ABLATIONS[a.ablate]}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)

    def evaluate(sel):
        net.eval()
        with torch.no_grad():
            pred, val, _ = net(obs_t[:, sel])
            m_ = mk_t[:, sel]; dn = m_.sum().clamp(min=1.0)
            mse = (((pred - tgt_v[:, sel]) ** 2).sum(-1) * m_).sum() / dn
            vmse = (((val - tv_t[:, sel]) ** 2) * m_).sum() / dn
            # 6 of the 16 primitives are DUPLICATES in net displacement (grid modes
            # 1 and 2 differ only in waypoint ORDER), so index equality would unfairly
            # penalise B. Credit a match if the snapped vector equals the teacher's.
            snap = torch.cdist(pred.reshape(-1, 2), P).argmin(-1)
            ta = tgt_a[:, sel].reshape(-1)
            ok = (P[snap] - P[ta]).norm(dim=-1).lt(1e-6).float() * m_.reshape(-1)
            agree = ok.sum() / m_.sum().clamp(min=1.0)
            mm = ((pred - tgt_v[:, sel]).norm(dim=-1) * m_).sum() / dn * 1000
        net.train()
        return float(mse), float(vmse), float(agree), float(mm)

    for ep in range(a.epochs):
        perm = ti[torch.randperm(len(ti), device=dev)]
        for k in range(0, len(perm), a.batch_seqs):
            sel = perm[k:k + a.batch_seqs]
            pred, val, _ = net(obs_t[:, sel])
            m_ = mk_t[:, sel]; dn = m_.sum().clamp(min=1.0)
            loss = ((((pred - tgt_v[:, sel]) ** 2).sum(-1) * m_).sum()
                    + a.value_coef * (((val - tv_t[:, sel]) ** 2) * m_).sum()) / dn
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            m, v, ag, mm = evaluate(vi)
            print(f"  ep {ep:4d}  val vec_mse {m:.5f}  vmse {v:.4f}  agree {ag:.3f}  err {mm:.1f} mm", flush=True)
    torch.save({"model": net.state_dict(), "obs_dim": so.OBS_DIM, "head": "residual_xy"},
               os.path.join(a.out, "student.pt"))
    m, v, ag, mm = evaluate(vi)
    json.dump({"vec_mse": m, "value_mse": v, "agree": ag, "mean_err_mm": mm},
              open(os.path.join(a.out, "final.json"), "w"), indent=1)
    print(f"FINAL agree {ag:.3f}  mean err {mm:.1f} mm  -> {a.out}")


if __name__ == "__main__":
    main()
