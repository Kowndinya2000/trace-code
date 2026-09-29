"""Carve a tier-stratified validation split out of a test suite.

Checkpoint selection must not read the suite we report on: evaluating N
checkpoints and keeping the best inflates the reported number by ~1.6*SE
(≈ +3 pp for 20 checkpoints at 85% on 505 scenes). This makes `val/` for
selection and leaves `test/` for the single final number.

Usage (from isaacgymenvs/):
  python tools/make_val_split.py test-cases/gen-v2 --val-frac 0.35
"""
import argparse
import json
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--val-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    src = os.path.join(a.root, "test")
    meta = json.load(open(os.path.join(src, "meta.json")))["scenes"]
    by_tier = {}
    for m in meta:
        by_tier.setdefault(m.get("tier", "?"), []).append(m)

    rng = random.Random(a.seed)
    out = {}
    for split in ("val", "test_final"):
        d = os.path.join(a.root, split)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        out[split] = []

    for tier, rows in sorted(by_tier.items()):
        rows = sorted(rows, key=lambda m: m["scene"])
        rng.shuffle(rows)
        k = int(round(len(rows) * a.val_frac))
        for split, sel in (("val", rows[:k]), ("test_final", rows[k:])):
            for m in sel:
                out[split].append(dict(m))
        print(f"  {tier}: {k} val / {len(rows)-k} test_final")

    for split, rows in out.items():
        d = os.path.join(a.root, split)
        rows.sort(key=lambda m: (m.get("tier", ""), m["scene"]))
        for i, m in enumerate(rows):
            shutil.copy(os.path.join(src, m["scene"]), os.path.join(d, f"{i:06d}.txt"))
            m["src_scene"], m["scene"] = m["scene"], f"{i:06d}.txt"
        json.dump({"scenes": rows}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"{split}: {len(rows)} scenes -> {d}")


if __name__ == "__main__":
    main()
