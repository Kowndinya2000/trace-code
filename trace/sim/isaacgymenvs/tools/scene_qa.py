"""Static quality audit of test-case scene files (no Isaac Gym needed).

Quantifies, per scene and per suite:
  - schema sanity: 11 lines, z == 0.024, roll/pitch == 0 (mod 2pi), classes known
  - workspace bounds: object centers inside the core policy workspace
    ([0.276, 0.724] x [-0.224, 0.224]) and the extended plate
    ([0.176, 0.724] x [-0.424, 0.224])
  - INTERPENETRATION: pairwise footprint overlap via 1 mm rasterization —
    overlapping layouts are the root cause of objects "flying away" when
    the physics engine resolves initial penetration in tight scenes
  - tightness: nearest-neighbor center distances, target isolation
  - duplicates + train/test leakage across suites (content hash)

Usage (any python with numpy+cv2):
  python tools/scene_qa.py test-cases/dataset/selected/train-512 \
      test-cases/dataset/selected/test-128 [...more suite dirs]
Writes per-scene CSVs and a summary to tools/qa_out/.
"""
import argparse
import hashlib
import math
import os
import sys

import numpy as np
import cv2

# --- block footprints: ONE source of truth, the meshes themselves ----------
# Isaac Gym collides with assets/urdf/more/blocks-more/<name>.obj, so anything
# that places or gates on a hand-written polygon can disagree with physics
# invisibly. It did: the old table had the triangle as an isoceles apex-+y
# 45x85 while the mesh is a right triangle 45x90 with its centroid 7.5/15 mm
# elsewhere, and the cylinder 1 mm oversized. Read the mesh instead.
BLOCK_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets", "urdf", "more", "blocks-more")

_POLY_CACHE = {}


def block_polygon(name):
    """(N, 2) metres: top face of <name>.obj at yaw 0, in the block's own
    origin frame — identical to open_loop.perceive_scene.silhouette_polygon."""
    if name not in _POLY_CACHE:
        verts, faces = [], []
        with open(os.path.join(BLOCK_ASSET_DIR, name + ".obj")) as f:
            for line in f:
                t = line.split()
                if t and t[0] == "v":
                    verts.append([float(v) for v in t[1:4]])
                elif t and t[0] == "f":
                    faces.append([int(v.split("/")[0]) - 1 for v in t[1:]])
        verts = np.asarray(verts)
        zmax = verts[:, 2].max()
        top = [fc for fc in faces if all(abs(verts[i, 2] - zmax) < 1e-6 for i in fc)]
        assert len(top) == 1, f"{name}.obj: expected one top face, got {len(top)}"
        _POLY_CACHE[name] = verts[top[0], :2].copy()
    return _POLY_CACHE[name]


BLOCK_NAMES = ("concave", "cube", "cylinder", "half-cube", "rect", "triangle")
# (x_len, y_len) bounding extents at yaw 0, derived — never hand-edit.
FOOTPRINT = {n: (float(block_polygon(n)[:, 0].ptp()),
                 float(block_polygon(n)[:, 1].ptp())) for n in BLOCK_NAMES}
CONCAVE_POLY = block_polygon("concave")     # kept: imported by build_dataset

TILE = 131          # odd; covers the largest footprint (90 mm) plus padding


def footprint_tile(name, yaw, pad_m=0.0, tile=TILE):
    """Rasterized footprint on a centred tile, rotated by yaw.

    pad_m grows the shape by a true outward offset (disc dilation = Minkowski
    sum), which stays honest for the concave cavity; the old code scaled the
    polygon about the origin, which shrank the cavity instead of growing it."""
    t = np.zeros((tile, tile), np.uint8)
    poly = block_polygon(name)
    c, sn = math.cos(yaw), math.sin(yaw)
    rot = poly @ np.array([[c, sn], [-sn, c]])
    cv2.fillPoly(t, [np.round(rot / RES).astype(np.int32) + tile // 2], 1)
    k = int(round(pad_m / RES))
    if k > 0:
        t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return t


def footprint_mask_at(name, x, y, yaw, grid, origin, pad_m=0.0):
    """Footprint of one block on a `grid` canvas whose pixel (0,0) is `origin`.

    Keeps SUB-PIXEL placement (round the vertex sum, not the centre): a nested
    cylinder touches a concave cavity along ~69 mm of arc, so quantising the
    centre to a whole pixel first invents tens of mm^2 of phantom overlap.
    Padding is a true outward offset (disc dilation), safe here because the
    canvas holds this one shape."""
    mask = np.zeros((grid, grid), np.uint8)
    poly = block_polygon(name)
    c, sn = math.cos(yaw), math.sin(yaw)
    rot = poly @ np.array([[c, sn], [-sn, c]])
    ctr = np.array([(x - origin[0]) / RES, (y - origin[1]) / RES])
    cv2.fillPoly(mask, [np.round(rot / RES + ctr).astype(np.int32)], 1)
    k = int(round(pad_m / RES))
    if k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return mask


CORE_WS = np.array([[0.276, 0.724], [-0.224, 0.224]])
PLATE_WS = np.array([[0.176, 0.724], [-0.424, 0.224]])
BLOCK_Z = 0.024
RES = 0.001            # rasterization resolution (1 mm)
GRID = 900             # 0.9 m square canvas, origin at (0.15, -0.45)
ORIGIN = np.array([0.15, -0.45])


def parse_scene(path):
    objs = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 10:
                continue
            objs.append({
                "name": p[0].split(".")[0],
                "rgb": (float(p[1]), float(p[2]), float(p[3])),
                "x": float(p[4]), "y": float(p[5]), "z": float(p[6]),
                "roll": float(p[7]), "pitch": float(p[8]), "yaw": float(p[9]),
            })
    return objs


def footprint_mask(obj):
    """Rasterized footprint on the shared canvas."""
    return footprint_mask_at(obj["name"], obj["x"], obj["y"], obj["yaw"],
                             GRID, ORIGIN)




def wrapped_zero(a, tol=1e-3):
    return min(abs(a) % (2 * math.pi), 2 * math.pi - (abs(a) % (2 * math.pi))) < tol


def audit_scene(path):
    objs = parse_scene(path)
    n = len(objs)
    issues = []
    if n != 11:
        issues.append(f"n_objects={n}")
    for i, o in enumerate(objs):
        if o["name"] not in FOOTPRINT:
            issues.append(f"obj{i}:unknown_class:{o['name']}")
        if abs(o["z"] - BLOCK_Z) > 1e-6:
            issues.append(f"obj{i}:z={o['z']:.4f}")
        if not wrapped_zero(o["roll"]) or not wrapped_zero(o["pitch"]):
            issues.append(f"obj{i}:tilted")
    xy = np.array([[o["x"], o["y"]] for o in objs])
    out_core = int(np.sum((xy[:, 0] < CORE_WS[0, 0]) | (xy[:, 0] > CORE_WS[0, 1]) |
                          (xy[:, 1] < CORE_WS[1, 0]) | (xy[:, 1] > CORE_WS[1, 1])))
    out_plate = int(np.sum((xy[:, 0] < PLATE_WS[0, 0]) | (xy[:, 0] > PLATE_WS[0, 1]) |
                           (xy[:, 1] < PLATE_WS[1, 0]) | (xy[:, 1] > PLATE_WS[1, 1])))

    masks = [footprint_mask(o) for o in objs]
    max_ov, tot_ov, bad_pairs = 0.0, 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            ov = float(np.logical_and(masks[i], masks[j]).sum()) * (RES * 100) ** 2  # cm^2
            if ov > 0:
                tot_ov += ov
                max_ov = max(max_ov, ov)
                if ov > 0.1:
                    bad_pairs += 1

    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1) + np.eye(n) * 9
    nn = d.min(axis=1)
    tgt_iso = float(d[0].min()) if n else float("nan")

    content = open(path, "rb").read()
    return {
        "scene": os.path.basename(path),
        "issues": ";".join(issues),
        "out_core": out_core, "out_plate": out_plate,
        "max_overlap_cm2": round(max_ov, 3),
        "total_overlap_cm2": round(tot_ov, 3),
        "overlap_pairs": bad_pairs,
        "min_nn_cm": round(float(nn.min()) * 100, 2),
        "mean_nn_cm": round(float(nn.mean()) * 100, 2),
        "target_iso_cm": round(tgt_iso * 100, 2),
        "hash": hashlib.md5(content).hexdigest()[:12],
        "classes": "/".join(o["name"] for o in objs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+", help="scene directories")
    ap.add_argument("--out", default="tools/qa_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_hashes = {}
    for suite in args.suites:
        files = sorted(f for f in os.listdir(suite) if f.endswith(".txt"))
        rows = [audit_scene(os.path.join(suite, f)) for f in files]
        name = os.path.basename(suite.rstrip("/"))
        csv = os.path.join(args.out, f"qa_{name}.csv")
        keys = list(rows[0].keys())
        with open(csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

        ov = np.array([r["max_overlap_cm2"] for r in rows])
        issues = sum(1 for r in rows if r["issues"])
        print(f"\n=== {name}: {len(rows)} scenes -> {csv} ===")
        print(f"schema issues        : {issues}")
        print(f"centers outside core : {sum(r['out_core'] for r in rows)} objs "
              f"in {sum(1 for r in rows if r['out_core'])} scenes")
        print(f"centers outside plate: {sum(r['out_plate'] for r in rows)} objs")
        print(f"interpenetration     : {np.sum(ov > 0.1)} scenes >0.1cm2, "
              f"{np.sum(ov > 0.5)} >0.5cm2, {np.sum(ov > 1.0)} >1.0cm2 "
              f"(max {ov.max():.2f}cm2)")
        worst = sorted(rows, key=lambda r: -r["max_overlap_cm2"])[:5]
        print("worst overlap scenes :", ", ".join(f"{r['scene']}({r['max_overlap_cm2']})"
                                                  for r in worst))
        print(f"tightness min_nn     : median {np.median([r['min_nn_cm'] for r in rows]):.2f}cm, "
              f"min {min(r['min_nn_cm'] for r in rows):.2f}cm")
        dup = len(rows) - len({r["hash"] for r in rows})
        print(f"in-suite duplicates  : {dup}")
        for r in rows:
            all_hashes.setdefault(r["hash"], []).append(f"{name}/{r['scene']}")

    leak = {h: v for h, v in all_hashes.items() if len({p.split('/')[0] for p in v}) > 1}
    if len(args.suites) > 1:
        print(f"\n=== cross-suite identical scenes: {len(leak)} hashes ===")
        train_like = [s for s in (os.path.basename(x.rstrip('/')) for x in args.suites)
                      if "train" in s]
        test_like = [s for s in (os.path.basename(x.rstrip('/')) for x in args.suites)
                     if "test" in s]
        bad = [v for v in leak.values()
               if any(p.split("/")[0] in train_like for p in v)
               and any(p.split("/")[0] in test_like for p in v)]
        print(f"TRAIN/TEST LEAKAGE   : {len(bad)} scenes appear in both train and test suites")
        for v in bad[:10]:
            print("   ", v)


if __name__ == "__main__":
    main()
