"""Settle-safe programmatic scene generator (scene generator).

Goal: reach manually-curated difficulty (tight, puzzle-like, target buried)
WITHOUT the manual pain: objects are packed by sliding them inward in 1 mm
steps until just-before-contact of rasterized footprints, so scenes are
contact-tight with ZERO interpenetration by construction — nothing can fly
away at load (the root cause found by tools/scene_qa.py in the old dataset).

Difficulty knobs (per generated suite):
  --gap-mm        residual gap after slide-in (0.5 = hard .. 5 = easy)
  --concave-frac  fraction of clutter drawn from {concave, triangle}
  --min-blocked   of the 16 gripper orientations around the target, how many
                  must be blocked at t=0 (16 = fully buried, like the manual
                  suite; also written per scene as the difficulty metric)
  --center-jitter workspace-center jitter (pushes scenes toward walls = harder)

Outputs per scene: NNNNNN.txt (More format, target first, 11 objects),
metadata in <out>/meta.json, a contact-sheet preview PNG per batch, and
optional wandb image/histogram logging (--wandb).

Validation: --settle runs every generated scene in a robot-free Isaac Gym
sim for --settle-steps and reports per-scene max object drift (gate: 5 mm).

Examples (from isaacgymenvs/):
  python tools/build_dataset.py --num 64 --gap-mm 1 --min-blocked 16 \
      --out test-cases/gen-pilot --seed 0 --wandb
  python tools/build_dataset.py --settle --out test-cases/gen-pilot
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_qa import (FOOTPRINT, CORE_WS, CONCAVE_POLY, audit_scene,  # noqa: E402
                      block_polygon, footprint_mask_at, TILE)
from scene_qa import footprint_tile as _mesh_tile  # mesh-derived, shared

RES = 0.001                    # 1 mm rasterization
GRID = 700                     # 0.7 m canvas
ORIGIN = np.array([0.10, -0.35])
BLOCK_Z = 0.024
TARGET_CLASSES = ["cube", "cylinder", "half-cube"]   # paper convention
CLUTTER_CLASSES = ["cube", "cylinder", "half-cube", "rect", "triangle", "concave"]
TARGET_COLOR = (0.29, 0.44, 0.85)                    # dataset blue
PALETTE = [(0.93, 0.79, 0.28), (1.0, 0.34, 0.35), (0.35, 0.63, 0.31),
           (0.95, 0.56, 0.17), (0.69, 0.48, 0.63), (0.46, 0.72, 0.70),
           (0.61, 0.46, 0.37), (1.0, 0.62, 0.65), (0.55, 0.25, 0.25)]
GRIPPER_L, GRIPPER_W = 0.12, 0.023                   # grasp corridor (more.py)
# EEF home keep-out: home (0.32, 0.034), pan noise arcs y by ~+-4.5cm;
# no footprint may come within HOME_CLEAR of the segment
HOME_SEG = ((0.320, -0.015), (0.320, 0.083))
HOME_CLEAR = 0.02
# workspace = 0.448 m square centred (0.5, 0); every footprint corner must lie
# inside the square inset by WS_INSET so no scene starts touching the OOW line
WS_CENTER = (0.5, 0.0)
WS_HALF = 0.224
WS_INSET = 0.05


def in_inset(pts):
    lo, hi = WS_HALF - WS_INSET, WS_HALF - WS_INSET
    return (abs(pts[:, 0] - WS_CENTER[0]) <= lo).all() and (abs(pts[:, 1] - WS_CENTER[1]) <= hi).all()


def home_ok(name, x, y):
    fx, fy = FOOTPRINT[name]
    r = ((fx / 2) ** 2 + (fy / 2) ** 2) ** 0.5
    (x1, y1), (x2, y2) = HOME_SEG
    t = min(max((y - y1) / (y2 - y1), 0.0), 1.0)
    d = ((x - x1) ** 2 + (y - (y1 + t * (y2 - y1))) ** 2) ** 0.5
    return d > HOME_CLEAR + r
NUM_OBJECTS = 11


def to_px(xy):
    return np.round((np.asarray(xy) - ORIGIN) / RES).astype(np.int32)


def footprint_mask(name, x, y, yaw, pad_m=0.0):
    """Footprint of one block on the GRID canvas (mesh top face, see scene_qa)."""
    return footprint_mask_at(name, x, y, yaw, GRID, ORIGIN, pad_m)


def footprint_tile(name, yaw, pad_m=0.0):
    """Canonical footprint mask on a TILE x TILE canvas, centered."""
    return _mesh_tile(name, yaw, pad_m)


def tile_hits(occupied, tile, x, y):
    """Does the tile centered at world (x, y) hit the occupied canvas?"""
    px, py = to_px((x, y))
    h = TILE // 2
    if px - h < 0 or py - h < 0 or px + h + 1 > GRID or py + h + 1 > GRID:
        return True
    win = occupied[py - h:py + h + 1, px - h:px + h + 1]
    return bool(np.logical_and(win, tile).any())


def corner_pts(name, x, y, yaw):
    """Mesh top-face vertices in world metres — used for inset/hull containment,
    so every vertex counts, not just the four bounding corners."""
    pts = block_polygon(name)
    cs, sn = math.cos(yaw), math.sin(yaw)
    return pts @ np.array([[cs, sn], [-sn, cs]]) + [x, y]


def rect_mask(cx, cy, L, W, yaw):
    mask = np.zeros((GRID, GRID), np.uint8)
    pts = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    cs, sn = math.cos(yaw), math.sin(yaw)
    rot = pts @ np.array([[cs, sn], [-sn, cs]])
    cv2.fillPoly(mask, [np.round(rot / RES).astype(np.int32) + to_px((cx, cy))], 1)
    return mask


def blocked_orientations(objs):
    """How many of the 16 grasp corridors around the target hit clutter."""
    tgt = objs[0]
    clutter = np.zeros((GRID, GRID), np.uint8)
    for o in objs[1:]:
        clutter |= footprint_mask(o["name"], o["x"], o["y"], o["yaw"])
    blocked = 0
    for k in range(16):
        ang = k * math.pi / 16.0
        corr = rect_mask(tgt["x"], tgt["y"], GRIPPER_L, GRIPPER_W, ang)
        corr &= ~footprint_mask(tgt["name"], tgt["x"], tgt["y"], tgt["yaw"])
        if np.logical_and(corr, clutter).any():
            blocked += 1
    return blocked


def sample_yaw(rng, yaw_mode):
    if yaw_mode == "axis":   # manual-suite look: near-axis-aligned mosaic
        return rng.choice([0.0, math.pi / 2]) + math.radians(rng.uniform(-5, 5))
    return rng.uniform(0, 2 * math.pi)


def compact(objs, gap, sweeps, rng, tgap=None):
    tgap = gap if tgap is None else tgap
    """Re-slide each clutter object toward the cluster centroid until contact.
    Densifies dendritic first-pass packings into mosaic-like layouts."""
    for _ in range(sweeps):
        order = rng.permutation(len(objs) - 1) + 1
        for idx in order:
            others = np.zeros((GRID, GRID), np.uint8)
            for j, o in enumerate(objs):
                if j != idx:
                    others |= footprint_mask(o["name"], o["x"], o["y"], o["yaw"],
                                             pad_m=gap)
            o = objs[idx]
            d_tgt = math.hypot(o["x"] - objs[0]["x"], o["y"] - objs[0]["y"])
            near = d_tgt < 0.11
            cen = (np.array([objs[0]["x"], objs[0]["y"]]) if near else
                   np.mean([[q["x"], q["y"]] for j, q in enumerate(objs) if j != idx],
                           axis=0))
            v = cen - np.array([o["x"], o["y"]])
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            v /= n
            tile = footprint_tile(o["name"], o["yaw"])
            x, y = o["x"], o["y"]
            while True:
                nx, ny = x + v[0] * 0.001, y + v[1] * 0.001
                if tile_hits(others, tile, nx, ny):
                    break
                x, y = nx, ny
                if np.linalg.norm(cen - [x, y]) < 0.002:
                    break
            o["x"], o["y"] = x, y
    return objs


BIG = ["concave", "rect", "triangle"]
SMALL = ["cube", "cylinder", "half-cube"]


def sample_class_list(rng, n_clutter, big_range):
    """Shuffled clutter class list honoring the big-object count range."""
    n_big = int(rng.integers(big_range[0], big_range[1] + 1))
    n_big = min(n_big, n_clutter)
    lst = [BIG[int(rng.integers(3))] for _ in range(n_big)] +           [SMALL[int(rng.integers(3))] for _ in range(n_clutter - n_big)]
    rng.shuffle(lst)
    return lst


def generate_scene(rng, gap_mm, concave_frac, min_blocked, center_jitter,
                   yaw_mode="random", compact_sweeps=2, target_gap_mm=0.5,
                   n_objects=NUM_OBJECTS, motif=False, big_range=None):
    gap = gap_mm / 1000.0
    tgap = target_gap_mm / 1000.0
    for _ in range(40):                                  # scene-level retries
        cx = WS_CENTER[0] + rng.uniform(-center_jitter, center_jitter)
        cy = WS_CENTER[1] + rng.uniform(-center_jitter, center_jitter)
        objs = [{"name": ("cylinder" if motif else rng.choice(TARGET_CLASSES)),
                 "x": cx + rng.uniform(-0.01, 0.01),
                 "y": cy + rng.uniform(-0.01, 0.01),
                 "yaw": sample_yaw(rng, yaw_mode), "color": TARGET_COLOR}]
        n_clamps = 2 if (motif and True) else 0
        class_list = None
        if big_range is not None:
            br = (max(0, big_range[0] - n_clamps), max(0, big_range[1] - n_clamps))
            class_list = sample_class_list(rng, n_objects - 1 - n_clamps, br)
        if motif and objs[0]["name"] == "cylinder":
            phi = sample_yaw(rng, yaw_mode) if yaw_mode == "axis" else rng.uniform(0, 2 * math.pi)
            for sgn in (1.0, -1.0):
                ux, uy = math.cos(phi) * sgn, math.sin(phi) * sgn
                objs.append({"name": "concave",
                             "x": objs[0]["x"] - 0.0235 * ux,
                             "y": objs[0]["y"] - 0.0235 * uy,
                             "yaw": math.atan2(uy, ux) - math.pi / 2,
                             "color": PALETTE[rng.integers(len(PALETTE))]})
        occupied = footprint_mask(objs[0]["name"], objs[0]["x"], objs[0]["y"],
                                  objs[0]["yaw"], pad_m=gap)
        for o in objs[1:]:
            occupied |= footprint_mask(o["name"], o["x"], o["y"], o["yaw"], pad_m=tgap)
        # stratified first ring: one neighbor per angular sector, slid toward
        # the TARGET with the tighter target gap -> uniform tight enclosure
        n_ring = min(5, n_objects - 1)  # clamps (if any) count toward the ring
        sector = 2 * math.pi / n_ring
        ring_angles = [(k + rng.uniform(0.15, 0.85)) * sector for k in range(n_ring)]
        rng.shuffle(ring_angles)
        ok = True
        hull_pts = [corner_pts(o["name"], o["x"], o["y"], o["yaw"]) for o in objs]
        while len(objs) < n_objects:
            in_ring = len(objs) - 1 < n_ring
            pad = tgap if in_ring else gap
            best_cand, best_area = None, None
            n_valid = 0
            for _try in range(60):
                if class_list is not None:
                    name = class_list[len(objs) - 1 - (2 if motif and objs[0]["name"] == "cylinder" else 0)]                         if len(objs) - 1 - (2 if motif and objs[0]["name"] == "cylinder" else 0) < len(class_list)                         else rng.choice(["cube", "cylinder", "half-cube"])
                elif rng.random() < concave_frac:
                    name = rng.choice(["concave", "triangle"])
                else:
                    name = rng.choice(["cube", "cylinder", "half-cube", "rect"])
                if in_ring and _try < 30:
                    ang = ring_angles[len(objs) - 1] + rng.uniform(-0.3, 0.3)
                else:
                    ang = rng.uniform(0, 2 * math.pi)
                yaw = sample_yaw(rng, yaw_mode)
                tile = footprint_tile(name, yaw)
                gx = objs[0]["x"] if in_ring else cx
                gy = objs[0]["y"] if in_ring else cy
                fx, fy = FOOTPRINT[name]
                half = max(fx, fy) / 2
                ux, uy = math.cos(ang), math.sin(ang)

                def ws_ok(x, y):
                    return (CORE_WS[0][0] + half < x < CORE_WS[0][1] - half and
                            CORE_WS[1][0] + half < y < CORE_WS[1][1] - half and
                            home_ok(name, x, y))
                # coarse 4mm slide to first contact, then 1mm refine backward
                r, hit = 0.30, None
                while r > 0.02:
                    x, y = gx + r * ux, gy + r * uy
                    if ws_ok(x, y) and tile_hits(occupied, tile, x, y):
                        hit = r
                        break
                    r -= 0.004
                if hit is None:
                    continue
                r = hit
                while r < 0.32:
                    r += 0.001
                    x, y = gx + r * ux, gy + r * uy
                    if ws_ok(x, y) and not tile_hits(occupied, tile, x, y):
                        break
                else:
                    continue
                if not ws_ok(x, y):
                    continue
                n_valid += 1
                if not in_inset(corner_pts(name, x, y, yaw)):
                    continue
                cand_pts = np.vstack(hull_pts + [corner_pts(name, x, y, yaw)])
                area = cv2.contourArea(cv2.convexHull(
                    (cand_pts * 1000).astype(np.int32)))
                if best_area is None or area < best_area:
                    best_area, best_cand = area, (name, x, y, yaw)
                if in_ring or n_valid >= 8:
                    break
            if best_cand is None:
                ok = False
                break
            name, x, y, yaw = best_cand
            objs.append({"name": name, "x": x, "y": y, "yaw": yaw,
                         "color": PALETTE[rng.integers(len(PALETTE))]})
            occupied |= footprint_mask(name, x, y, yaw, pad_m=pad)
            hull_pts.append(corner_pts(name, x, y, yaw))
        if not ok:
            continue
        if compact_sweeps > 0:
            objs = compact(objs, tgap, compact_sweeps, rng)
        nb = blocked_orientations(objs)
        if nb >= min_blocked:
            return objs, nb
    return None, 0


def tightness_metrics(objs):
    """Edge-gap tightness from several angles (mm, 1mm raster):
    t3nn: gaps to the target's 3 nearest neighbors (visual 'target buried')
    nn:   every object's nearest-neighbor gap (global packing density)
    contact_frac: fraction of objects touching (<2mm) a neighbor"""
    masks = [footprint_mask(o["name"], o["x"], o["y"], o["yaw"]) for o in objs]
    dts = [cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
           for m in masks]
    n = len(objs)
    gap = np.full((n, n), 1e9)
    for i in range(n):
        for j in range(n):
            if i != j:
                gap[i, j] = float(dts[i][masks[j] > 0].min())
    t3 = np.sort(gap[0][1:])[:3]
    nn = gap.min(axis=1)
    xy = np.array([[o["x"], o["y"]] for o in objs])
    t4c = float(np.sort(np.linalg.norm(xy[1:] - xy[0], axis=1))[:4].sum() * 100)
    return {"t3nn_gap_mm": [round(v, 1) for v in t3],
            "t3nn_max_mm": round(float(t3.max()), 1),
            "nn_gap_median_mm": round(float(np.median(nn)), 1),
            "contact_frac": round(float((nn <= 2.0).mean()), 2),
            "t4c_sum_cm": round(t4c, 1)}


def layout_metrics(objs):
    """Whole-layout gap metrics (visual density of the full cluster):
    void_frac:   empty fraction inside the cluster's convex hull
    max_pocket_mm: radius of the largest empty pocket inside the hull
    adj_gap_p90_mm: p90 edge gap over adjacent pairs (centers < 9 cm)"""
    union = np.zeros((GRID, GRID), np.uint8)
    masks = [footprint_mask(o["name"], o["x"], o["y"], o["yaw"]) for o in objs]
    for m in masks:
        union |= m
    pts = np.column_stack(np.nonzero(union)[::-1]).astype(np.int32)
    hull = np.zeros_like(union)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    void = hull & (1 - union)
    void_frac = float(void.sum()) / max(1, int(hull.sum()))
    dt = cv2.distanceTransform((1 - union).astype(np.uint8), cv2.DIST_L2, 5)
    max_pocket = float(dt[void > 0].max()) if void.any() else 0.0
    dts = [cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
           for m in masks]
    xy = np.array([[o["x"], o["y"]] for o in objs])
    gaps = []
    n = len(objs)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(xy[i] - xy[j]) < 0.09:
                gaps.append(float(dts[i][masks[j] > 0].min()))
    p90 = float(np.percentile(gaps, 90)) if gaps else 0.0
    return {"void_frac": round(void_frac, 3),
            "max_pocket_mm": round(max_pocket, 1),
            "adj_gap_p90_mm": round(p90, 1)}


def is_motif(objs):
    n = [o["name"] for o in objs]
    return n[0] == "cylinder" and n[1] == "concave" and n[2] == "concave"


def passes_gates(objs, tm, lm, a):
    """Motif-aware quality gates: the concave clamp structurally shields the
    target (3rd neighbor blocked by the 9cm arms; voids behind them), so
    clamp scenes gate on the 2-nearest gaps and relaxed layout limits."""
    if is_motif(objs):
        t_ok = tm["t3nn_gap_mm"][1] <= a["max_t3nn_mm"]
        v_ok = lm["void_frac"] <= a["max_void_frac"] + 0.08
        p_ok = lm["max_pocket_mm"] <= a["max_pocket_mm"] + 10.0
    else:
        t_ok = tm["t3nn_max_mm"] <= a["max_t3nn_mm"]
        v_ok = lm["void_frac"] <= a["max_void_frac"]
        p_ok = lm["max_pocket_mm"] <= a["max_pocket_mm"]
    return t_ok and v_ok and p_ok and tm["t4c_sum_cm"] <= a["max_t4c_cm"]


def _gen_one(payload):
    """Worker: generate ONE gate-passing scene (for --workers pools)."""
    seed, idx, a = payload
    rng = np.random.default_rng(seed)
    while True:
        n_obj = int(rng.integers(a["num_objects"][0], a["num_objects"][1] + 1))
        objs, nb = generate_scene(rng, a["gap_mm"], a["concave_frac"],
                                  a["min_blocked"], a["center_jitter"],
                                  a["yaw_mode"], a["compact_sweeps"],
                                  a["target_gap_mm"], n_obj,
                                  motif=idx < a["motif_prob"] * a["num"],
                                  big_range=a["big_range"])
        if objs is None:
            continue
        tm0 = tightness_metrics(objs)
        lm0 = layout_metrics(objs)
        if not passes_gates(objs, tm0, lm0, a):
            continue
        return objs, nb


def preview(scenes, path, cols=8):
    tile = 160
    n = len(scenes)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, (objs, nb) in enumerate(scenes):
        img = np.zeros((GRID, GRID, 3), np.uint8)
        for o in objs:
            m = footprint_mask(o["name"], o["x"], o["y"], o["yaw"]).astype(bool)
            img[m] = tuple(int(c * 255) for c in o["color"][::-1])
        p0, p1 = to_px((CORE_WS[0][0], CORE_WS[1][0])), to_px((CORE_WS[0][1], CORE_WS[1][1]))
        cv2.rectangle(img, tuple(p0), tuple(p1), (0, 0, 255), 1)
        img = cv2.resize(img, (tile, tile))
        cv2.putText(img, f"b{nb}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
    cv2.imwrite(path, sheet)
    return path


def write_scene(objs, path):
    with open(path, "w") as f:
        for o in objs:
            r, g, b = o["color"]
            f.write(f"{o['name']}.urdf {r:.6f} {g:.6f} {b:.6f} "
                    f"{o['x']:.6f} {o['y']:.6f} {BLOCK_Z:.6f} 0.0 0.0 {o['yaw']:.6f}\n")


# --------------------------------------------------------------- settle test
def settle_validate(out_dir, steps, headless=True, snapshots=True, freeze=False,
                    chunk=128):
    """Chunked settle pass: >~200 envs with cameras in one sim segfaults
    (graphics memory), so scenes are processed 'chunk' at a time."""
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".txt"))
    all_drifts = []
    for k in range(0, len(files), chunk):
        sub = files[k:k + chunk]
        sheet = os.path.join(out_dir, "preview_sim.png" if k == 0
                             else f"preview_sim_{k // chunk}.png")
        print(f"-- settle chunk {k // chunk + 1}/{(len(files) + chunk - 1) // chunk}")
        all_drifts += _settle_chunk(out_dir, sub, steps, headless, snapshots, freeze,
                                    sheet if k == 0 else sheet.replace(".png", f"_{k}.png"))
    print(f"TOTAL: max drift {max(all_drifts) * 1000:.2f} mm over {len(files)} scenes")
    return all_drifts


def _settle_chunk(out_dir, files, steps, headless=True, snapshots=True,
                  freeze=False, sheet_path=None):
    from isaacgym import gymapi
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.dt = 1 / 60.0
    sp.substeps = 2
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 24
    sp.physx.contact_offset = 0.005
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)   # graphics dev for cameras
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.9, 0.9, 0.9),
                             gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0, 0, 0))
    for li in (1, 2, 3):
        gym.set_light_parameters(sim, li, gymapi.Vec3(0, 0, 0),
                                 gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)   # default is Y-up; sim is Z-up
    gym.add_ground(sim, plane)
    aopt = gymapi.AssetOptions()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "assets", "urdf", "more", "blocks-more")
    root = os.path.abspath(root)
    assets = {}
    envs, handles, names, cams = [], [], [], []
    for fi, fname in enumerate(files):
        env = gym.create_env(sim, gymapi.Vec3(-0.6, -0.6, 0), gymapi.Vec3(0.6, 0.6, 1),
                             int(math.ceil(math.sqrt(len(files)))))
        hs = []
        for ln in open(os.path.join(out_dir, fname)):
            p = ln.split()
            name = p[0].split(".")[0]
            if name not in assets:
                o = gymapi.AssetOptions()
                if name == "concave":
                    o.vhacd_enabled = True
                    o.vhacd_params.resolution = 64000000
                assets[name] = gym.load_asset(sim, root, f"{name}.urdf", o)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(float(p[4]), float(p[5]), float(p[6]))
            pose.r = gymapi.Quat.from_euler_zyx(float(p[7]), float(p[8]), float(p[9]))
            h = gym.create_actor(env, assets[name], pose, name, fi, 0)
            gym.set_rigid_body_color(env, h, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                     gymapi.Vec3(float(p[1]), float(p[2]), float(p[3])))
            hs.append(h)
        envs.append(env)
        handles.append(hs)
        names.append(fname)
        if snapshots:
            cp = gymapi.CameraProperties()
            cp.width, cp.height = 640, 640
            cp.horizontal_fov = 16.0     # narrow FOV from high above ->
            cam = gym.create_camera_sensor(env, cp)   # quasi-orthographic, sharp
            gym.set_camera_location(cam, env, gymapi.Vec3(0.5, 0.0, 2.0),
                                    gymapi.Vec3(0.5, 0.001, 0.0))
            cams.append(cam)

    def poses():
        out = []
        for env, hs in zip(envs, handles):
            out.append([np.array([gym.get_actor_rigid_body_states(
                env, h, gymapi.STATE_POS)["pose"]["p"][0][k] for k in "xyz"])
                for h in hs])
        return out

    for _ in range(5):
        gym.simulate(sim)
    gym.fetch_results(sim, True)
    start = poses()
    for _ in range(steps):
        gym.simulate(sim)
    gym.fetch_results(sim, True)
    end = poses()
    print(f"\n=== settle validation ({steps} steps, {len(files)} scenes) ===")
    bad = 0
    drifts = []
    for fname, s, e in zip(names, start, end):
        d = max(float(np.linalg.norm(a - b)) for a, b in zip(s, e))
        drifts.append(d)
        if d > 0.005:
            bad += 1
            print(f"  DRIFT {fname}: {d * 1000:.1f} mm")
    print(f"max drift {max(drifts) * 1000:.2f} mm, median {np.median(drifts) * 1000:.2f} mm, "
          f"scenes >5mm: {bad}/{len(files)}")
    if freeze:
        # settle-then-freeze: write settled poses back (z stays authored);
        # reject toppled objects (|roll|,|pitch| > ~11 deg)
        frozen, dropped = 0, 0
        for env, hs, fname in zip(envs, handles, names):
            lines = open(os.path.join(out_dir, fname)).read().splitlines()
            new, bad = [], False
            for h, ln in zip(hs, lines):
                st = gym.get_actor_rigid_body_states(env, h, gymapi.STATE_POS)["pose"]
                px, py = float(st["p"]["x"][0]), float(st["p"]["y"][0])
                qx, qy, qz, qw = (float(st["r"][k][0]) for k in ("x", "y", "z", "w"))
                if qx * qx + qy * qy > 0.01:
                    bad = True
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                p = ln.split()
                new.append(f"{p[0]} {p[1]} {p[2]} {p[3]} {px:.6f} {py:.6f} {p[6]} 0.0 0.0 {yaw:.6f}")
            if bad:
                os.remove(os.path.join(out_dir, fname))
                dropped += 1
            else:
                open(os.path.join(out_dir, fname), "w").write("\n".join(new) + "\n")
                frozen += 1
        print(f"freeze: rewrote {frozen} scenes with settled poses, dropped {dropped} toppled")
    if snapshots:
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        tiles = []
        for env, cam in zip(envs, cams):
            img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
            tiles.append(img.reshape(640, 640, 4)[:, :, :3])
        cols = 8
        S = 640
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * S, cols * S, 3), np.uint8)
        # snapshot camera: fov 16 deg at z=2.0 -> 0.562 m across 640 px, centred
        # on the workspace centre, so both squares are centred in the tile
        ppm = 640 / (2 * 2.0 * math.tan(math.radians(8.0)))
        ws_h, in_h = int(round(0.224 * ppm)), int(round((0.224 - WS_INSET) * ppm))
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            tile = np.ascontiguousarray(t[:, :, ::-1])
            cv2.rectangle(tile, (320 - ws_h, 320 - ws_h), (320 + ws_h, 320 + ws_h), (255, 255, 0), 2)   # workspace (OOW line)
            cv2.rectangle(tile, (320 - in_h, 320 - in_h), (320 + in_h, 320 + in_h), (90, 90, 90), 1)     # 5 cm generation inset
            cv2.rectangle(tile, (0, 0), (S - 1, S - 1), (255, 255, 255), 2)                                # scene border
            cv2.putText(tile, os.path.basename(files[i])[:-4], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            sheet[r * S:(r + 1) * S, c * S:(c + 1) * S] = tile
        cv2.imwrite(sheet_path, sheet)
        print(f"sim-render sheet: {sheet_path}")
    gym.destroy_sim(sim)
    return drifts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--out", default="test-cases/gen-pilot")
    ap.add_argument("--gap-mm", type=float, default=1.0)
    ap.add_argument("--concave-frac", type=float, default=0.3)
    ap.add_argument("--min-blocked", type=int, default=16)
    ap.add_argument("--center-jitter", type=float, default=0.03)
    ap.add_argument("--yaw-mode", choices=["random", "axis"], default="random")
    ap.add_argument("--compact-sweeps", type=int, default=2)
    ap.add_argument("--target-gap-mm", type=float, default=0.5)
    ap.add_argument("--num-objects", type=int, nargs=2, default=[11, 11],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--workers", type=int, default=1,
                    help=">1: parallel scene generation across processes")
    ap.add_argument("--motif-prob", type=float, default=0.3,
                    help="fraction of scenes with the cylinder-in-concave-clamp "
                         "motif (deterministic quota, hardest tier: the clamp "
                         "unlocks only via the seam between the two concaves)")
    ap.add_argument("--big-range", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="clutter big-object count range (concave/rect/triangle); "
                         "rest are small fillers (cube/cylinder/half-cube)")
    ap.add_argument("--max-void-frac", type=float, default=0.32,
                    help="reject scenes with more empty hull area (PMBS p90: .32)")
    ap.add_argument("--max-pocket-mm", type=float, default=30.0)
    ap.add_argument("--max-t4c-cm", type=float, default=20.0,
                    help="reject scenes whose sum of target->4NN center "
                         "distances exceeds this (PMBS hard median: 18.8)")
    ap.add_argument("--max-t3nn-mm", type=float, default=4.0,
                    help="reject scenes whose worst target-3NN edge gap exceeds this")
    ap.add_argument("--pad-to", type=int, default=0,
                    help="pad scenes to this count with dummies at y=-0.30 "
                         "(outside the 55x45 rule area -> OOW-exempt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--settle", action="store_true", help="validate existing --out dir in sim")
    ap.add_argument("--freeze", action="store_true", help="with --settle: write settled poses back")
    ap.add_argument("--settle-steps", type=int, default=120)
    args = ap.parse_args()

    if args.settle:
        settle_validate(args.out, args.settle_steps, freeze=args.freeze)
        return

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    scenes, meta = [], []
    if args.workers > 1:
        import multiprocessing as mp
        payloads = [(args.seed * 1000003 + i, i, vars(args)) for i in range(args.num)]
        with mp.Pool(args.workers) as pool:
            for objs, nb in pool.imap_unordered(_gen_one, payloads):
                sid = args.start_id + len(scenes)
                pi = 0
                while args.pad_to and len(objs) < args.pad_to:
                    objs.append({"name": "cube", "x": 0.30 + 0.06 * pi, "y": -0.45,
                                 "yaw": 0.0, "color": (0.5, 0.5, 0.5)})
                    pi += 1
                path = os.path.join(args.out, f"{sid:06d}.txt")
                write_scene(objs, path)
                qa = audit_scene(path)
                tm = tightness_metrics(objs)
                tm.update(layout_metrics(objs))
                scenes.append((objs, nb))
                names = [o["name"] for o in objs]
                meta.append({"scene": f"{sid:06d}.txt", "blocked16": nb,
                             "min_nn_cm": qa["min_nn_cm"], "classes": qa["classes"],
                             "n_big": sum(1 for x in names[1:] if x in BIG),
                             "motif": int(is_motif(objs)),
                             "max_overlap_cm2": qa["max_overlap_cm2"], **tm})
                if len(scenes) % 32 == 0:
                    print(f"  {len(scenes)}/{args.num} scenes")
    while len(scenes) < args.num:
        n_obj = int(rng.integers(args.num_objects[0], args.num_objects[1] + 1))
        objs, nb = generate_scene(rng, args.gap_mm, args.concave_frac,
                                  args.min_blocked, args.center_jitter,
                                  args.yaw_mode, args.compact_sweeps,
                                  args.target_gap_mm, n_obj,
                                  motif=len(scenes) < args.motif_prob * args.num,
                                  big_range=args.big_range)
        if objs is None:
            print("  retry batch (packing failed to satisfy constraints)")
            continue
        tm0 = tightness_metrics(objs)
        lm0 = layout_metrics(objs)
        if not passes_gates(objs, tm0, lm0, vars(args)):
            continue                      # target neighborhood or layout too loose
        pi = 0
        while args.pad_to and len(objs) < args.pad_to:
            objs.append({"name": "cube", "x": 0.30 + 0.06 * pi, "y": -0.45,
                         "yaw": 0.0, "color": (0.5, 0.5, 0.5)})
            pi += 1
        sid = args.start_id + len(scenes)
        path = os.path.join(args.out, f"{sid:06d}.txt")
        write_scene(objs, path)
        qa = audit_scene(path)
        assert qa["max_overlap_cm2"] <= 0.30, f"generator produced overlap: {qa}"  # <=5px raster aliasing vs scene_qa grid
        scenes.append((objs, nb))
        tm = tightness_metrics(objs)
        tm.update(layout_metrics(objs))
        meta.append({"scene": f"{sid:06d}.txt", "blocked16": nb,
                     "min_nn_cm": qa["min_nn_cm"], "classes": qa["classes"],
                     "max_overlap_cm2": qa["max_overlap_cm2"], **tm})
        if len(scenes) % 16 == 0:
            print(f"  {len(scenes)}/{args.num} scenes")

    sheet = preview(scenes, os.path.join(args.out, "preview.png"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "scenes": meta}, f, indent=2)

    nb = [m["blocked16"] for m in meta]
    nn = [m["min_nn_cm"] for m in meta]
    ov = [m["max_overlap_cm2"] for m in meta]
    t3m = [m["t3nn_max_mm"] for m in meta]
    cf = [m["contact_frac"] for m in meta]
    t4 = [m["t4c_sum_cm"] for m in meta]
    vf = [m["void_frac"] for m in meta]
    pk = [m["max_pocket_mm"] for m in meta]
    print(f"\n=== generated {len(scenes)} scenes -> {args.out} ===")
    print(f"blocked16   : min {min(nb)} median {int(np.median(nb))} (gate >= {args.min_blocked})")
    print(f"min NN dist : median {np.median(nn):.2f} cm  min {min(nn):.2f} cm")
    print(f"interpenetration: max {max(ov)} cm^2 (raster aliasing only; settle test is the physical gate)")
    print(f"target 3NN gap  : worst-of-3 median {np.median(t3m):.1f} mm, p90 {np.percentile(t3m, 90):.1f} mm  (PMBS hard ~3mm)")
    print(f"contact fraction: median {np.median(cf):.2f}  (manual suite ~1.0)")
    print(f"tgt->4NN centers: median {np.median(t4):.1f} cm, p90 {np.percentile(t4, 90):.1f} cm  (PMBS 18.8 / manual 20.0)")
    print(f"layout void frac: median {np.median(vf):.3f} (PMBS .18 / manual .22)  "
          f"max pocket median {np.median(pk):.1f} mm (PMBS 18)")
    print(f"preview     : {sheet}")

    if args.wandb:
        import wandb
        run = wandb.init(project="push-ret-dataset", name=os.path.basename(args.out),
                         config=vars(args))
        wandb.log({"preview": wandb.Image(sheet),
                   "blocked16": wandb.Histogram(nb),
                   "min_nn_cm": wandb.Histogram(nn),
                   "t3nn_max_mm": wandb.Histogram(t3m),
                   "t4c_sum_cm": wandb.Histogram(t4),
                   "void_frac": wandb.Histogram(vf),
                   "max_pocket_mm": wandb.Histogram(pk),
                   "contact_frac": wandb.Histogram(cf)})
        simsheet = os.path.join(args.out, "preview_sim.png")
        if os.path.exists(simsheet):
            wandb.log({"preview_sim": wandb.Image(simsheet)})
        run.finish()
        print("logged to wandb project push-ret-dataset")


if __name__ == "__main__":
    main()
