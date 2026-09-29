"""Stage 1: single external RGB-D view -> digital-twin scene file.

PMBS-style perception (user decision, Aug 2026): Mask R-CNN classifies each
object into the 6 known block classes, the purple HSV band flags the target,
poses come from mask geometry (minAreaRect + brute-force rotation IoU against
the class footprint), and the result is written in the More test-case format

    <name>.urdf  r g b  x y z  roll pitch yaw      (11 lines, target first)

so the twin needs zero env changes (MoreOpenLoop just loads the file).

Typical use (from isaacgymenvs/):

  # live capture from the external RealSense
  python open_loop/perceive_scene.py --capture --calib <cam2base_4x4.txt> \\
  # --from-dump <run_dir>  when record_cameras.py already owns the camera
      --maskrcnn <maskrcnn_ckpt.pth>

  # offline, from saved images (testing without hardware)
  python open_loop/perceive_scene.py --color scene.png --depth scene_depth.npy \\
      --intrinsics intr.json --calib <cam2base_4x4.txt> --maskrcnn <ckpt.pth>

Output: test-cases/real2sim/000000.txt (+ debug overlay and metadata JSON).

Calibration: eye-TO-hand (external camera -> robot base), the PMBS
convention — produce it with a script adapted from
trace/hardware/pmbs_calibrate_eye_to_hand.py (trace/hardware/calib_utils
has the same calibrate_eye_hand(eye_to_hand=True) machinery).

Mask R-CNN: 6 block classes + background. PMBS's checkpoint
(the PMBS release's parallel_mcts/logs_image/nmaskrcnn10.pth) is a starting
point; retrain with trace/hardware/train_maskrcnn.py + MobileSAM auto-annotation
if the new viewpoint degrades it. TODO(hardware): validate class mapping of
whichever checkpoint is used against CLASS_ID_TO_NAME below.
"""
import argparse
import json
import os

import numpy as np
import cv2
import torch

from isaacgymenvs.open_loop import frames

# class-id -> block name, PMBS convention (PMBS real_robot_main.obj_list)
CLASS_ID_TO_NAME = {1: "concave", 2: "cube", 3: "cylinder",
                    4: "half-cube", 5: "rect", 6: "triangle"}
# true silhouettes come from the block meshes' top face (one polygon each)
BLOCK_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets", "urdf", "more", "blocks-more")
ASYMMETRIC = {"concave", "triangle"}     # need the full 360-degree yaw search
# De-penetration is OFF by default. relax_penetration cannot tell "nested" from
# "interpenetrating": a cylinder seated in a concave cavity has only 1.5 mm of
# true clearance (cavity r 23.5 mm vs cylinder r 22 mm), so a couple of mm of
# pose noise reads as overlap and the block gets evicted from the cavity -- a
# 15 mm error that destroys exactly the enclosures that make clutter hard
# (measured: 37.5/39.7 mm apart with clearance 1 mm, vs 25.0/23.3
# nested and correct with 0). MoreOpenLoop.settle() already resolves genuine
# overlap against the real 3D meshes, which is strictly better than this 2D
# proxy. Set --clearance > 0 only if settle reports a large displacement.
CLEARANCE_M = 0.0                        # min footprint gap after de-penetration
# purple target band — PMBS real_purple (PMBS constants.py:137). MEASURED
# on the live D455 (3-object scene): the purple cylinder reads
# H 113-118, S 82-119, V 78-117, so the old floor of V=120 excluded it entirely
# (1.6% of mask inside the band). This band covers 89.8% of the target mask and
# 0.0% of both concave distractors. Re-measure with tools/probe if the lighting
# or the target block changes.
# Narrowed from PMBS's real_purple [90,20,70]-[150,240,255]: that
# band also swallowed a BLUE cube in the scene, and perception aborted with
# "expected exactly 1 purple target, found 2". Measured on the live D455:
#   purple cylinder (target): H 113-116, S  78-116, V  88-130
#   blue cube      (not it):  H 101-104, S 153-246, V 120-170
# Saturation is the decisive axis (110 vs 227); hue adds a second margin. Both
# bounds sit between the two populations, so either alone would separate them.
# Colour written into the TWIN for the target. Not the measured real colour:
# PMBS recolours the target to blue before the grasp network sees it, because
# the real purple block measures HSV (116,107,107), one hue unit outside the
# network's blue band -- which empties the target mask and pins grasp Q at
# ~0 forever (observed Aug 29 2026: 447 steps, final_q 0.0005). The measured
# colour is used ONLY to decide which mask is the target; it never reaches the
# twin.
SIM_TARGET_RGB = (0.29, 0.44, 0.85)
# The grasp network masks its Q map to this blue band, so a DISTRACTOR whose
# measured colour lands inside it would be treated as a second target.
SIM_TARGET_HSV_LOWER = np.array([95, 87, 99])
SIM_TARGET_HSV_UPPER = np.array([115, 187, 199])

# Accepted physical target colours. The two purple bands come from the
# operator's measured tgt_hsv_map. The blue band was measured on the live D455
# for scene2: the royal-blue cube is H103/S192-255/V130-157, while the cyan
# cylinder and cyan distractors are H83-88 and remain outside it.
TARGET_HSV_BANDS = (((100, 50, 120), (130, 200, 255)),     # purple
                    ((115, 80, 60), (145, 255, 255)),      # purple2
                    ((96, 170, 90), (110, 255, 220)))      # blue
TARGET_HSV_LOWER = np.array(TARGET_HSV_BANDS[0][0])        # kept for callers
TARGET_HSV_UPPER = np.array(TARGET_HSV_BANDS[0][1])

MIN_INSTANCE_SCORE = 0.75
MIN_INSTANCE_PIXELS = 400        # in camera image
NUM_OBJECTS = 11
# dummy blocks appended when fewer than 11 objects are detected — parked at
# the sim workspace boundary, mirroring rl_policy.py's padded_obj_centers
PAD_CENTERS_SIM = [(0.850, 0.0), (0.850, -0.108), (0.850, -0.208),
                   (0.850, -0.300), (0.850, 0.108), (0.850, 0.208),
                   (0.850, 0.300), (0.750, 0.300)]
PAD_COLOR = (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------
def settle_color_sensor(profile, pipeline, warmup=30, tag="perceive"):
    """Converge auto exposure/white balance, then freeze them.

    EVERY path that opens this camera must call this, because the HSV target
    bands are fitted to the colours it produces. With auto left on, the SAME
    purple cylinder measured H115, H146 and H149 across consecutive captures --
    a bigger swing than the gap to the blue block, so target selection flipped
    between frames. And with no setup at all, the D455 hands back a red-cast
    frame in which every instance scores 0.000 purple: that is exactly what the
    recorder did, and the in-run re-sense found no target in a
    scene that graspability then scored 1.10 through this path.

    Auto is re-enabled FIRST because the options persist on the device between
    processes -- a value frozen by a previous run under different light would
    otherwise stick. Freezing what auto converged to beats forcing a constant:
    PMBS's exposure 200 is right for their cell and far too dark for ours.
    """
    import pyrealsense2 as rs
    try:
        cs = profile.get_device().first_color_sensor()
        cs.set_option(rs.option.power_line_frequency, 2)
        cs.set_option(rs.option.enable_auto_exposure, 1)
        if cs.supports(rs.option.enable_auto_white_balance):
            cs.set_option(rs.option.enable_auto_white_balance, 1)
    except Exception as e:
        print(f"[{tag}] WARNING: could not reset exposure/white balance ({e})")
    # Auto needs frames to converge BEFORE it is frozen; a short warmup froze a
    # half-converged exposure and the target stopped matching any colour band.
    for _ in range(max(warmup, 30)):
        pipeline.wait_for_frames()
    try:
        cs = profile.get_device().first_color_sensor()
        cs.set_option(rs.option.enable_auto_exposure, 0)
        if cs.supports(rs.option.enable_auto_white_balance):
            cs.set_option(rs.option.enable_auto_white_balance, 0)
        for _ in range(5):
            pipeline.wait_for_frames()
    except Exception as e:
        print(f"[{tag}] WARNING: could not freeze exposure/WB ({e})")


def capture_realsense(warmup=30):
    """Grab one aligned RGB-D frame + intrinsics from the external RealSense."""
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        settle_color_sensor(profile, pipeline, warmup)
        fs = align.process(pipeline.wait_for_frames())
        depth_frame, color_frame = fs.get_depth_frame(), fs.get_color_frame()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        color = np.asanyarray(color_frame.get_data())  # BGR
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
    finally:
        pipeline.stop()
    return color, depth, K


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def build_maskrcnn(num_classes=len(CLASS_ID_TO_NAME) + 1):
    """PMBS architecture (PMBS train_maskrcnn.get_model_instance_segmentation):
    custom single-size anchors per FPN level + 512-wide mask head. Loads
    the upstream PMBS logs_image/nmaskrcnn10.pth strictly."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    anchors = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (256,)),
                              aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                rpn_anchor_generator=anchors)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 512, num_classes)
    return model


@torch.no_grad()
def load_maskrcnn(ckpt_path, device):
    """Build + load once. Loading inside segment() costs ~2 s per call, which
    is dead time in the middle of a timed real run; the executor preloads."""
    model = build_maskrcnn().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def segment(color_bgr, ckpt_path, device, model=None):
    if model is None:
        model = load_maskrcnn(ckpt_path, device)
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred = model([torch.from_numpy(rgb).permute(2, 0, 1).to(device)])[0]
    instances = []
    for score, label, mask in zip(pred["scores"].cpu().numpy(),
                                  pred["labels"].cpu().numpy(),
                                  pred["masks"].cpu().numpy()):
        m = (mask[0] > 0.5)
        if score < MIN_INSTANCE_SCORE or m.sum() < MIN_INSTANCE_PIXELS:
            continue
        if int(label) not in CLASS_ID_TO_NAME:
            continue
        instances.append({"class": int(label), "mask": m, "score": float(score)})
    return instances


def filter_instances_by_depth(instances, depth, K, cam2base, min_z=0.025,
                              max_z=0.065, min_fraction=0.80,
                              min_valid_pixels=50):
    """Reject mid-execution robot masks using calibrated base-frame depth.

    Every block top in this setup lies near z=0.048 m. Robot links and spurious
    off-table masks lie above or below the deliberately padded [min_z, max_z]
    band. Keep a Mask R-CNN instance only when most of its valid depth samples
    occupy that band. Rejected masks are returned for the occlusion-GN's
    unavailable channel.
    """
    _, _, z_base = deproject_to_sim(depth, K, cam2base)
    valid_depth = np.isfinite(depth) & (depth > 0.05)
    kept, rejected, diagnostics = [], [], []
    for index, inst in enumerate(instances):
        mask = np.asarray(inst["mask"], bool)
        values = z_base[mask & valid_depth]
        n_valid = int(len(values))
        if n_valid:
            in_band = (values >= min_z) & (values <= max_z)
            fraction = float(in_band.mean())
            quantiles = [float(x) for x in np.quantile(values, [0.1, 0.5, 0.9])]
        else:
            fraction, quantiles = 0.0, [None, None, None]
        accepted = n_valid >= min_valid_pixels and fraction >= min_fraction
        reason = None if accepted else (
            f"depth-band fraction {fraction:.3f} below {min_fraction:.3f}"
            if n_valid >= min_valid_pixels else
            f"only {n_valid} valid depth pixels; need {min_valid_pixels}")
        record = dict(index=index, class_id=int(inst["class"]),
                      name=CLASS_ID_TO_NAME[int(inst["class"])],
                      score=float(inst["score"]), accepted=accepted,
                      reason=reason, valid_depth_pixels=n_valid,
                      depth_band_z_m=[float(min_z), float(max_z)],
                      depth_band_fraction=fraction,
                      depth_z_p10_p50_p90_m=quantiles)
        diagnostics.append(record)
        (kept if accepted else rejected).append(inst)
    return kept, rejected, diagnostics


def purple_score(color_bgr, mask):
    """Fraction of instance pixels in an accepted purple/blue target band."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    band = np.zeros(hsv.shape[:2], bool)
    for lo, hi in TARGET_HSV_BANDS:
        band |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    inside = band[mask]
    return float(inside.mean()) if inside.size else 0.0


def pick_target(color_bgr, instances, min_score=0.15, margin=0.10):
    """Index of the target: the BEST-scoring instance, not the first over a
    threshold.

    A fixed threshold on "fraction of pixels in band" is brittle both ways. Too
    wide and a blue block qualifies (it scored 0.67-0.79 against the operator's
    own purple bands); too narrow and the real target falls under
    the bar (the purple cylinder reads H115 S116 V123, squarely in band, yet
    under half its pixels land inside it, so a >0.5 rule rejected it in 4 of 5
    captures). Ranking sidesteps the calibration problem: whichever object is
    most purple wins. `margin` refuses rather than guesses when two are close.
    """
    scored = sorted(((purple_score(color_bgr, i["mask"]), k)
                     for k, i in enumerate(instances)), reverse=True)
    if not scored or scored[0][0] < min_score:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        # Two high scorers are only ambiguous if they are two OBJECTS. Mask
        # R-CNN sometimes returns the target twice -- 12 instances for 11
        # blocks, both halves scoring purple (0.95 and 0.91) --
        # and refusing there blocks a run over a segmentation artefact, not a
        # real second target. Overlapping masks are the same block: keep the
        # larger and compare against the next DISTINCT one.
        a, b = instances[scored[0][1]]["mask"], instances[scored[1][1]]["mask"]
        inter = float(np.logical_and(a, b).sum())
        if inter / max(1.0, min(a.sum(), b.sum())) > 0.5:
            best = scored[0][1] if a.sum() >= b.sum() else scored[1][1]
            rest = [sc for sc, k in scored[2:]]
            if rest and max(scored[0][0], scored[1][0]) - rest[0] < margin:
                return None
            return best
        return None                      # two distinct candidates - do not guess
    return scored[0][1]


def is_purple(color_bgr, mask, min_score=0.15):
    return purple_score(color_bgr, mask) >= min_score


def nudge_out_of_target_band(rgb):
    """Keep a distractor out of the grasp network's blue target band.

    A real block whose measured colour happens to land inside the sim blue band
    would be counted as target pixels by the Q mask. Desaturating toward grey is
    enough to leave the band and never moves a colour INTO it (S only drops)."""
    a = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.float32) * 255.0
    hsv = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    if not np.all((hsv >= SIM_TARGET_HSV_LOWER) & (hsv <= SIM_TARGET_HSV_UPPER)):
        return tuple(float(c) for c in rgb)
    hsv[1] = max(0, int(SIM_TARGET_HSV_LOWER[1]) - 20)
    out = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    return tuple(float(c) for c in out)


# --------------------------------------------------------------------------
# geometry: camera pixels -> sim-frame heightmap
# --------------------------------------------------------------------------
def deproject_to_sim(depth, K, cam2base):
    """Per-camera-pixel sim-frame (x_s, y_s, z) via calibration + swap-negate."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (us - K["cx"]) * z / K["fx"]
    y = (vs - K["cy"]) * z / K["fy"]
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)      # (h, w, 4)
    pts_base = pts_cam @ cam2base.T                              # (h, w, 4)
    x_s = -(pts_base[..., 1] - frames.REAL_FRAME_OFFSET[1])
    y_s = pts_base[..., 0] - frames.REAL_FRAME_OFFSET[0]
    return x_s, y_s, pts_base[..., 2]


def mask_to_sim_pixels(mask, valid, x_s, y_s):
    """Camera-image mask -> (N, 2) integer heightmap pixel coords in sim frame."""
    sel = mask & valid
    px = np.round((x_s[sel] - frames.SIM_WORKSPACE_LIMITS[0, 0]) / frames.PIXEL_SIZE)
    py = np.round((y_s[sel] - frames.SIM_WORKSPACE_LIMITS[1, 0]) / frames.PIXEL_SIZE)
    ok = (px >= 0) & (px < frames.IMAGE_SIZE) & (py >= 0) & (py < frames.IMAGE_SIZE)
    return np.stack([px[ok], py[ok]], axis=1).astype(np.int32)


_SIL_CACHE = {}


def silhouette_polygon(block_name):
    """(N, 2) metres: the top-face polygon of <name>.obj at yaw 0, centred
    on the block origin — the same mesh Isaac Gym simulates."""
    if block_name not in _SIL_CACHE:
        verts, faces = [], []
        with open(os.path.join(BLOCK_ASSET_DIR, block_name + ".obj")) as f:
            for line in f:
                t = line.split()
                if t and t[0] == "v":
                    verts.append([float(v) for v in t[1:4]])
                elif t and t[0] == "f":
                    faces.append([int(v.split("/")[0]) - 1 for v in t[1:]])
        verts = np.asarray(verts)
        zmax = verts[:, 2].max()
        top = [fc for fc in faces if all(abs(verts[i, 2] - zmax) < 1e-6 for i in fc)]
        assert len(top) == 1, f"{block_name}.obj: expected one top face, got {len(top)}"
        _SIL_CACHE[block_name] = verts[top[0], :2].copy()
    return _SIL_CACHE[block_name]


def raster_silhouette(canvas, block_name, x_px, y_px, yaw, res, value=1):
    """fillPoly the block silhouette into canvas[x_px_axis, y_px_axis]
    (row = sim x, col = sim y, `res` metres per pixel)."""
    poly = silhouette_polygon(block_name)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = poly @ np.array([[c, s], [-s, c]])                 # rotate by yaw
    pts = np.round(rot / res + np.array([x_px, y_px])).astype(np.int32)
    cv2.fillPoly(canvas, [pts[:, ::-1]], value)             # fillPoly wants (col, row)
    return canvas


def rasterize_located_target(objects):
    """Canonical sim-like target mask from the current partial pose estimate."""
    canvas = np.zeros((frames.CANVAS_SIZE, frames.CANVAS_SIZE), np.uint8)
    for obj in objects:
        if not obj.get("target", False):
            continue
        row, col = frames.sim_to_canvas_pix(obj["x"], obj["y"])
        raster_silhouette(canvas, obj["name"], row, col, obj["yaw"],
                          frames.PIXEL_SIZE, value=1)
    return canvas.astype(bool)


def relax_penetration(objects, clearance=CLEARANCE_M, res=0.0005, max_iter=80):
    """Nudge overlapping footprints apart (0.5 mm steps along the centre-
    centre direction, both objects) until every pair has >= `clearance`.
    Isaac Gym's depenetration (max 1000 m/s) hurls interpenetrating blocks
    at reset; the perceived poses of touching objects always overlap by
    1-3 mm. Returns the per-object displacement in metres."""
    n = len(objects)
    x0 = np.array([[o["x"], o["y"]] for o in objects])
    xy = x0.copy()
    size = int(0.16 / res)                                   # local canvas per object
    k = max(1, int(round(clearance / 2 / res)))              # dilate each by half the gap
    kern = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    for it in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = xy[j] - xy[i]
                dist = np.linalg.norm(d)
                if dist > 0.14:
                    continue
                mid = (xy[i] + xy[j]) / 2
                cvs = np.zeros((2, size, size), np.uint8)
                for m, idx in enumerate((i, j)):
                    o = objects[idx]
                    px = (xy[idx] - mid) / res + size / 2
                    raster_silhouette(cvs[m], o["name"], px[0], px[1], o["yaw"], res)
                    cvs[m] = cv2.dilate(cvs[m], kern)
                if (cvs[0] & cvs[1]).any():
                    step = (d / dist if dist > 1e-6 else np.array([1.0, 0.0])) * res
                    xy[i] -= step
                    xy[j] += step
                    moved = True
        if not moved:
            break
    for o, p in zip(objects, xy):
        o["x"], o["y"] = float(p[0]), float(p[1])
    return np.linalg.norm(xy - x0, axis=1)


def estimate_pose(sim_pixels, block_name):
    """SE(2) pose from the projected mask: yaw by brute-force rotation IoU of
    the TRUE silhouette (mesh top face) against the mask -- the PMBS-style
    template match -- with the template placed so its pixel centroid matches
    the mask's, and the block origin read back from that placement.

    Not minAreaRect's centre: the triangle is a right isosceles, so two
    rectangles of equal area enclose it (the 9x4.5 bbox and a 6.4x6.4 square
    on a leg) and the square's centre is 2.25 cm off the mesh origin. Which
    one OpenCV returns is a coin flip on a real mask, and the template then
    lands 2.25 cm off: every real triangle fitted at IoU ~0.5 with a wrong yaw
    (demo40 twin layout). Centroid matching has no such ambiguity, and it
    is exact for every mesh, whatever its origin (the concave's centroid is
    7 mm off its origin too).

    Symmetric shapes search +-90 deg around the minAreaRect angle in 1-degree
    steps; concave / triangle search the full circle (resolves the 180-degree
    flip).
    """
    rect = cv2.minAreaRect(sim_pixels.astype(np.float32))
    base_yaw = np.deg2rad(rect[2])

    tile = int(np.ceil(0.10 / frames.PIXEL_SIZE))
    obs = np.zeros((2 * tile, 2 * tile), np.uint8)
    cxi, cyi = np.floor(sim_pixels.mean(axis=0)).astype(int)
    centered = sim_pixels.astype(int) - np.array([cxi, cyi])
    inside = (np.abs(centered[:, 0]) < tile) & (np.abs(centered[:, 1]) < tile)
    obs[centered[inside, 0] + tile, centered[inside, 1] + tile] = 1
    m_obs = np.argwhere(obs > 0).mean(axis=0)                # mask centroid, tile frame

    span = 180 if block_name in ASYMMETRIC else 90
    best_yaw, best_iou, best_org = base_yaw, -1.0, np.array([tile, tile], float)
    for dyaw in np.deg2rad(np.arange(-span, span + 1, 1)):
        yaw = base_yaw + dyaw
        tmpl = raster_silhouette(np.zeros_like(obs), block_name, tile, tile, yaw, frames.PIXEL_SIZE)
        org = np.array([tile, tile], float) + m_obs - np.argwhere(tmpl > 0).mean(axis=0)
        tmpl = raster_silhouette(np.zeros_like(obs), block_name, org[0], org[1], yaw, frames.PIXEL_SIZE)
        inter = np.logical_and(obs, tmpl).sum()
        union = np.logical_or(obs, tmpl).sum()
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_iou, best_yaw, best_org = iou, yaw, org
    best_yaw = float((best_yaw + np.pi) % (2 * np.pi) - np.pi)
    x_sim, y_sim = frames.pix_to_sim(cxi + best_org[0] - tile, cyi + best_org[1] - tile)
    return x_sim, y_sim, best_yaw, float(best_iou)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
# A mask overlapping this much of the SMALLER of two detections is the same
# block found twice, not two blocks standing side by side.
DUPLICATE_CONTAINMENT = 0.5
# A real block matches its own silhouette closely once the pose fit has run:
# every genuine detection on hardware scores 0.92-0.97. A sliver detection -- a
# quarter of a block's footprint, overlapping nothing, so the duplicate filter
# cannot see it -- fits at around 0.21 and aborts the run. The 50-valid-pixel
# floor below is far too low to catch that on its own.
MIN_FIT_IOU = 0.5


def locate_objects(color, depth, K, cam2base, model=None, table_z=0.005,
                   maskrcnn_ckpt=None, device=None, require_target=True,
                   instances=None, expected=None):
    """Segment + locate every block on the mat, then pick the target among them.

    -> (objects, tgt_idx). objects is a list of dicts {name, x, y, yaw, target,
    ...} in the SIM frame; tgt_idx indexes it, or is None when no unambiguous
    purple target is on the mat and require_target is False. The closed-loop
    student needs that tolerance: with the arm over the scene the target is
    often occluded, and an occluded target is a visibility-0 token, not an
    error. The CLI keeps require_target=True and refuses.
    """
    import torch as _torch
    if device is None:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if instances is None:
        instances = segment(color, maskrcnn_ckpt, device, model=model)
    print(f"[perceive] {len(instances)} instances above threshold")
    x_s, y_s, z_base = deproject_to_sim(depth, K, cam2base)
    valid = (depth > 0.05) & (z_base > table_z)
    # Locate every instance FIRST, and only then pick the target among the ones
    # that are actually on the mat. The robot arm intrudes into the top-right of
    # the frame, Mask R-CNN labels it a block, and its steel-blue falls inside
    # the purple bands -- it scored 0.95 against the cylinder's
    # 0.91 and the ambiguity guard refused the whole run. Selecting before the
    # workspace filter let a thing that is not on the mat compete.
    located = []
    duplicates = slivers = 0
    for k_inst, inst in enumerate(instances):
        sim_px = mask_to_sim_pixels(inst["mask"], valid, x_s, y_s)
        if len(sim_px) < 50:
            continue
        name = CLASS_ID_TO_NAME[inst["class"]]
        x_sim, y_sim, yaw, iou = estimate_pose(sim_px, name)
        if not frames.in_sim_workspace(x_sim, y_sim, margin=0.02):
            continue
        located.append((k_inst, inst, name, x_sim, y_sim, yaw, iou, len(sim_px)))
    # Mask R-CNN's NMS runs on BOXES, so a partial mask nested inside a larger
    # detection of the SAME block survives it: a second concave (score 0.962,
    # 1498 px) lying inside the real one (1.000, 3679 px) persisted across
    # consecutive captures, and a block count of 12 aborts the run before it
    # can move. Suppress by CONTAINMENT -- overlap over the
    # smaller mask -- not IoU, which stays small when one mask is a third the
    # size of the other. Keep the more confident detection, larger one first on
    # a tie.
    located.sort(key=lambda item: (-item[1]["score"], -item[7]))
    survivors = []
    for item in located:
        mask = item[1]["mask"].astype(bool)
        area = int(mask.sum())
        if any(int((mask & other).sum()) > DUPLICATE_CONTAINMENT * min(area, int(other.sum()))
               for other in (kept[1]["mask"].astype(bool) for kept in survivors)):
            duplicates += 1
            continue
        survivors.append(item)
    if duplicates:
        print(f"[perceive] suppressed {duplicates} duplicate mask(s) nested in a "
              f"higher-scoring detection")
    located = survivors
    # Poor silhouette fit alone cannot condemn a detection: today's phantoms fitted
    # at 0.19-0.36 but a HALF-OCCLUDED REAL cube fitted at 0.47, and dropping it
    # aborted a scene 7 run at 10 objects. So prune by fit only while there are
    # MORE detections than the scene should have, worst fit first -- a phantom is
    # by definition a surplus. With no surplus, every detection survives.
    if expected is not None and len(located) > expected:
        ranked = sorted(located, key=lambda item: item[6])
        doomed = [item for item in ranked[:len(located) - expected] if item[6] < MIN_FIT_IOU]
        for item in doomed:
            slivers += 1
            print(f"[perceive] dropped a {item[2]} fitting its silhouette at "
                  f"{item[6]:.2f} over {item[7]} px (score {item[1]['score']:.2f}); "
                  f"{len(located)} detections for {expected} objects")
        if doomed:
            located = [item for item in located if item not in doomed]
    tgt_local = pick_target(color, [ln[1] for ln in located]) if located else None
    if tgt_local is None and require_target:
        raise SystemExit("[perceive] no unambiguous purple/blue target — check the "
                         "scene or the HSV bands (scores: " +
                         ", ".join(f"{purple_score(color, ln[1]['mask']):.2f}"
                                   for ln in located) + ")")
    tgt_idx = located[tgt_local][0] if tgt_local is not None else None
    objects = []
    for k_inst, inst, name, x_sim, y_sim, yaw, iou, n_px in located:
        bgr = color[inst["mask"]].mean(axis=0) / 255.0
        is_target = (k_inst == tgt_idx)
        rgb = SIM_TARGET_RGB if is_target else nudge_out_of_target_band(
            (bgr[2], bgr[1], bgr[0]))
        objects.append({
            "name": name, "x": x_sim, "y": y_sim, "yaw": yaw,
            "color": rgb,                               # rgb 0-1
            "measured_color": (bgr[2], bgr[1], bgr[0]),  # what the camera saw
            "target": is_target,
            "score": inst["score"], "fit_iou": iou, "pixels": int(n_px),
            "mask": inst["mask"],
        })
    return objects, (tgt_local if tgt_local is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", action="store_true", help="grab a live RealSense frame")
    src.add_argument("--color", help="offline color image (BGR png/jpg)")
    src.add_argument("--from-dump", metavar="REC_DIR",
                     help="ask a running record_cameras.py for the frame instead of "
                          "opening the camera. Required whenever the cameras are "
                          "already recording -- a RealSense opens once, and going "
                          "through the recorder also guarantees perception and the "
                          "re-sense see an identically configured sensor.")
    ap.add_argument("--depth", help="offline depth (metres): .npy, or 16-bit png in mm")
    ap.add_argument("--intrinsics", help="offline intrinsics json {fx,fy,cx,cy}")
    ap.add_argument("--calib", required=True, help="4x4 camera-to-base txt (eye-to-hand)")
    ap.add_argument("--maskrcnn", required=True, help="Mask R-CNN checkpoint (6 classes + bg)")
    ap.add_argument("--out-dir", default="test-cases/real2sim")
    ap.add_argument("--scene-id", type=int, default=0, help="output NNNNNN index")
    ap.add_argument("--table-z", type=float, default=0.005,
                    help="min base-frame z (m) for a point to count as an object")
    ap.add_argument("--clearance", type=float, default=CLEARANCE_M,
                    help="min footprint gap (m) enforced by de-penetration; "
                         "0 (default) disables it and lets settle() resolve overlap")
    ap.add_argument("--frame-offset", type=float, nargs=2, metavar=("DX", "DY"),
                    help="override frames.REAL_FRAME_OFFSET (m) for this capture")
    ap.add_argument("--dump-name", default="d455_topdown", help="recorder --name of the D455")
    args = ap.parse_args()
    if args.frame_offset is not None:
        frames.REAL_FRAME_OFFSET[:] = args.frame_offset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.capture:
        color, depth, K = capture_realsense()
    elif args.from_dump:
        from isaacgymenvs.open_loop import recorder_io
        recorder_io.mark(args.from_dump, "perceive", "capturing scene")
        color, depth, K, base = recorder_io.request_dump(args.from_dump, args.dump_name)
        print(f"[perceive] frame from recorder: {base}")
    else:
        assert args.depth and args.intrinsics, "--color needs --depth and --intrinsics"
        color = cv2.imread(args.color)
        depth = (np.load(args.depth) if args.depth.endswith(".npy")
                 else cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0)
        with open(args.intrinsics) as f:
            K = json.load(f)
        if "cam_intr" in K:                       # recorded-capture config.json style
            m = K["cam_intr"]
            K = {"fx": m[0][0], "fy": m[1][1], "cx": m[0][2], "cy": m[1][2]}

    cam2base = np.loadtxt(args.calib)
    assert cam2base.shape == (4, 4), f"calibration must be 4x4, got {cam2base.shape}"


    objects, _tgt = locate_objects(color, depth, K, cam2base, model=None,
                                   table_z=args.table_z, maskrcnn_ckpt=args.maskrcnn,
                                   device=torch.device(device), require_target=True)
    if objects:
        xs, ys = [o["x"] for o in objects], [o["y"] for o in objects]
        print(f"[perceive] {len(objects)} objects in the sim workspace: x [{min(xs):.3f}, {max(xs):.3f}] "
              f"y [{min(ys):.3f}, {max(ys):.3f}] (frame offset {frames.REAL_FRAME_OFFSET.tolist()})")
    targets = [o for o in objects if o["target"]]
    assert len(targets) == 1, \
        f"expected exactly 1 purple/blue target, found {len(targets)} — check HSV band / scene"
    ordered = targets + [o for o in objects if not o["target"]]

    if len(ordered) > NUM_OBJECTS:
        print(f"[perceive] WARNING: {len(ordered)} objects, keeping the {NUM_OBJECTS} largest clutter")
        ordered = ordered[:1] + sorted(ordered[1:], key=lambda o: -o["pixels"])[:NUM_OBJECTS - 1]
    pads = 0
    while len(ordered) < NUM_OBJECTS:
        px, py = PAD_CENTERS_SIM[pads]
        ordered.append({"name": "cube", "x": px, "y": py, "yaw": 0.0,
                        "color": PAD_COLOR, "target": False, "pad": True})
        pads += 1
    if pads:
        print(f"[perceive] padded with {pads} dummy cubes at the workspace boundary")

    if args.clearance > 0:
        disp = relax_penetration(ordered, clearance=args.clearance)
        print(f"[perceive] de-penetration: moved {int((disp > 1e-6).sum())} objects, "
              f"max {1000 * disp.max():.1f} mm (clearance {1000 * args.clearance:.1f} mm)")

    os.makedirs(args.out_dir, exist_ok=True)
    scene_path = os.path.join(args.out_dir, f"{args.scene_id:06d}.txt")
    with open(scene_path, "w") as f:
        for o in ordered:
            r, g, b = o["color"]
            f.write(f"{o['name']}.urdf {r:.6f} {g:.6f} {b:.6f} "
                    f"{o['x']:.6f} {o['y']:.6f} {frames.BLOCK_Z:.6f} "
                    f"0.0 0.0 {o['yaw']:.6f}\n")

    meta_path = os.path.join(args.out_dir, f"{args.scene_id:06d}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"objects": [{k: v for k, v in o.items() if k != "mask"} for o in ordered],
                   "calib": os.path.abspath(args.calib),
                   "maskrcnn": os.path.abspath(args.maskrcnn)}, f, indent=2)

    # debug overlay: detected instances + target flag on the camera image
    dbg = color.copy()
    for o in objects:
        ys, xs = np.nonzero(o["mask"])
        if len(xs) == 0:
            continue
        col = (255, 0, 255) if o["target"] else (0, 255, 0)
        cv2.rectangle(dbg, (xs.min(), ys.min()), (xs.max(), ys.max()), col, 2)
        cv2.putText(dbg, f"{o['name']} {np.rad2deg(o['yaw']):.0f}deg",
                    (xs.min(), max(0, ys.min() - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    dbg_path = os.path.join(args.out_dir, f"{args.scene_id:06d}_debug.png")
    cv2.imwrite(dbg_path, dbg)
    print(f"[perceive] scene: {scene_path}\n[perceive] meta:  {meta_path}\n"
          f"[perceive] debug: {dbg_path}")


if __name__ == "__main__":
    main()
