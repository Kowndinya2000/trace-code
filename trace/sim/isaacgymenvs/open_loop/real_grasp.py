"""Graspability + grasp pose from the REAL camera, PMBS-style.

After the open-loop pushes the twin's prediction is stale: the real blocks have
moved by whatever the pushes actually did, not by what the simulator thought.
So the grasp is recomputed from a fresh D455 frame, the way
the PMBS release's real_robot_main.py does it -- Mask R-CNN for instances, HSV for the
target, an orthographic heightmap, then the grasp network.

Differences from the PMBS release, both deliberate:
  * the heightmap is built with open_loop.frames (320 px canvas, 2 mm/px) rather
    than PMBS's 224 px workspace heightmap, because that is the frame this
    project's grasp network and calibration are defined in;
  * the x16 GPN helper is utils.mtcs_utils, whose winning-pixel un-rotation was
    fixed to use the actual image size (it hardcoded 224, which displaced the
    grasp by up to 272 mm on the 320 canvas).
"""
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import cv2
import torch

from isaacgymenvs.open_loop import frames
from isaacgymenvs.open_loop import perceive_scene as ps
from isaacgymenvs.open_loop.grasp_artifacts import camera_debug, save_grasp_record
from isaacgymenvs.utils.prediction_vis import (
    ACCENT_BGR, BACKGROUND_BGR, DEFAULT_TILE_SIZE, HIGH_BGR, INK_BGR, prediction_colors,
)

_ASSET_DIR = Path(__file__).resolve().parents[1]

# Match PMBS Demo's original real-robot grasp threshold. This governs the
# single clean terminal stock-GN evaluation and its grasp decision.
GRASPABLE_Q_THRESHOLD = 0.70
GPN_TARGET_RGB = (69, 108, 149)   # PMBS real_robot_main recolour
# Deliberately zero. An empirical XY correction measured against one calibration
# does not transfer to the next: carried over after a recalibration it shifted an
# independently validated camera estimate by 4 mm, moved the commanded grasp
# toward a neighbouring block and caused a finger collision. Keep the calibrated
# target centre unchanged; any correction must be re-measured against the camera
# transform actually deployed.
GRASP_XY_CORRECTION_REAL_M = np.zeros(2, dtype=np.float64)
# Neighbours are kept up to this far outside the workspace: the grasp canvas extends
# CANVAS_WS_OFFSET (48 px = 96 mm) beyond it on every side.
EDGE_CLUTTER_MARGIN_M = 0.09
TOP_FACE_BAND_M = 0.006   # keep points within 6 mm of an object's top face


def enforce_hardware_grasp_checks(grasp):
    """Accept only a valid PMBS-post-checked real grasp proposal.

    ``get_grasp_q(post_checking=True, is_real=True)`` has already masked the
    target, multi-object grasps, and outer-finger collisions before selecting
    the pixel and rotation.  This function deliberately adds no second
    geometric policy and does not replace the selected PMBS pose.
    """
    if grasp is None:
        return None

    q = float(grasp.get("q", 0.0))
    network_graspable = q >= GRASPABLE_Q_THRESHOLD
    grasp["network_graspable"] = network_graspable
    pose_fields = ("x_real", "y_real", "surface_z_m", "rotation_idx")
    checks = {
        "network_graspable": network_graspable,
        "valid_pose": all(grasp.get(key) is not None for key in pose_fields),
        "pmbs_post_checked": grasp.get("grasp_post_processing") == "pmbs_demo",
    }
    passed = all(checks.values())
    grasp["hardware_grasp_checks"] = checks
    grasp["hardware_grasp_checks_passed"] = passed
    grasp["graspable"] = passed
    if passed:
        return grasp

    failures = []
    if not checks["network_graspable"]:
        failures.append(f"GN score below {GRASPABLE_Q_THRESHOLD:.2f}")
    if not checks["valid_pose"]:
        failures.append("no valid grasp pose")
    if not checks["pmbs_post_checked"]:
        failures.append("proposal did not use PMBS post-checking")
    if not failures:
        failures.append(grasp.get("reject_reason", "shared graspability check rejected"))
    grasp["reject_reason"] = "; ".join(failures)
    print(f"[real_grasp] hardware gate: {grasp['reject_reason']}; keep pushing")
    return grasp


def apply_grasp_xy_correction(x_real, y_real):
    """Apply the measured camera-to-grasp residual in the real robot frame."""
    corrected = np.asarray([x_real, y_real], dtype=np.float64) + GRASP_XY_CORRECTION_REAL_M
    return float(corrected[0]), float(corrected[1])


def build_real_heightmap(color_bgr, depth_m, K, cam2base, instances, target_idx):
    """Orthographic top-down heightmap on the 320 px canvas, from one RGB-D view.

    Returns (rgb_canvas uint8, height_canvas float32 metres, segm_canvas int32)
    with the PMBS/closed-loop segmentation ids: target 255, others 60, 70, ...
    """
    x_s, y_s, z = ps.deproject_to_sim(depth_m, K, cam2base)
    valid = depth_m > 0

    rgb = np.zeros((frames.CANVAS_SIZE, frames.CANVAS_SIZE, 3), np.uint8)
    hgt = np.zeros((frames.CANVAS_SIZE, frames.CANVAS_SIZE), np.float32)
    seg = np.zeros((frames.CANVAS_SIZE, frames.CANVAS_SIZE), np.int32)

    rows = np.round((x_s - frames.CANVAS_ORIGIN[0]) / frames.PIXEL_SIZE)
    cols = np.round((y_s - frames.CANVAS_ORIGIN[1]) / frames.PIXEL_SIZE)
    inb = (valid & (rows >= 0) & (rows < frames.CANVAS_SIZE)
           & (cols >= 0) & (cols < frames.CANVAS_SIZE))

    for idx, inst in enumerate(instances):
        m = inst["mask"] & inb
        if not m.any():
            continue
        r = rows[m].astype(np.int32)
        c = cols[m].astype(np.int32)
        h = z[m].astype(np.float32)
        # Keep only samples near this object's TOP face. The camera is
        # perspective, so it also sees the side walls of every block, and those
        # points scatter into neighbouring grid cells -- each footprint comes out
        # fatter than the true top face, worst away from the image centre. That
        # phantom material closes the gripper's approach corridor and made the
        # grasp network zero every candidate on a graspable target.
        if len(h):
            keep = h >= (h.max() - TOP_FACE_BAND_M)
            r, c, h = r[keep], c[keep], h[keep]
        is_target = target_idx is not None and idx == target_idx
        # During continuous pushing the target may be completely hidden. Keep
        # every visible non-target object in the heightmap; there is simply no
        # target id 255 until the target is observed again.
        clutter_idx = idx if target_idx is None or idx < target_idx else idx - 1
        sid = 255 if is_target else 60 + 10 * clutter_idx
        # keep the TALLEST sample per cell: a top-down heightmap wants the top face
        order = np.argsort(h)
        r, c, h = r[order], c[order], h[order]
        hgt[r, c] = h
        seg[r, c] = sid
        if is_target:
            # Paint the target the CANONICAL SIM TARGET COLOUR, exactly as
            # the PMBS release's real_robot_main.py recolours it before the grasp network
            # (`color[mask == 255] = [69,108,149]  # blue`). The GPN masks its Q
            # map to the sim blue band [95,87,99]-[115,187,199]; the real purple
            # measures HSV ~(114,114,107), one unit inside the hue ceiling, so
            # most of its pixels fall outside and every candidate is zeroed --
            # q comes out exactly 0.0 on a target that is plainly graspable.
            # PMBS's exact recolour value, rgb(69,108,149) -> hsv(105,137,149),
            # which sits CENTRED in the grasp network's blue band
            # [95,87,99]-[115,187,199]. The twin's canonical (0.29,0.44,0.85) is
            # rgb(74,112,217) -> V 217, above the band's ceiling of 199.
            rgb[r, c] = np.array(GPN_TARGET_RGB, np.uint8)
        else:
            rgb[r, c] = color_bgr[m][order][:, ::-1]      # store RGB

    # Fill the 1-px gaps left by scattering a perspective image onto a grid, but
    # ONLY interior ones. A plain dilate also grows every object outward: at
    # 2 mm/px that is ~2 mm of phantom material per side, which closes the
    # gripper's approach corridor and made the grasp network zero every
    # candidate (q exactly 0.0 on a target that was plainly graspable; eroding
    # the clutter back 1 px restored q=0.25, 2 px gave 0.58). A morphological
    # CLOSE fills holes and then restores the boundary, so objects keep size.
    k = np.ones((3, 3), np.uint8)
    occupied = (seg > 0).astype(np.uint8)
    filled = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, k)
    new_cells = (filled > 0) & (occupied == 0)
    if new_cells.any():
        seg_d = cv2.dilate(seg.astype(np.uint16), k).astype(np.int32)
        hgt_d = cv2.dilate(hgt, k)
        rgb_d = cv2.dilate(rgb, k)
        seg = np.where(new_cells, seg_d, seg)
        hgt = np.where(new_cells, hgt_d, hgt)
        rgb = np.where(new_cells[..., None], rgb_d, rgb)
    hgt = np.clip(hgt, 0.0, None)
    return rgb, hgt, seg


def masks_to_canvas(depth_m, K, cam2base, masks, dilation_px=5):
    """Project rejected camera masks into the canonical grasp canvas.

    In a live student step these masks are predominantly robot links that Mask
    R-CNN mislabeled as blocks. The occlusion classifier was trained with a
    10 mm padded robot mask; five 2 mm canvas pixels reproduce that padding.
    """
    x_s, y_s, _ = ps.deproject_to_sim(depth_m, K, cam2base)
    rows = np.round((x_s - frames.CANVAS_ORIGIN[0]) / frames.PIXEL_SIZE).astype(np.int32)
    cols = np.round((y_s - frames.CANVAS_ORIGIN[1]) / frames.PIXEL_SIZE).astype(np.int32)
    valid = ((depth_m > 0.05) & (rows >= 0) & (rows < frames.CANVAS_SIZE) &
             (cols >= 0) & (cols < frames.CANVAS_SIZE))
    canvas = np.zeros((frames.CANVAS_SIZE, frames.CANVAS_SIZE), np.uint8)
    for mask in masks:
        use = np.asarray(mask, bool) & valid
        canvas[rows[use], cols[use]] = 1
    if dilation_px > 0 and canvas.any():
        kernel = np.ones((2*dilation_px+1, 2*dilation_px+1), np.uint8)
        canvas = cv2.dilate(canvas, kernel)
    return canvas.astype(bool)



def _compute_pmbs_postchecked_grasp(color_bgr, depth_m, K, cam2base, maskrcnn_ckpt,
                                    device="cuda", helper=None, maskrcnn=None,
                                    instances=None):
    """Return PMBS's post-checked grasp proposal, or ``None`` without a target."""
    if instances is None:
        instances = ps.segment(color_bgr, maskrcnn_ckpt, torch.device(device), model=maskrcnn)
    # Choose the target only among instances ON THE MAT. The robot arm reaches
    # into the D455 frame, Mask R-CNN labels it a block, and its steel-blue sits
    # inside the purple bands -- it scored 0.95 against the real cylinder's 0.91
    # on hardware. perceive_scene was fixed for this; this path was not, and
    # it only escaped notice because the re-sense normally runs from PMBS home
    # with the arm retracted out of shot.
    _xs, _ys, _zb = ps.deproject_to_sim(depth_m, K, cam2base)
    _valid = (depth_m > 0.05) & (_zb > 0.005)
    on_mat, near_edge = [], []
    for _k, _i in enumerate(instances):
        _sel = _i["mask"].astype(bool) & _valid
        if int(_sel.sum()) < 50:
            continue
        # Measured top-face centroid, not the pose fit: estimate_pose only sees the
        # part of a mask inside the 224 px workspace raster, so a block that sits
        # past the workspace edge came back with too few pixels and was dropped.
        _x, _y = float(_xs[_sel].mean()), float(_ys[_sel].mean())
        if frames.in_sim_workspace(_x, _y, margin=0.02):
            on_mat.append(_k)
        elif frames.in_sim_workspace(_x, _y, margin=EDGE_CLUTTER_MARGIN_M):
            near_edge.append(_k)
    # The target is chosen only among workspace instances (the PMBS-home robot
    # edge is sometimes labelled a twelfth block, in the target's colour band).
    # Neighbours are different: the black mat extends past the workspace, and on
    # a concave pressed around the target just outside it was once dropped,
    # so the grasp check saw an isolated cylinder, found 16 clear orientations and
    # drove a jaw into the concave wall. Blocks within the canvas margin whose
    # masks sit in the block-height band are kept as clutter, so PMBS's
    # post-processing collision masks see them.
    edge_blocks, _, _ = ps.filter_instances_by_depth([instances[k] for k in near_edge],
                                                     depth_m, K, cam2base)
    instances = [instances[k] for k in on_mat]
    target_idx = ps.pick_target(color_bgr, instances)
    if target_idx is None:
        return None
    instances = instances + list(edge_blocks)

    rgb, hgt, seg = build_real_heightmap(color_bgr, depth_m, K, cam2base, instances, target_idx)

    if helper is None:
        # explicit package path: a bare "utils.mtcs_utils" resolves to the hardware copy
        # when the hardware directory is on PYTHONPATH, and that copy takes torch tensors, not numpy
        from isaacgymenvs.utils.mtcs_utils import MCTSHelper
        helper = MCTSHelper(str(_ASSET_DIR / "logs_grasp/snapshot-post-020000.reinforcement.pth"),
                            str(_ASSET_DIR / "logs_grasp/grasp_model-89.pth"), device=str(device))
    # PMBS's sequential get_grasp_q, NOT the accelerated x16 variant. The x16
    # helper un-rotates only the winning PIXEL and put the grasp 111 mm away on
    # a neighbouring block; get_grasp_q un-rotates the prediction
    # MAPS and takes argmax there, which is what real_robot_main.py trusts.
    # Grasp eval runs once per re-sense, so 16 sequential passes cost nothing.
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if torch.device(device).type == "cuda" else nullcontext())
    with autocast:
        q, best, preds, _ = helper.get_grasp_q(
            rgb.astype(np.float32), hgt.astype(np.float32), seg.astype(np.int32),
            post_checking=True, is_real=True)

    q = float(q)
    out = {"q": q, "rotation_idx": int(best[0]),
           "n_instances": len(instances), "target_idx": target_idx,
           "edge_clutter_instances": len(edge_blocks),
           "grasp_post_processing": "pmbs_demo",
           "network_graspable": q >= GRASPABLE_Q_THRESHOLD,
           "_debug": {"rgb": rgb, "height": hgt, "segm": seg,
                      "preds": preds.detach().cpu().numpy(),
                      "best": [int(v) for v in best.detach().cpu().numpy()],
                      "helper": helper,
                      **camera_debug(color_bgr, depth_m, K, cam2base)}}
    if q <= 0.0:
        # every candidate was masked out (target buried, or blocked by the
        # gripper-collision check): argmax then returns index 0, which is not a
        # pose. Report ungraspable rather than a coordinate at the canvas corner.
        # Still record WHERE the target ended up -- that is the number needed to
        # compare against the twin's predicted final layout, and a run that
        # fails is exactly the run worth measuring.
        _tr, _tc = np.nonzero(seg == 255)
        if len(_tr):
            _tx, _ty = frames.canvas_pix_to_sim(_tr.mean(), _tc.mean())
            out["target_sim"] = (float(_tx), float(_ty))
            out["target_px"] = int(len(_tr))
        out.update({"px": None, "py": None, "x_sim": None, "y_sim": None,
                    "x_real": None, "y_real": None})
        return out

    px = int(best[1]) - frames.CANVAS_WS_OFFSET      # canvas -> workspace heightmap
    py = int(best[2]) - frames.CANVAS_WS_OFFSET
    x_sim, y_sim = frames.pix_to_sim(px, py)
    x_real, y_real = frames.sim_to_real(x_sim, y_sim)
    # Record the PMBS-selected point relative to the target centre for audit.
    # Do not snap it to the centre or replace its rotation: post-processing was
    # evaluated at this exact pixel and orientation.
    tr, tc = np.nonzero(seg == 255)
    if len(tr):
        tx, ty = frames.canvas_pix_to_sim(tr.mean(), tc.mean())
        out["target_sim"] = (float(tx), float(ty))
        out["grasp_offset_m"] = float(np.hypot(x_sim - tx, y_sim - ty))
    cr, cc = int(best[1]), int(best[2])
    if 0 <= cr < frames.CANVAS_SIZE and 0 <= cc < frames.CANVAS_SIZE:
        out["surface_z_m"] = float(hgt[cr, cc])          # top-face height at the grasp
    uncorrected_x_real, uncorrected_y_real = float(x_real), float(y_real)
    x_real, y_real = apply_grasp_xy_correction(x_real, y_real)
    out.update({"px": px, "py": py, "x_sim": float(x_sim), "y_sim": float(y_sim),
                "uncorrected_x_real": uncorrected_x_real,
                "uncorrected_y_real": uncorrected_y_real,
                "grasp_xy_correction_real_m": GRASP_XY_CORRECTION_REAL_M.tolist(),
                "x_real": float(x_real), "y_real": float(y_real)})
    return out


def compute_hardware_grasp(*args, **kwargs):
    """The single production entry point for a real-robot grasp decision."""
    proposal = _compute_pmbs_postchecked_grasp(*args, **kwargs)
    return enforce_hardware_grasp_checks(proposal)


def render_rotation_panel(dbg, g, path, threshold=GRASPABLE_Q_THRESHOLD,
                          tile_size=DEFAULT_TILE_SIZE, save_tiles=None):
    """Save the grid/source data, plus individual tiles for accepted grasps.

    For ``path/to/grasp.png``, accepted grasps also save
    ``path/to/grasp_tiles/bin_00.png`` through ``bin_15.png`` with captions
    and borders. Source RGB-D and Q arrays are always archived; ``save_tiles``
    explicitly overrides the acceptance condition for offline re-export.
    Return the original grid path for existing callers.

    Each tile magnifies a common target crop, preserving the bin's rotation.
    Crosshairs mark per-bin peaks without covering their centers; a gripper
    rectangle shows the selected grasp. Insets retain the complete rotating
    canvas. Output resolution scales independently of the network's Q grid.

    Added here: a caption per tile (bin, angle, that bin's max Q) and a legend,
    because a bare heat grid does not tell an audience which bin won or by how
    much. Rendering is pure post-processing -- it never runs inside a timed
    phase.
    """
    save_grasp_record(dbg, g, path, threshold=threshold, tile_size=tile_size)
    preds, rgb, best = dbg["preds"], dbg["rgb"], dbg["best"]
    q = g["q"]
    # px is None when every candidate was masked out. rotation_idx is then the
    # argmax fallback of 0, not an executed choice.
    picked = g.get("px") is not None and q >= threshold and g.get("graspable", True)
    ex = int(g.get("rotation_idx", best[0])) if picked else None
    # Public visualization has one choice. Point the renderer's gripper marker
    # at the executed orientation so the raw pre-selection maximum is not
    # exposed as a second, competing highlight.
    display_best = np.asarray(best).copy()
    if ex is not None:
        display_best[0] = ex
    else:
        # The renderer uses best[0] to decide which tile receives the gripper
        # outline.  A rejected argmax is still useful for centering/crosshairs,
        # but it must not look like an accepted grasp in the public grid.
        display_best[0] = -1
    target_mask = dbg["segm"] == 255 if "segm" in dbg else None
    canvas = dbg["helper"].get_prediction_vis(
        preds, rgb, display_best, tile_size=tile_size,
        target_mask=target_mask, threshold=threshold,
    )
    n = preds.shape[0]
    th, tw = canvas.shape[0] // 4, canvas.shape[1] // 4
    f, u = cv2.FONT_HERSHEY_SIMPLEX, max(1.0, tw / 320.0)
    header_h = int(100 * u)
    caption_h = int(44 * u)
    out = np.full((canvas.shape[0] + header_h + 4 * caption_h,
                   canvas.shape[1], 3), BACKGROUND_BGR, np.uint8)

    # Only the orientation we actually execute is called out. The raw GN argmax
    # remains in the archived data but has no distinct public visual treatment.
    if picked:
        hdr = (f"Selected: bin {ex} ({ex * 360.0 / n:.1f} deg)   |   "
               f"Q {q:.3f}   |   GRASPABLE (threshold {threshold:.2f})")
    else:
        hdr = (f"No viable grasp   |   Q {q:.3f}   |   threshold {threshold:.2f}"
               + (f"   |   {g['reject_reason']}" if g.get("reject_reason") else ""))
    cv2.putText(out, f"Grasp network  /  {n} orientations  /  target detail", (int(12*u), int(24*u)),
                f, 0.65*u, INK_BGR, max(1, int(2*u)), cv2.LINE_AA)
    sc = 0.52 * u
    while cv2.getTextSize(hdr, f, sc, max(1, int(u)))[0][0] > out.shape[1] - int(16 * u):
        sc *= 0.95
    cv2.putText(out, hdr, (int(12 * u), int(49 * u)), f, sc,
                HIGH_BGR if picked else INK_BGR,
                max(1, int(u)), cv2.LINE_AA)

    # Absolute Q colors are shared by all bins, including all-zero scenes.
    lx, ly, lw, lh = int(12*u), int(65*u), int(230*u), int(14*u)
    ramp = prediction_colors(np.linspace(0, 1, lw)[None, :])
    out[ly:ly+lh, lx:lx+lw] = ramp
    for value, label in [(0, "0"), (0.5, "0.5"), (1, "1+")]:
        x = lx + int(value*(lw-1))
        cv2.putText(out, label, (x, ly+int(29*u)), f, 0.34*u,
                    INK_BGR, max(1, int(u)), cv2.LINE_AA)
    cv2.putText(out, "Q score: brighter = higher  |  zero = no heat  |  colors saturate at 1",
                (lx+lw+int(22*u), ly+int(12*u)), f, 0.43*u,
                INK_BGR, max(1, int(u)), cv2.LINE_AA)
    cv2.putText(out, "Crosshair = peak per angle  |  yellow border + gripper = selected orientation",
                (lx+lw+int(22*u), ly+int(29*u)), f, 0.34*u,
                INK_BGR, max(1, int(u)), cv2.LINE_AA)

    for k in range(n):
        r, c = divmod(k, 4)
        x0, y0 = c * tw, header_h + r * (th + caption_h)
        out[y0:y0+th, x0:x0+tw] = canvas[r*th:(r+1)*th, c*tw:(c+1)*tw]
        qk = float(np.clip(preds[k], 0, None).max())
        run = (ex is not None and k == ex)
        txt = (f"BIN {k:02d}  |  {k * 360.0 / n:5.1f} deg  |  Q {qk:.3f}"
               + (" | SELECTED" if run else ""))
        cv2.rectangle(out, (x0, y0+th), (x0+tw-1, y0+th+caption_h-1),
                      (68, 61, 52), -1)
        text_scale = 0.48*u
        while cv2.getTextSize(txt, f, text_scale, max(1, int(u)))[0][0] > tw-int(16*u):
            text_scale *= 0.95
        cv2.putText(out, txt, (x0+int(8*u), y0+th+int(21*u)), f, text_scale,
                    HIGH_BGR if run else INK_BGR, max(1, int(u)), cv2.LINE_AA)
        high_count = int(np.count_nonzero(preds[k] >= threshold))
        detail = f"{high_count} pixels with Q >= {threshold:.2f}"
        cv2.putText(out, detail, (x0+int(8*u), y0+th+int(37*u)), f, .34*u,
                    (188, 191, 177), max(1, int(u)), cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x0+tw-1, y0+th+caption_h-1), (91, 82, 70), max(1, int(u)))
        if run:
            cv2.rectangle(out, (x0+1, y0+1), (x0+tw-2, y0+th+caption_h-2),
                          HIGH_BGR, max(2, int(3*u)))
    panel_path = Path(path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(panel_path), out):
        raise OSError(f"Could not write grasp panel: {path}")
    tiles_dir = panel_path.with_name(panel_path.stem + "_tiles")
    export_tiles = picked if save_tiles is None else save_tiles
    if not export_tiles:
        # A reused grid filename must not retain tiles from an older accepted
        # result. Only remove this renderer's exact filenames, not other files.
        if tiles_dir.is_dir():
            for k in range(n):
                old_tile = tiles_dir / f"bin_{k:02d}.png"
                if old_tile.is_file():
                    old_tile.unlink()
            if not any(tiles_dir.iterdir()):
                tiles_dir.rmdir()
        return path
    tiles_dir.mkdir(exist_ok=True)
    # Export after all grid annotations are complete so each PNG matches the
    # corresponding grid region exactly, including the highlighted bin border.
    for k in range(n):
        r, c = divmod(k, 4)
        x0, y0 = c * tw, header_h + r * (th + caption_h)
        tile = out[y0:y0+th+caption_h, x0:x0+tw]
        tile_path = tiles_dir / f"bin_{k:02d}.png"
        if not cv2.imwrite(str(tile_path), tile):
            raise OSError(f"Could not write grasp tile: {tile_path}")
    return path
