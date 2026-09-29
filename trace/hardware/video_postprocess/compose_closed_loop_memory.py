#!/usr/bin/env python3
"""master_v4 closed-loop layout plus the student's object memory and plan window.

Everything in compose_closed_loop.py (master_v4) is kept, with these additions:

* The D455 panel shrinks to 520x390; a legend for its overlay sits beside it.
* The twin's teacher plan is drawn on the D455 panel: the whole nominal sequence
  faintly (it is given all at once), the look-ahead window the student reads at
  each step (student_obs.encode_plan: next PLAN_LOOKAHEAD waypoints) brightly,
  the student's executed EEF path in cyan, and its deviation from the plan.
* An Object Memory timeline fills the column under it: one row per tracked
  object (its mask from the first scene estimate, target first), one column per
  student step, growing to the right because the recurrent memory has no fixed
  length. Solid = observed that step; striped = occluded, pose carried in memory;
  the badge counts steps since the object was last observed.
* The digital-twin tile holds its final frame with the rollout's EEF trace in pink.

Source recordings are read only; the output is always a new file.
"""

from __future__ import annotations

import argparse
from camera_sources import third_person
import glob
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import compose_closed_loop as base
import exposure
from compose_closed_loop import (
    CANVAS_H, CANVAS_W, DASH_H, DASH_W, DASH_Y, LABEL_D455_W, LABEL_GN_W, LABEL_OBS_W,
    LABEL_SIM_W, MARGIN, OBS_H, OBS_LABEL_H, OBS_W, OBS_X, OBS_Y, SIM_CONTENT_SIZE,
    SIM_INNER_SIZE, SIM_INNER_X, SIM_INNER_Y, SIM_WORKSPACE_LABEL_H, SOLVE_OUT_W,
    SOURCE_SOLVE_W, SOURCE_SOLVE_X, SPEED_BADGE_H, STANDARD_BAND_H, TILE_H, TILE_LABEL_H,
    TILE_W, TILE_X1, TILE_X2, TILE_Y1, TILE_Y2, EXECUTE_OUT_W, TOTAL_OUT_W,
    PERCEPTION_OUT_X, SOLVE_OUT_X, EXECUTE_OUT_X, PERCEPTION_OUT_W, TITLE_OUT_W,
    SPEED_BADGE_W,
)

# Left column: the D455 (4:3) with its overlay legend to the right, and the
# memory timeline below, ending level with the bottom tiles.
COLUMN_X, COLUMN_W = 30, 800
MAIN_W, MAIN_H = 520, 390
MAIN_X, MAIN_Y = COLUMN_X, TILE_Y1
LEGEND_X, LEGEND_Y = MAIN_X + MAIN_W + 14, MAIN_Y
LEGEND_W, LEGEND_H = COLUMN_X + COLUMN_W - LEGEND_X, MAIN_H
MEM_X, MEM_Y = COLUMN_X, MAIN_Y + MAIN_H + 14
MEM_W, MEM_H = COLUMN_W, TILE_Y2 + TILE_H - MEM_Y
MEM_HEADER_H = 58
THUMB_H = 20

PLAN_LOOKAHEAD = 4          # isaacgymenvs/open_loop/student_obs.py
PLAN_Z = 0.020              # trace height used by record_cameras' overlays
PINK = (255, 64, 160)
CYAN = (34, 211, 238)
PANEL_BG = "#202124"
TEXT, SUBTEXT, RULE = "#F1F3F4", "#9AA0A6", "#5F6368"
# Object-memory palettes, designed against the rest of the frame (dark #202124
# dashboard, #111820 canvas, soft state colours). Masks are lifted toward white by
# thumb_lift so they read on a dark panel.
PALETTES = {
    "dashboard": dict(bg="#202124", ink="#E8EAED", subink="#9AA0A6", empty=(45, 46, 49),
                      stripe=(110, 114, 119), observed=(129, 201, 149), cursor="#FDD663",
                      badge="#3C4043", badge_ink="#E8EAED", thumb_lift=0.45),
    "slate": dict(bg="#263039", ink="#E6EDF3", subink="#93A1AD", empty=(49, 61, 71),
                  stripe=(122, 137, 150), observed=(138, 180, 248), cursor="#F28B82",
                  badge="#18212A", badge_ink="#E6EDF3", thumb_lift=0.45),
    "warm": dict(bg="#2A2623", ink="#F1EBE4", subink="#B3A89C", empty=(58, 52, 47),
                 stripe=(142, 128, 114), observed=(215, 168, 110), cursor="#8AB4F8",
                 badge="#171412", badge_ink="#F1EBE4", thumb_lift=0.45),
}


def apply_palette(name: str) -> None:
    global MEM_BG, MEM_INK, MEM_SUBINK, MEM_EMPTY, STRIPE, OBSERVED, CURSOR, BADGE, BADGE_INK, THUMB_LIFT
    pal = PALETTES[name]
    MEM_BG, MEM_INK, MEM_SUBINK, MEM_EMPTY = pal["bg"], pal["ink"], pal["subink"], pal["empty"]
    STRIPE, OBSERVED, CURSOR = pal["stripe"], pal["observed"], pal["cursor"]
    BADGE, BADGE_INK, THUMB_LIFT = pal["badge"], pal["badge_ink"], pal["thumb_lift"]


apply_palette("dashboard")
SIM_FRAME_LO, SIM_FRAME_SPAN = 570, 2700   # 0.448 m workspace inside the 3840 render
RAW_W, RAW_H, RAW_CROP_X = 1280, 720, 320  # D455 raw; composer keeps the right 75 %
LATO = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
LATO_SEMI = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"


# --------------------------------------------------------------- trial data
def load_student_trial(trial: Path, phases: dict, video_zero_offset: float) -> dict:
    meta = trial / "metadata"
    traj = base.load_json(meta / "trajectory.json")
    timing = base.load_json(meta / "real_timing.json")
    steps = [e for e in timing["log"] if e.get("kind") == "primitive"]
    t0 = phases["t0"]
    step_times = [e["t"] - t0 - video_zero_offset for e in phases["events"]
                  if e.get("phase") == "student" and (
                      str(e.get("detail", "")).startswith("internal: student step;") or
                      str(e.get("detail", "")).startswith("Occlusion-aware grasp classifier score:"))]
    step_times = step_times[:len(steps)]
    if len(step_times) != len(steps):
        raise SystemExit(f"{len(steps)} logged steps but {len(step_times)} step events")

    dense = traj["dense"]
    start = traj["start_eef"][:2]
    # run_student.load_plan: waypoint j is the nominal EEF pose at step j.
    plan = np.asarray([start] + [d["eef"][:2] for d in dense[:-1]], float)
    rollout_trace = np.asarray([start] + [d["eef"][:2] for d in dense], float)
    objects = [line.split() for line in traj["scene_text"].strip().splitlines()]

    vis = np.asarray([s["vis"] for s in steps], int)
    stale = np.zeros(vis.shape, int)
    for t in range(len(steps)):         # student_obs.step_staleness, applied after step t
        stale[t] = np.where(vis[t] > 0, 0, (stale[t - 1] if t else 0) + 1)
    return {"plan": plan, "trace": rollout_trace, "objects": objects, "vis": vis,
            "stale": stale, "eef": np.asarray([s["eef_sim"][:2] for s in steps], float),
            "step_times": step_times}


def camera_projection(trial: Path):
    records = sorted(glob.glob(str(trial / "perception" / "*_gn16_predictions.npz")))
    if not records:
        raise SystemExit("No grasp-network record with camera_to_base under perception/")
    with np.load(records[-1], allow_pickle=False) as data:
        b2c = np.linalg.inv(data["camera_to_base"])
    K = base.load_json(trial / "perception" / "d455_topdown_dump0_K.json")
    scale = MAIN_W / (RAW_W - RAW_CROP_X)

    def to_panel(x_sim: float, y_sim: float) -> tuple[float, float]:
        p = b2c @ np.array([y_sim, -x_sim, PLAN_Z, 1.0])      # record_cameras._project
        u = p[0] * K["fx"] / p[2] + K["cx"]
        v = p[1] * K["fy"] / p[2] + K["cy"]
        # crop the right 75 %, then hflip + vflip, then scale into the panel
        return ((RAW_W - 1 - u) * scale, (RAW_H - 1 - v) * scale)
    return to_panel


def object_thumbnails(trial: Path, objects: list, size: tuple) -> list:
    """Each track's mask from the first scene estimate, oriented like the D455 panel."""
    from PIL import Image

    masks_path = sorted(glob.glob(str(trial / "perception" / "*" / "camera_masks_step_000.npz")) +
                        glob.glob(str(trial / "perception" / "camera_masks_step_000.npz")) +
                        glob.glob(str(trial / "observations" / "camera_masks_step_000.npz")))
    if not masks_path:
        raise SystemExit("No camera_masks_step_000.npz for the initial scene")
    with np.load(masks_path[0], allow_pickle=False) as data:
        segm, rgb = data["scene_segm"], data["scene_rgb"]
    crops, colors = [], []
    for obj in objects:
        x, y = float(obj[4]), float(obj[5])
        row, col = int(round((x - 0.180) / 0.002)), int(round((y + 0.320) / 0.002))
        window = segm[max(0, row - 4):row + 5, max(0, col - 4):col + 5]
        ids, counts = np.unique(window[window > 0], return_counts=True)
        if not len(ids):
            crops.append(None); colors.append((154, 160, 166)); continue
        mask = segm == ids[counts.argmax()]
        rows, cols = np.where(mask)
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask, :3], rgba[mask, 3] = rgb[mask], 255
        crop = rgba[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        # canvas raster -> record_cameras tile (90 deg CW) -> composer (180 deg) = 90 deg CCW
        crops.append(Image.fromarray(crop).rotate(90, expand=True))
        colors.append(tuple(int(c) for c in rgb[mask].mean(axis=0)))
    # One scale for all objects so their sizes compare, fitted to a short, wide row cell.
    width, height = size
    s = min(width / max(c.width for c in crops if c is not None),
            height / max(c.height for c in crops if c is not None))
    thumbs = []
    for crop in crops:
        if crop is None:
            thumbs.append(None); continue
        thumbs.append(crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))),
                                  Image.Resampling.LANCZOS))
    return thumbs, colors


# ------------------------------------------------------------ memory timeline
def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line); line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def hatch_cell(w: int, h: int, color: tuple):
    """Striped cell: occluded, so the student relies on the last known pose."""
    from PIL import Image, ImageDraw

    cell = Image.new("RGBA", (w, h), MEM_EMPTY + (255,))
    draw = ImageDraw.Draw(cell)
    stripe = tuple(color[:3]) + (255,)
    for x in range(-h, w + h, 6):
        draw.line((x, h, x + h, 0), fill=stripe, width=2)
    draw.rectangle((0, 0, w - 1, h - 1), outline=stripe, width=1)
    return cell


def memory_geometry(n_objects: int, n_steps: int) -> dict:
    thumb_x, thumb_w = 14, 50
    raster_x = thumb_x + thumb_w + 10
    badge_w = 42
    raster_w = MEM_W - raster_x - badge_w - 18
    step_w = max(6, min(34, raster_w // max(1, n_steps)))
    pitch = (MEM_H - MEM_HEADER_H - 10) // n_objects
    return {"thumb_x": thumb_x, "thumb_w": thumb_w, "raster_x": raster_x, "step_w": step_w,
            "row_top": MEM_HEADER_H, "pitch": pitch, "cell_h": pitch - 5,
            "badge_x": MEM_W - badge_w - 12, "badge_w": badge_w}


def draw_memory_header(draw):
    from PIL import ImageFont

    title = ImageFont.truetype(LATO_SEMI, 15)
    small = ImageFont.truetype(LATO, 13)
    heading = "TRACE'S OBJECT MEMORY"
    draw.text((14, 9), heading, font=title, fill=MEM_INK)
    draw.text((14 + draw.textlength(heading, font=title) + 12, 11),
              "one row per tracked object  \u00b7  one column per step, growing with the episode",
              font=small, fill=MEM_SUBINK)
    return small


def draw_memory(step: int | None, data: dict, thumbs: list, colors: list):
    from PIL import Image, ImageDraw, ImageFont

    n = len(thumbs)
    vis, stale = data["vis"], data["stale"]
    g = memory_geometry(n, len(vis))
    panel = Image.new("RGB", (MEM_W, MEM_H), MEM_BG)
    draw = ImageDraw.Draw(panel)
    small = draw_memory_header(draw)
    badge_font = ImageFont.truetype(LATO_SEMI, 13)

    # Legend row: every mark used below, named (12 px so the full row fits 800 px).
    legend = ImageFont.truetype(LATO, 12)
    lx, ly = 14, 34
    gap = 12
    draw.rounded_rectangle((lx, ly + 2, lx + 22, ly + 14), radius=2, fill=OBSERVED)
    draw.text((lx + 28, ly), "Observed", font=legend, fill=MEM_INK)
    lx += int(28 + draw.textlength("Observed", font=legend) + gap)
    panel.paste(hatch_cell(22, 13, STRIPE), (lx, ly + 2))
    text = "Occluded \u2013 tracker updates visibility and pose age"
    draw.text((lx + 28, ly), text, font=legend, fill=MEM_INK)
    lx += int(28 + draw.textlength(text, font=legend) + gap)
    draw.rounded_rectangle((lx, ly, lx + 24, ly + 17), radius=8, fill=BADGE)
    draw.text((lx + 8, ly), "k", font=badge_font, fill=BADGE_INK)
    text = "steps since last observed"
    draw.text((lx + 30, ly), text, font=legend, fill=MEM_INK)
    lx += int(30 + draw.textlength(text, font=legend) + gap)
    draw.rounded_rectangle((lx, ly + 1, lx + 22, ly + 15), radius=3, outline=MEM_INK, width=2)
    draw.text((lx + 28, ly), "target", font=legend, fill=MEM_INK)
    lx += int(28 + draw.textlength("target", font=legend) + gap)
    draw.rectangle((lx, ly - 1, lx + 12, ly + 17), outline=CURSOR, width=2)
    draw.text((lx + 18, ly), "current scene state", font=legend, fill=MEM_INK)

    for i in range(n):
        y0 = g["row_top"] + i * g["pitch"]
        cy = y0 + g["cell_h"] // 2
        seen_now = step is None or vis[step][i] > 0
        if i == 0:   # target: outline the whole row
            draw.rounded_rectangle((g["thumb_x"] - 6, y0 - 2, MEM_W - 6, y0 + g["cell_h"] + 2),
                                   radius=5, outline=MEM_INK, width=2)
        thumb = thumbs[i]
        if thumb is not None:
            tile = thumb.copy()
            if THUMB_LIFT:
                r, gch, b, a = tile.split()
                lift = lambda v: int(v + (255 - v) * THUMB_LIFT)
                tile = Image.merge("RGBA", (r.point(lift), gch.point(lift), b.point(lift), a))
            if not seen_now:
                tile.putalpha(tile.getchannel("A").point(lambda a: a * 0.4))
            panel.paste(tile, (g["thumb_x"] + (g["thumb_w"] - tile.width) // 2,
                               cy - tile.height // 2), tile)
        if step is None:
            continue
        cw = g["step_w"] - 3
        for t in range(step + 1):
            x0 = g["raster_x"] + t * g["step_w"]
            if vis[t][i] > 0:
                draw.rounded_rectangle((x0, y0, x0 + cw, y0 + g["cell_h"]), radius=2, fill=OBSERVED)
            else:
                panel.paste(hatch_cell(cw + 1, g["cell_h"] + 1, STRIPE), (x0, y0))
        if not seen_now:
            label = str(int(stale[step][i]))
            w = draw.textlength(label, font=badge_font)
            bx = int(g["badge_x"] + (g["badge_w"] - w - 14) // 2)
            draw.rounded_rectangle((bx, cy - 9, bx + w + 14, cy + 9), radius=9, fill=BADGE)
            draw.text((bx + 7, cy - 8), label, font=badge_font, fill=BADGE_INK)
    if step is not None:
        # current scene state: a cursor over the newest column
        x0 = g["raster_x"] + step * g["step_w"] - 3
        draw.rectangle((x0, g["row_top"] - 5, x0 + g["step_w"] + 1, g["row_top"] + n * g["pitch"] - 1),
                       outline=CURSOR, width=3)
    return panel


# ----------------------------------------------------------- plan overlay
def dashed(draw, p0, p1, fill, width, dash, gap):
    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d = 0.0
    while d < length:
        e = min(d + dash, length)
        draw.line((x0 + ux * d, y0 + uy * d, x0 + ux * e, y0 + uy * e), fill=fill, width=width)
        d = e + gap


def draw_plan(step: int | None, data: dict, to_panel, supersample: int = 2):
    """RGBA overlay: nominal teacher plan, look-ahead window, executed path, deviation."""
    from PIL import Image, ImageDraw

    S = supersample
    layer = Image.new("RGBA", (MAIN_W * S, MAIN_H * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    plan = data["plan"]
    pts = [tuple(c * S for c in to_panel(*p)) for p in plan]
    draw.line(pts, fill=PINK + (95,), width=3 * S, joint="curve")
    for x, y in pts:
        r = 3 * S
        draw.ellipse((x - r, y - r, x + r, y + r), fill=PINK + (130,))
    if step is None:
        return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)

    n = len(plan)
    window = sorted({min(step + k, n - 1) for k in range(PLAN_LOOKAHEAD)})
    wpts = [pts[j] for j in window]
    lens = 24 * S
    if len(wpts) > 1:
        draw.line(wpts, fill=PINK + (85,), width=lens, joint="curve")
    for x, y in (wpts[0], wpts[-1]):
        draw.ellipse((x - lens / 2, y - lens / 2, x + lens / 2, y + lens / 2), fill=PINK + (85,))
    if len(wpts) > 1:
        draw.line(wpts, fill=PINK + (255,), width=6 * S, joint="curve")
    for x, y in wpts:
        r = 5 * S
        draw.ellipse((x - r, y - r, x + r, y + r), fill=PINK + (255,),
                     outline=(255, 255, 255, 255), width=S)

    executed = [tuple(c * S for c in to_panel(*p)) for p in data["eef"][:step + 1]]
    if len(executed) > 1:
        draw.line(executed, fill=CYAN + (255,), width=5 * S, joint="curve")
    ex, ey = executed[-1]
    dashed(draw, (ex, ey), pts[min(step, n - 1)], (255, 255, 255, 235), 2 * S, 6 * S, 4 * S)
    r = 7 * S
    draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=CYAN + (255,),
                 outline=(255, 255, 255, 255), width=2 * S)
    return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)


def render_plan_legend(path: Path) -> None:
    """Static legend beside the D455 panel naming every overlay mark."""
    from PIL import Image, ImageDraw, ImageFont

    S = 2
    panel = Image.new("RGBA", (LEGEND_W * S, LEGEND_H * S), (32, 33, 36, 255))
    draw = ImageDraw.Draw(panel)
    title = ImageFont.truetype(LATO_SEMI, 15 * S)
    name = ImageFont.truetype(LATO_SEMI, 14 * S)
    sub = ImageFont.truetype(LATO, 12 * S)
    draw.text((14 * S, 12 * S), "TOP-DOWN OVERLAY", font=title, fill=TEXT)
    icon_x, text_x = 14 * S, 64 * S
    text_w = LEGEND_W * S - text_x - 12 * S
    entries = [
        ("plan", "Nominal plan", "teacher's EEF path from the digital twin, given in full at the start"),
        ("window", "Look-ahead window", f"next {PLAN_LOOKAHEAD} plan waypoints the student conditions on; slides every step"),
        ("path", "Executed path", "student's own EEF path on the real robot"),
        ("deviation", "Deviation from plan", "real EEF to the nominal waypoint for this step"),
    ]
    y = 44 * S
    for kind, label, detail in entries:
        cy = y + 10 * S
        if kind == "plan":
            draw.line((icon_x, cy, icon_x + 40 * S, cy), fill=PINK + (120,), width=3 * S)
            for k in range(3):
                x = icon_x + k * 20 * S
                draw.ellipse((x - 2 * S, cy - 2 * S, x + 2 * S, cy + 2 * S), fill=PINK + (160,))
        elif kind == "window":
            draw.line((icon_x + 6 * S, cy, icon_x + 34 * S, cy), fill=PINK + (85,), width=18 * S)
            for x in (icon_x + 6 * S, icon_x + 34 * S):
                draw.ellipse((x - 9 * S, cy - 9 * S, x + 9 * S, cy + 9 * S), fill=PINK + (85,))
            draw.line((icon_x + 6 * S, cy, icon_x + 34 * S, cy), fill=PINK + (255,), width=6 * S)
            for x in (icon_x + 6 * S, icon_x + 20 * S, icon_x + 34 * S):
                draw.ellipse((x - 4 * S, cy - 4 * S, x + 4 * S, cy + 4 * S), fill=PINK + (255,),
                             outline=(255, 255, 255, 255), width=S)
        elif kind == "path":
            draw.line((icon_x, cy + 5 * S, icon_x + 18 * S, cy - 4 * S, icon_x + 34 * S, cy),
                      fill=CYAN + (255,), width=5 * S)
            draw.ellipse((icon_x + 28 * S, cy - 6 * S, icon_x + 40 * S, cy + 6 * S),
                         fill=CYAN + (255,), outline=(255, 255, 255, 255), width=2 * S)
        else:
            dashed(draw, (icon_x + 4 * S, cy), (icon_x + 40 * S, cy), (255, 255, 255, 235),
                   2 * S, 6 * S, 4 * S)
        draw.text((text_x, y), label, font=name, fill=TEXT)
        for k, line in enumerate(wrap(draw, detail, sub, text_w)):
            draw.text((text_x, y + (20 + 16 * k) * S), line, font=sub, fill=SUBTEXT)
        y += (20 + 16 * len(wrap(draw, detail, sub, text_w)) + 18) * S
    panel.resize((LEGEND_W, LEGEND_H), Image.Resampling.LANCZOS).save(path)


def state_at(ts: float, step_times: list, show_from: float, stop: float) -> int | None | str:
    if ts < show_from or ts >= stop:
        return "hidden"
    if ts < step_times[0]:
        return None
    return int(np.searchsorted(step_times, ts, side="right") - 1)


def render_state_video(path: Path, total: float, fps: int, size: tuple, state_fn, draw_fn,
                       alpha: bool) -> None:
    w, h = size
    pix_in = "rgba" if alpha else "rgb24"
    codec = ["-c:v", "png", "-f", "mov"] if alpha else [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p"]
    command = [base.find_encoder(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-an", *codec, str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    cache = {}
    try:
        for frame_number in range(round(total * fps) + 1):
            state = state_fn(frame_number / fps)
            if state not in cache:
                cache[state] = draw_fn(state).tobytes()
            process.stdin.write(cache[state])
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def render_sim_trace(path: Path, trace: np.ndarray) -> None:
    """RGBA tile (500x350): the rollout EEF trace mapped through the composer's sim transform."""
    from PIL import Image, ImageDraw

    S = 2
    layer = Image.new("RGBA", (TILE_W * S, TILE_H * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = SIM_CONTENT_SIZE / 3840.0
    x_pad = (TILE_W - SIM_CONTENT_SIZE) // 2

    def tile_px(x_sim, y_sim):
        u, v = (x_sim - 0.276) / 0.448, (y_sim + 0.224) / 0.448
        col, row = SIM_FRAME_LO + v * SIM_FRAME_SPAN, SIM_FRAME_LO + u * SIM_FRAME_SPAN
        ic, ir = col - SIM_INNER_X, row - SIM_INNER_Y
        fx, fy = SIM_INNER_X + ir, SIM_INNER_Y + (SIM_INNER_SIZE - 1 - ic)   # transpose=2
        return ((x_pad + fx * scale) * S, fy * scale * S)

    pts = [tile_px(*p) for p in trace]
    draw.line(pts, fill=PINK + (255,), width=3 * S, joint="curve")
    x, y = pts[-1]
    draw.ellipse((x - 4 * S, y - 4 * S, x + 4 * S, y + 4 * S), fill=PINK + (255,))
    layer.resize((TILE_W, TILE_H), Image.Resampling.LANCZOS).save(path)


# ------------------------------------------------------------ filter graph
def build_filter(total, raw_d455_end, overlay_start, obs_hold, dashboard_hold, solve_start,
                 gn_start, end_slide_start, webcam_advance, rollout_duration, solver_panel,
                 sim_end, legend_start, preview, d455_tone=(1.0, 1.0), d415_tone=(1.0, 1.0)) -> str:
    # Same graph as compose_closed_loop.build_filter, with the D455 resized and
    # four extra inputs: [9] plan overlay, [10] memory timeline, [11] sim trace,
    # [12] overlay legend.
    tail = max(0.0, total - raw_d455_end + 0.25)
    dashboard_tail = max(0.0, total - dashboard_hold + 0.25)
    obs_tail = max(0.0, total - obs_hold + 0.25)
    obs_inner = TILE_H
    obs_xpad = (TILE_W - obs_inner) // 2
    def tone(setting):
        # A measured exposure.video_transfer chain, or the legacy fixed gamma (gamma, saturation).
        if isinstance(setting, str):
            return setting
        gamma, saturation = setting
        if gamma == 1.0 and saturation == 1.0:
            return ""
        g = 1.0 / gamma
        return (f"format=gbrp,lutrgb=r=gammaval({g:.4f}):g=gammaval({g:.4f}):b=gammaval({g:.4f}),"
                f"format=yuv420p,eq=saturation={saturation},")
    d455_tone = tone(d455_tone)
    d415_tone = tone(d415_tone)
    chains = [
        f"color=c=0x111820:s={CANVAS_W}x{CANVAS_H}:r=30:d={total:.6f}[bg]",
        (f"[1:v]crop=iw*0.75:ih:iw*0.25:0,hflip,vflip,{d455_tone}"
         f"tpad=stop_mode=clone:stop_duration={tail:.6f},"
         f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         f"scale={MAIN_W}:{MAIN_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         "setsar=1[d455raw]"),
        f"[9:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,format=rgba[planlayer]",
        ("[d455raw][planlayer]overlay=0:0:format=auto,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xE91E63:t=6[d455base]"),
        "[0:v]split=2[atop][aobscontent]",
        (f"[atop]trim=end={dashboard_hold:.6f},"
         f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={dashboard_tail:.6f}"
         "[headerheld]"),
        "[headerheld]split=2[asolve][aprogress]",
        (f"[asolve]crop={SOURCE_SOLVE_W}:{STANDARD_BAND_H}:"
         f"{SOURCE_SOLVE_X}:0,scale={SOLVE_OUT_W}:{DASH_H}:flags=lanczos,"
         "setsar=1[solveoriginal]"),
        (f"[aprogress]crop=1280:5:0:147,scale={DASH_W}:8:flags=lanczos,"
         "setsar=1[progress]"),
        (f"[aobscontent]crop={OBS_W}:{OBS_H - OBS_LABEL_H}:"
         f"{OBS_X}:{OBS_Y + OBS_LABEL_H},hflip,vflip,{d455_tone}"
         f"trim=end={obs_hold:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={obs_tail:.6f},"
         f"scale={obs_inner}:{obs_inner}:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:{obs_xpad}:0:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x34A853:t=4[obsbase]"),
        (f"[4:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,format=rgba,"
         "split=8[atitle][apactive][apdone][aspeed]"
         "[ad455label][aobslabel][agnlabel][asimlabel]"),
        f"[atitle]crop={TITLE_OUT_W}:{DASH_H}:0:0[titlepanel]",
        f"[apactive]crop={PERCEPTION_OUT_W}:{DASH_H}:0:260[perceptionactive]",
        f"[apdone]crop={PERCEPTION_OUT_W}:{DASH_H}:{PERCEPTION_OUT_W}:260[perceptiondone]",
        f"[aspeed]crop={SPEED_BADGE_W}:{SPEED_BADGE_H}:0:520[speedbadge]",
        f"[ad455label]crop={LABEL_D455_W}:{SPEED_BADGE_H}:110:520[label455]",
        f"[aobslabel]crop={LABEL_OBS_W}:{TILE_LABEL_H}:550:520[labelobs]",
        f"[agnlabel]crop={LABEL_GN_W}:{TILE_LABEL_H}:296:520[labelgn]",
        f"[asimlabel]crop={LABEL_SIM_W}:{TILE_LABEL_H}:452:520[labelsim]",
    ]
    if solver_panel:
        chains.append(f"[5:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
                      f"scale={SOLVE_OUT_W}:{DASH_H}:flags=lanczos,setsar=1[solverdynamic]")
    else:
        chains.append("[5:v]nullsink")
    chains.append(f"[6:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
                  f"scale={EXECUTE_OUT_W + TOTAL_OUT_W}:{DASH_H}:flags=lanczos,"
                  "setsar=1[executiondynamic]")
    chains += [
        f"color=c=0x202124:s={DASH_W}x{DASH_H}:r=30:d={total:.6f}[headerbg]",
        "[headerbg][titlepanel]overlay=0:0[h0]",
        f"[h0][perceptionactive]overlay={PERCEPTION_OUT_X}:0[h1]",
        (f"[h1][perceptiondone]overlay={PERCEPTION_OUT_X}:0:"
         f"enable='gte(t,{overlay_start:.6f})'[h2]"),
        f"[h2][solveoriginal]overlay={SOLVE_OUT_X}:0[h3]",
    ]
    chains.append(f"[h3][solverdynamic]overlay={SOLVE_OUT_X}:0:eof_action=repeat[h4]"
                  if solver_panel else "[h3]null[h4]")
    chains += [
        f"[h4][executiondynamic]overlay={EXECUTE_OUT_X}:0[h5]",
        f"[h5][progress]overlay=0:{DASH_H - 8}[dashboard]",
        "[speedbadge]split=2[badge455][badge415]",
        "[d455base][badge455]overlay=12:H-h-12[d455speed]",
        "[d455speed][label455]overlay=W-w-12:H-h-12[d455]",
        "[obsbase][labelobs]overlay=W-w-12:H-h-12[obs]",
        (f"[10:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xA142F4:t=4[memory]"),
        "[12:v]format=rgba,drawbox=x=0:y=0:w=iw:h=ih:color=0xE91E63@0.55:t=2[planlegend]",
        (f"[2:v]trim=start={webcam_advance:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={tail + webcam_advance + 2.0:.6f},"
         f"trim=duration={total:.6f},crop=iw*0.75:ih:iw*0.05:0,{d415_tone}"
         f"scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x4285F4:t=4[webcam]"),
        "[webcam][badge415]overlay=12:H-h-12[d415]",
        (f"[7:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xFB8C00:t=4[gnselected]"),
        "[gnselected][labelgn]overlay=W-w-12:H-h-12[gn]",
        f"[8:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1[gnslide]",
        (f"[3:v]trim=duration={rollout_duration:.6f},setpts=PTS-STARTPTS,"
         "split=3[simbase][siminner][simlabels]"),
        (f"[siminner]crop={SIM_INNER_SIZE}:{SIM_INNER_SIZE}:{SIM_INNER_X}:{SIM_INNER_Y},"
         f"drawbox=x=0:y=0:w={SIM_INNER_SIZE}:h={SIM_WORKSPACE_LABEL_H}:"
         "color=black:t=fill,transpose=2[simrotated]"),
        f"[simbase][simrotated]overlay={SIM_INNER_X}:{SIM_INNER_Y}[simscene]",
        (f"[simlabels]crop={SIM_INNER_SIZE}:{SIM_WORKSPACE_LABEL_H}:"
         f"{SIM_INNER_X}:{SIM_INNER_Y}[simlabelstrip]"),
        (f"[simscene][simlabelstrip]overlay={SIM_INNER_X}:{SIM_INNER_Y},"
         f"scale={SIM_CONTENT_SIZE}:{SIM_CONTENT_SIZE}:flags=lanczos,setsar=1,"
         f"setpts=PTS+{solve_start:.6f}/TB[simvideo]"),
        (f"color=c=black:s={SIM_CONTENT_SIZE}x{SIM_CONTENT_SIZE}:r=30:d={total:.6f},"
         f"drawtext=fontfile={LATO_SEMI}:"
         "text='Loading Isaac Gym...':fontcolor=0xDADCE0:fontsize=20:"
         "x=(w-text_w)/2:y=(h-text_h)/2[simblank]"),
        ("[simblank][simvideo]overlay=0:0:eof_action=repeat:shortest=0,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black[simpadded]"),
        "[11:v]format=rgba[simtrace]",
        f"[simpadded][simtrace]overlay=0:0:enable='gte(t,{sim_end:.6f})'[simtraced]",
        "[simtraced]drawbox=x=0:y=0:w=iw:h=ih:color=0xFBBC04:t=4[simbox]",
        "[simbox][labelsim]overlay=W-w-12:H-h-12[sim]",
        f"[bg][dashboard]overlay={MARGIN}:{DASH_Y}[c0]",
        f"[c0][d455]overlay={MAIN_X}:{MAIN_Y}[c1]",
        f"[c1][planlegend]overlay={LEGEND_X}:{LEGEND_Y}:enable='gte(t,{legend_start:.6f})'[c1l]",
        f"[c1l][memory]overlay={MEM_X}:{MEM_Y}:enable='gte(t,{overlay_start:.6f})'[c1m]",
        f"[c1m][obs]overlay={TILE_X1}:{TILE_Y1}:enable='gte(t,{overlay_start:.6f})'[c2]",
        f"[c2][gn]overlay={TILE_X2}:{TILE_Y1}:enable='gte(t,{gn_start:.6f})'[c3]",
        f"[c3][d415]overlay={TILE_X1}:{TILE_Y2}[c4]",
        f"[c4][sim]overlay={TILE_X2}:{TILE_Y2}[c5]",
        f"[c5][gnslide]overlay=0:0:enable='gte(t,{end_slide_start:.6f})'[c6]",
    ]
    chains.append("[c6]scale=960:540:flags=lanczos,format=yuv420p[out]" if preview
                  else "[c6]format=yuv420p[out]")
    return ";".join(chains)


def render_asset_atlas(path, title, perception_active, perception_done, speed, d455_label,
                       obs_label, gn_label, sim_label) -> None:
    from PIL import Image

    atlas = Image.new("RGBA", (base.ATLAS_W, base.ATLAS_H), (0, 0, 0, 0))
    for source, xy in ((title, (0, 0)), (perception_active, (0, DASH_H)),
                       (perception_done, (PERCEPTION_OUT_W, DASH_H)), (speed, (0, DASH_H * 2)),
                       (d455_label, (110, DASH_H * 2)), (gn_label, (296, DASH_H * 2)),
                       (sim_label, (452, DASH_H * 2)), (obs_label, (550, DASH_H * 2))):
        atlas.alpha_composite(Image.open(source).convert("RGBA"), dest=xy)
    atlas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="student closed-loop trial directory")
    parser.add_argument("output", type=Path, help="new MP4 to create")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--limit", type=float, help="render only the first N seconds")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="dashboard",
                        help="object-memory panel palette")
    parser.add_argument("--d455-gamma", type=float, default=1.8,
                        help="brightening gamma for the D455 camera and observation panels (1 = untouched)")
    parser.add_argument("--exposure", choices=("measured", "fixed"), default="measured",
                        help="measured: exposure.video_transfer (snapshot curve, gain from the clip); "
                             "fixed: the legacy constant gamma")
    parser.add_argument("--d455-saturation", type=float, default=1.08)
    parser.add_argument("--d415-gamma", type=float, default=1.3,
                        help="brightening gamma for the D415 scene camera panel (1 = untouched)")
    parser.add_argument("--d415-saturation", type=float, default=1.04)
    parser.add_argument("--preview", action="store_true",
                        help="fast check render: 960x540, 15 fps, crf 32, veryfast")
    args = parser.parse_args()

    apply_palette(args.palette)
    trial = args.trial.resolve()
    annotated_d455 = trial / "d455_topdown.mp4"
    raw_d455 = trial / "d455_topdown_raw.mp4"
    raw_webcam, webcam_recorder = third_person(trial)
    sim_candidates = sorted((trial / "sim").glob("*_closed-loop.mp4"))
    if not sim_candidates:
        raise SystemExit(f"No simulator video found under {trial / 'sim'}")
    sim = sim_candidates[0]
    for source in (annotated_d455, raw_d455, raw_webcam, sim):
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")

    phases = base.load_json(trial / "metadata" / "phases.json")
    sim_timing_candidates = sorted((trial / "sim").glob("*_sim_timing.json"))
    sim_timing = base.load_json(sim_timing_candidates[0]) if sim_timing_candidates else {}
    total = min(base.duration(annotated_d455), args.limit or float("inf"))
    sim_duration = base.duration(sim)
    raw_duration = base.duration(raw_d455)
    webcam_duration = base.duration(raw_webcam)
    raw_end = min(raw_duration, total)
    recorder = base.load_json(trial / "metadata" / "d455_topdown_rec.json")
    video_zero_offset = max(0.0, float(recorder.get("duration_s", raw_duration)) - raw_duration)
    webcam_zero_offset = max(
        0.0, float(webcam_recorder.get("duration_s", webcam_duration)) - webcam_duration)
    webcam_advance = max(0.0, video_zero_offset - webcam_zero_offset)

    def video_event(phase: str) -> float | None:
        value = base.event_offset(phases, phase)
        return max(0.0, value - video_zero_offset) if value is not None else None

    overlay_start = video_event("twin") or 0.0
    overlay_stop = video_event("home") or raw_end
    solve_start = video_event("solve") or 0.0
    solve_done = video_event("solve-done") or solve_start
    gn_start = video_event("grasp") or video_event("no-grasp") or raw_end
    obs_hold = max(overlay_start, overlay_stop - 0.5)
    dashboard_hold = min(raw_end, gn_start + 1.2)
    gn_candidates = sorted((trial / "perception").glob("*_gn16.png"))
    if not gn_candidates:
        raise SystemExit(f"No grasp-network panel found under {trial / 'perception'}")
    gn_panel = gn_candidates[-1]
    selected_gn_panel = gn_panel
    if video_event("grasp") is not None:
        prediction_candidates = sorted((trial / "perception").glob("*_gn16_predictions.npz"))
        with np.load(prediction_candidates[-1], allow_pickle=False) as predictions:
            prediction_metadata = json.loads(str(predictions["metadata"].item()))
        selected_bin = int(prediction_metadata["grasp"]["rotation_idx"])
        selected_gn_panel = trial / "perception" / f"{gn_panel.stem}_tiles/bin_{selected_bin:02d}.png"
        if not selected_gn_panel.is_file():
            raise SystemExit(f"Missing selected grasp bin: {selected_gn_panel}")
    done_event = video_event("done")
    end_slide_start = done_event if done_event is not None else max(gn_start, total - 4.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing output: {args.output}")

    breakdown = None
    if not {"twin-settle", "solve-export"} <= {e.get("phase") for e in phases.get("events", [])}:
        breakdown = base.legacy_solver_breakdown(phases, sim_timing)
    rollout_duration = breakdown["ppo_rollout_simulated_seconds"] if breakdown else sim_duration
    # Newer twin videos hold the final frame for 2 s after the motion; the trace
    # belongs to the end of the motion, i.e. num_steps control steps in.
    control_hz = float(sim_timing.get("control_hz") or 15.06)
    motion_s = float(sim_timing.get("num_steps") or 0) / control_hz or sim_duration
    sim_end = solve_start + min(sim_duration, rollout_duration, motion_s)

    data = load_student_trial(trial, phases, video_zero_offset)
    to_panel = camera_projection(trial)
    pitch = memory_geometry(len(data["objects"]), len(data["vis"]))["cell_h"]
    thumbs, colors = object_thumbnails(trial, data["objects"], (46, pitch + 2))

    with tempfile.TemporaryDirectory(prefix="closed_loop_memory_") as temporary:
        tmp = Path(temporary)
        scene_match = re.search(r"scene\s*0*(\d+)", trial.parent.name, re.IGNORECASE)
        scene_name = f"Scene {int(scene_match.group(1)):02d}" if scene_match else "Scene"
        first_perceive = base.event_offset(phases, "perceive") or 0.0
        twin_event = base.event_offset(phases, "twin") or first_perceive
        perception_seconds = max(0.0, twin_event - first_perceive)
        base.render_title_panel(tmp / "title.png", scene_name)
        base.render_perception_panel(tmp / "perception_active.png", complete=False,
                                     seconds=perception_seconds)
        base.render_perception_panel(tmp / "perception_done.png", complete=True,
                                     seconds=perception_seconds)
        base.render_speed_badge(tmp / "speed.png")
        base.render_text_badge(tmp / "d455_label.png", "Orthographic Camera", LABEL_D455_W, SPEED_BADGE_H)
        base.render_text_badge(tmp / "obs_label.png", "Occluded\nObservations", LABEL_OBS_W, TILE_LABEL_H)
        base.render_text_badge(tmp / "gn_label.png", "Grasp\nEvaluation", LABEL_GN_W, TILE_LABEL_H)
        base.render_text_badge(tmp / "sim_label.png", "Digital\nTwin", LABEL_SIM_W, TILE_LABEL_H)
        base.render_fitted_image(selected_gn_panel, tmp / "selected_gn.png", TILE_W, TILE_H)
        base.render_fitted_image(gn_panel, tmp / "gn_slide.png", CANVAS_W, CANVAS_H)
        if breakdown:
            base.render_solver_panel_video(tmp / "solver_timing.mp4", breakdown, total,
                                           overlay_start, solve_start)
        execution_timings = {
            "policy_start": video_event("student-load"), "policy_end": video_event("student"),
            "push_start": video_event("student"), "push_end": video_event("home"),
            "evaluation_start": video_event("re-sense"),
            "evaluation_end": video_event("grasp") or video_event("no-grasp"),
            "grasp_start": video_event("grasp"),
            "grasp_end": video_event("done") if video_event("grasp") else None,
            "total_time_offset": 0.45,
        }
        base.render_execution_and_time_video(tmp / "execution_timing.mp4", total, execution_timings)
        render_asset_atlas(tmp / "assets.png", tmp / "title.png", tmp / "perception_active.png",
                           tmp / "perception_done.png", tmp / "speed.png", tmp / "d455_label.png",
                           tmp / "obs_label.png", tmp / "gn_label.png", tmp / "sim_label.png")

        step_times = data["step_times"]

        def plan_frame(state):
            from PIL import Image
            if state == "hidden":
                return Image.new("RGBA", (MAIN_W, MAIN_H), (0, 0, 0, 0))
            return draw_plan(state, data, to_panel)

        # The plan exists once the twin has solved; the window follows the student's steps.
        render_state_video(tmp / "plan.mov", total, 10, (MAIN_W, MAIN_H),
                           lambda ts: state_at(ts, step_times, solve_done, overlay_stop),
                           plan_frame, alpha=True)
        # Memory holds its last row after the student stops.
        render_state_video(tmp / "memory.mp4", total, 10, (MEM_W, MEM_H),
                           lambda ts: (None if ts < step_times[0] else
                                       int(np.searchsorted(step_times, ts, side="right") - 1)),
                           lambda s: draw_memory(s, data, thumbs, colors), alpha=False)
        render_sim_trace(tmp / "sim_trace.png", data["trace"])
        render_plan_legend(tmp / "plan_legend.png")

        loop = lambda path: ["-loop", "1", "-framerate", "30", "-i", str(path)]
        command = [base.find_encoder(), "-hide_banner", "-loglevel", "warning", "-nostdin",
                   "-stats", "-y",
                   "-i", str(annotated_d455), "-i", str(raw_d455), "-i", str(raw_webcam),
                   "-i", str(sim), *loop(tmp / "assets.png")]
        command += (["-i", str(tmp / "solver_timing.mp4")] if breakdown else loop(tmp / "assets.png"))
        command += ["-i", str(tmp / "execution_timing.mp4"), *loop(tmp / "selected_gn.png"),
                    *loop(tmp / "gn_slide.png"), "-i", str(tmp / "plan.mov"),
                    "-i", str(tmp / "memory.mp4"), *loop(tmp / "sim_trace.png"),
                    *loop(tmp / "plan_legend.png")]
        d455_setting = (args.d455_gamma, args.d455_saturation)
        d415_setting = (args.d415_gamma, args.d415_saturation)
        if args.exposure == "measured":
            d455_setting, d455_meta = exposure.video_transfer(
                raw_d455, "d455", crop="crop=iw*0.75:ih:iw*0.25:0,hflip,vflip",
                saturation=args.d455_saturation)
            d415_setting, d415_meta = exposure.video_transfer(
                raw_webcam, "d415", crop="crop=iw*0.75:ih:iw*0.05:0",
                saturation=args.d415_saturation)
            exposure.write_metadata(args.output.with_suffix(".exposure.json"),
                                    {"d455": d455_meta, "webcam": d415_meta})
            print(f"exposure: D455 gain {d455_meta['brightness_gain']:.3f} (gamma {d455_meta['gamma']}), "
                  f"webcam gain {d415_meta['brightness_gain']:.3f} (gamma {d415_meta['gamma']})")
        command += [
            "-filter_complex", build_filter(total, raw_end, overlay_start, obs_hold,
                                            dashboard_hold, solve_start, gn_start,
                                            end_slide_start, webcam_advance, rollout_duration,
                                            bool(breakdown), sim_end, solve_done, args.preview,
                                            d455_setting, d415_setting),
            "-map", "[out]", "-an", "-r", "15" if args.preview else "30", "-t", f"{total:.6f}",
            "-c:v", "libx264", "-preset", "veryfast" if args.preview else args.preset,
            "-crf", "32" if args.preview else str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        print("Creating", args.output)
        print(f"Duration {total:.2f} s; {len(step_times)} student steps from {step_times[0]:.2f} s; "
              f"plan {len(data['plan'])} waypoints; twin trace from {sim_end:.2f} s")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
