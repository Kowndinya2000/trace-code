#!/usr/bin/env python3
"""Compose one spiral or PMBS closed-loop trial into the master review layout.

PMBS trials (controller "pmbs") use the same layout: PMBS re-senses the full scene before
every decision, so its tile is the re-sensed observation.  Because its pushes are
discrete, the camera overlay shows only the current EEF pose during an active push;
completed push paths are never accumulated.


Same canvas, palette, borders, badges and end slide as the teacher master
(compose_teacher_closed_loop.py). What differs follows the spiral pipeline: the
dashboard is PERCEIVE / PREPARE / SPIRAL ON THE ROBOT, the top-left tile is the
observation each spiral step was decided on (partial, or re-sensed after a
retract), the D455 panel carries the executed spiral path in pink, and an
out-of-workspace stop is shown as an OOW failure with the frame that triggered it
(the block over the black-mat edge highlighted) in place of the grasp panel.
The D455 and webcam panels get the same gamma brightening as the student masters.
Source recordings are read only; the output is always a new file.
"""

from __future__ import annotations

import argparse
from camera_sources import third_person
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import exposure
from compose_closed_loop import (
    CANVAS_H, CANVAS_W, DASH_H, DASH_W, LABEL_D455_W, LABEL_GN_W, LABEL_OBS_W, MAIN_H, MAIN_W,
    SPEED_BADGE_H, TILE_H, TILE_LABEL_H, TILE_W, duration, event_offset, find_encoder, load_json,
    render_fitted_image, render_speed_badge, render_text_badge,
)
import compose_teacher_closed_loop as teacher
from compose_teacher_closed_loop import (
    GRAY, GREEN, BLUE, LATO, LATO_SEMI, PANEL, WHITE, camera_projection, draw_trace,
    render_trace_legend, state, trace_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
RED = "#EA4335"
OOW_STOPS = ("out_of_workspace", "off_mat", "object_missing")
TITLE_W, PERCEIVE_W, PREPARE_W, SOLVE_W = 320, 340, 360, 520
PROFILES = {
    "circular_spiral": dict(title="Spiral Baseline", load="spiral-load", observe="spiral-observe",
                            models="Perception + grasp models", loop="Grasp check + spiral push",
                            obs_label="Robot's\nObservation"),
    "pmbs": dict(title="PMBS Baseline", load="pmbs-load", observe="pmbs-observe",
                 models="PMBS models + simulator", loop="Grasp check + MCTS push",
                 obs_label="Re-sensed\nObservation"),
}
PROFILE = PROFILES["circular_spiral"]


def status_text(detail: str) -> str:
    text = detail
    text = re.sub(r"Evaluating grasp network on partial observation \d+",
                  "grasp check on the partial observation", text)
    text = re.sub(r"partial observation \d+", "observing the scene", text)
    text = re.sub(r"target hidden at step \d+", "target hidden - retracting to re-sense", text)
    text = re.sub(r"in-place GN graspable at step \d+", "graspable in place - confirming on a clean view", text)
    text = re.sub(r"returning to saved push pose after step \d+", "resuming spiral", text)
    text = text.replace("clean observation at PMBS home", "re-sensing the full scene")
    text = re.sub(r"evaluating Original Grasp Network on the clean observation",
                  "grasp check on the re-sensed scene", text, flags=re.IGNORECASE)
    text = text.replace("loading the observed scene into the parallel simulator", "loading the scene into PMBS's simulator")
    text = text.replace("retracting to PMBS home for the next observation", "retracting to re-sense")
    text = text.replace("full observation at PMBS home", "re-sensing the full scene")
    text = re.sub(r"^evaluating Original Grasp Network$", "grasp check", text, flags=re.IGNORECASE)
    text = re.sub(r"original grasp network", "grasp network", text, flags=re.IGNORECASE)
    return text


def detail_at(phases: dict, ts: float) -> str:
    current = ""
    for event in phases["events"]:
        if event["t"] - phases["t0"] > ts:
            break
        detail = str(event.get("detail", ""))
        if not detail.startswith("internal:"):
            current = detail
    return current


def step_timings(phases: dict) -> dict:
    first = lambda phase: event_offset(phases, phase)
    grasp, no_grasp, done = first("grasp"), first("no-grasp"), first("done")
    loop_start = first(PROFILE["observe"]) or first("spiral")
    end_loop = grasp if grasp is not None else no_grasp
    return {
        "segmentation": (first("perceive"), first(PROFILE["load"])),
        "models": (first(PROFILE["load"]), loop_start),
        "loop": (loop_start, end_loop if end_loop is not None else done),
        "grasp": (grasp, done if grasp is not None else None),
        "no_grasp": no_grasp,
        "start": first("perceive"),
        "done": done,
    }


def draw_dashboard(ts: float, steps: dict, phases: dict, scene_name: str, arc_cm: float,
                   oow: str | None):
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGB", (DASH_W, DASH_H), PANEL)
    draw = ImageDraw.Draw(panel)
    head = ImageFont.truetype(LATO_SEMI, 17)
    body = ImageFont.truetype(LATO, 18)
    big_body = ImageFont.truetype(LATO, 20)
    small = ImageFont.truetype(LATO, 16)
    status = ImageFont.truetype(LATO, 15)
    color = {"pending": GRAY, "active": BLUE, "done": GREEN}
    failed = oow is not None and steps["no_grasp"] is not None and ts >= steps["no_grasp"]

    def group_color(names):
        states = [state(steps[name], ts)[0] for name in names]
        if all(s == "done" for s in states):
            return GREEN
        return BLUE if any(s != "pending" for s in states) else GRAY

    def dot(x, y, phase_state, fill=None):
        if phase_state == "pending":
            draw.ellipse((x, y + 5, x + 10, y + 15), outline=GRAY, width=2)
        else:
            draw.ellipse((x, y + 5, x + 10, y + 15), fill=fill or color[phase_state])

    def step_row(x0, width, y, label, name, font=body):
        phase_state, elapsed = state(steps[name], ts)
        fill = color[phase_state]
        dot(x0 + 18, y, phase_state)
        draw.text((x0 + 36, y), label, font=font, fill=fill)
        value = "-" if phase_state == "pending" else f"{elapsed:.1f} s"
        draw.text((x0 + width - draw.textlength(value, font=font) - 18, y), value, font=font, fill=fill)
        return phase_state

    draw.text((18, 22), scene_name, font=ImageFont.truetype(LATO_SEMI, 25), fill=WHITE)
    draw.text((18, 79), PROFILE["title"], font=ImageFont.truetype(LATO_SEMI, 20), fill=WHITE)
    draw.text((18, 118), "Closed-Loop", font=ImageFont.truetype(LATO_SEMI, 19), fill=WHITE)

    # PERCEIVE (clean initial scene), two-line entry as in the teacher master
    x = TITLE_W
    seg_state, seg_elapsed = state(steps["segmentation"], ts)
    seg_color = color[seg_state]
    draw.text((x + 18, 18), "PERCEIVE   real camera", font=head, fill=seg_color)
    if seg_state == "pending":
        draw.ellipse((x + 18, 71, x + 29, 82), outline=GRAY, width=2)
    else:
        draw.ellipse((x + 18, 71, x + 29, 82), fill=seg_color)
    draw.text((x + 38, 64), "Initial scene", font=big_body, fill=seg_color)
    draw.text((x + 38, 101), "segmentation", font=big_body, fill=seg_color)
    value = "-" if seg_state == "pending" else f"{seg_elapsed:.1f} s"
    draw.text((x + PERCEIVE_W - draw.textlength(value, font=small) - 18, 136), value, font=small, fill=seg_color)

    # PREPARE
    x += PERCEIVE_W
    draw.text((x + 18, 18), "PREPARE", font=head, fill=group_color(("models",)))
    step_row(x, PREPARE_W, 67, PROFILE["models"], "models")

    # SPIRAL ON THE ROBOT
    x += PREPARE_W
    heading_color = RED if failed else group_color(("loop", "grasp"))
    heading = ("PMBS ON THE ROBOT   re-sensed before every push" if PROFILE is PROFILES["pmbs"]
               else f"SPIRAL ON THE ROBOT   observed before every {arc_cm:g} cm step")
    draw.text((x + 18, 18), heading, font=head, fill=heading_color)
    if step_row(x, SOLVE_W, 67, PROFILE["loop"], "loop") == "active":
        live = status_text(detail_at(phases, ts))
        if live:
            draw.text((x + 48, 97), live, font=status, fill=GRAY)
    if failed:
        dot(x + 18, 143, "active", fill=RED)
        draw.text((x + 36, 143), "Grasp + retrieve", font=body, fill=RED)
        value = "OOW failure"
        draw.text((x + SOLVE_W - draw.textlength(value, font=body) - 18, 143), value, font=body, fill=RED)
        reason = {"off_mat": "an object crossed the black-mat edge",
                  "object_missing": "an object is no longer on the mat",
                  "out_of_workspace": "an object left the workspace"}[oow]
        draw.text((x + 48, 170), reason, font=status, fill=RED)
    else:
        step_row(x, SOLVE_W, 143, "Grasp + retrieve", "grasp")

    x += SOLVE_W
    start, done = steps["start"], steps["done"]
    total = 0.0 if start is None else max(0.0, (min(ts, done) if done is not None else ts) - start)
    draw.text((x + 22, 52), "Total Time:", font=ImageFont.truetype(LATO_SEMI, 20), fill=WHITE)
    draw.text((x + 22, 88), f"{total:.1f} s", font=ImageFont.truetype(LATO_SEMI, 32),
              fill=RED if failed else WHITE)
    for boundary in (TITLE_W, TITLE_W + PERCEIVE_W, TITLE_W + PERCEIVE_W + PREPARE_W,
                     TITLE_W + PERCEIVE_W + PREPARE_W + SOLVE_W):
        draw.line((boundary, 0, boundary, DASH_H), fill=GRAY, width=2)
    return panel


def render_dashboard_video(path: Path, total: float, zero_offset: float, steps: dict, phases: dict,
                           scene_name: str, arc_cm: float, oow: str | None) -> None:
    fps = 10
    command = [find_encoder(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{DASH_W}x{DASH_H}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_number in range(round(total * fps) + 1):
            ts = frame_number / fps + zero_offset
            process.stdin.write(draw_dashboard(ts, steps, phases, scene_name, arc_cm, oow).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def spiral_pushes(trial: Path, phases: dict, video_zero_offset: float) -> list:
    """[(t_start, t_end, [eef, midpoint, waypoint])] in video seconds, one per executed step."""
    traj = load_json(trial / "metadata" / "trajectory.json")
    t0 = phases["t0"]
    events = phases["events"]
    starts = []
    for k, event in enumerate(events):
        if event["phase"] == "spiral" and str(event.get("detail", "")).startswith("internal: spiral step"):
            end = next((e["t"] for e in events[k + 1:]), event["t"] + 0.6)
            starts.append((event["t"] - t0 - video_zero_offset,
                           min(end, event["t"] + 1.5) - t0 - video_zero_offset))
    pushes = []
    for (start, end), dense, wp in zip(starts, traj["dense"], traj["waypoints"]):
        a, b = dense["eef"][:2], wp["wp"][:2]
        pushes.append((start, max(end, start + 0.2), [a, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], b]))
    return pushes


def pmbs_pushes(trial: Path, phases: dict, video_zero_offset: float) -> list:
    """[(t_start, t_end, [start, midpoint, end])] per executed PMBS push, in video seconds."""
    traj = load_json(trial / "metadata" / "trajectory.json")
    t0 = phases["t0"]
    push_t = [e["t"] - t0 - video_zero_offset for e in phases["events"] if e["phase"] == "pmbs-push"]
    retract_t = [e["t"] - t0 - video_zero_offset for e in phases["events"] if e["phase"] == "pmbs-retract"]
    pushes = []
    for k, wp in enumerate(traj["waypoints"][:len(push_t)]):
        a, b = wp["wp1"][:2], wp["wp2"][:2]
        end = retract_t[k] if k < len(retract_t) else push_t[k] + 2.0
        pushes.append((push_t[k], max(end, push_t[k] + 0.2), [a, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], b]))
    return pushes


def pmbs_pose_state(ts: float, pushes: list):
    """Return ``(push index, progress)`` only while a PMBS push is active."""
    for index, (start, end, _) in enumerate(pushes):
        if start <= ts < end:
            progress = (ts - start) / max(1e-3, end - start)
            return index, round(min(1.0, progress), 2)
    return None


def draw_pmbs_pose(state, pushes: list, to_panel, supersample: int = 2):
    """Draw one current-pose marker, with no historical or in-progress path."""
    from PIL import Image, ImageDraw

    scale = supersample
    layer = Image.new("RGBA", (MAIN_W * scale, MAIN_H * scale), (0, 0, 0, 0))
    if state is None:
        return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)

    index, progress = state
    points = pushes[index][2]
    left = progress * (len(points) - 1)
    segment = min(int(left), len(points) - 2)
    fraction = min(1.0, left - segment)
    a, b = points[segment], points[segment + 1]
    pose = (a[0] + (b[0] - a[0]) * fraction,
            a[1] + (b[1] - a[1]) * fraction)
    x, y = (value * scale for value in to_panel(*pose))

    draw = ImageDraw.Draw(layer)
    pink = (255, 64, 160)
    halo = 15 * scale
    core = 7 * scale
    draw.ellipse((x - halo, y - halo, x + halo, y + halo), fill=pink + (80,))
    draw.ellipse((x - core, y - core, x + core, y + core),
                 fill=pink + (255,), outline=(255, 255, 255, 255), width=2 * scale)
    return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)


def render_pmbs_pose_legend(path: Path) -> None:
    """Badge describing the point-only PMBS camera overlay."""
    from PIL import Image, ImageDraw, ImageFont

    scale, height = 2, SPEED_BADGE_H
    font = ImageFont.truetype(LATO_SEMI, 13 * scale)
    label = "Current EEF pose"
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    width = int(10 * scale + 16 * scale + 7 * scale
                + probe.textlength(label, font=font) + 10 * scale)
    badge = Image.new("RGBA", (width, height * scale), (32, 33, 36, 224))
    draw = ImageDraw.Draw(badge)
    x, y = 18 * scale, height * scale // 2
    draw.ellipse((x - 6 * scale, y - 6 * scale, x + 6 * scale, y + 6 * scale),
                 fill=(255, 64, 160, 255), outline=(255, 255, 255, 255), width=scale)
    draw.text((x + 10 * scale, 6 * scale), label, font=font, fill="#F1F3F4")
    badge.resize((width // scale, height), Image.Resampling.LANCZOS).save(path)


def observation_inset_present(annotated: Path, when: float) -> bool:
    """True when the recorder burned its observation inset into the D455 video.

    The recorder skips that overlay when the run died before writing real_timing.json /
    trajectory.json; the inset area is then plain table, and cropping it yields a wrong tile.
    """
    import cv2
    import numpy as np
    from compose_closed_loop import OBS_X, OBS_Y, OBS_W, OBS_H
    with tempfile.TemporaryDirectory(prefix="obs_probe_") as probe:
        frame = Path(probe) / "frame.png"
        subprocess.run([find_encoder(), "-v", "error", "-y", "-ss", f"{max(when, 0.0):.3f}",
                        "-i", str(annotated), "-frames:v", "1", str(frame)], check=False)
        if not frame.exists():
            return True
        image = cv2.imread(str(frame))
    patch = image[OBS_Y:OBS_Y + OBS_H, OBS_X:OBS_X + OBS_W]
    if patch.size == 0:
        return True
    return float((patch.max(axis=2) < 60).mean()) > 0.35     # the inset is a black panel


def render_observation_tile(trial: Path, output: Path) -> Path | None:
    """Observation tile straight from the controller's last saved masks (scene_rgb)."""
    import numpy as np
    from PIL import Image
    saved = sorted((trial / "perception" / "observations").glob("camera_masks_step_*.npz"),
                   key=lambda q: int(re.search(r"step_(\d+)", q.name).group(1)))
    if not saved:
        return None
    with np.load(saved[-1], allow_pickle=False) as data:
        if "scene_rgb" not in data:
            return None
        Image.fromarray(data["scene_rgb"]).save(output)
    return output


def render_oow_evidence(trial: Path, timing: dict, output: Path) -> Path:
    """Last observation with the mat outline and the block over the edge highlighted in red."""
    import cv2
    import numpy as np
    import torch
    from isaacgymenvs.open_loop import mat_boundary as mb, perceive_scene as ps

    perception = trial / "perception"
    base = perception / Path(timing["observations"][-1]["base"]).name
    load = lambda b: (cv2.imread(str(b) + "_color.png"), np.load(str(b) + "_depth.npy"),
                      json.load(open(str(b) + "_K.json")))
    cam2base = np.loadtxt(os.environ.get("TRACE_CALIB", "data/calibration/camera_to_base.txt"))
    c0, d0, k0 = load(perception / "d455_topdown_dump0")
    mat = mb.detect_mat_polygon(c0, d0, k0, cam2base)
    color, depth, K = load(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.environ.get("TRACE_MASKRCNN", "data/segmentation/maskrcnn.pth")
    model = ps.load_maskrcnn(ckpt, device)
    blocks, _, _ = ps.filter_instances_by_depth(ps.segment(color, ckpt, device, model=model), depth, K, cam2base)
    xs, ys, zb = ps.deproject_to_sim(depth, K, cam2base)
    contour = mat["corners_sim"].astype(np.float32).reshape(-1, 1, 2)
    out = color.copy()
    cv2.polylines(out, [mat["corners_px"].astype(np.int32)], True, (4, 188, 251), 3, cv2.LINE_AA)
    for inst in blocks:
        band = inst["mask"].astype(bool) & (depth > 0.05) & (zb >= 0.025) & (zb <= 0.065)
        vv, uu = np.nonzero(band)
        if len(vv) < 50:
            continue
        signed = np.array([cv2.pointPolygonTest(contour, (float(xs[v, u]), float(ys[v, u])), True)
                           for v, u in zip(vv, uu)])
        beyond = signed < -0.003
        if beyond.any():
            out[vv[beyond], uu[beyond]] = (0.45 * out[vv[beyond], uu[beyond]] + (53, 67, 234)).clip(0, 255)
            contours, _ = cv2.findContours(inst["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (53, 67, 234), 3, cv2.LINE_AA)
    out = cv2.flip(out[:, out.shape[1] // 4:], -1)        # same crop and 180 degree view as the D455 panel
    cv2.imwrite(str(output), out)
    return output


def build_filter(total, raw_end, obs_start, obs_hold, gn_start, end_slide_start, webcam_advance,
                 trace_start, preview, d455_tone, webcam_tone, obs_tile_input=None) -> str:
    chains = teacher.build_filter(total, raw_end, obs_start, obs_hold, gn_start, end_slide_start,
                                  webcam_advance, trace_start, preview).split(";")
    def tone(setting, saturation=None):
        # A measured exposure.video_transfer chain, or the legacy fixed (gamma, saturation).
        if isinstance(setting, str):
            return setting
        g, s = setting
        return (f"format=gbrp,lutrgb=r=gammaval({1 / g:.4f}):g=gammaval({1 / g:.4f}):"
                f"b=gammaval({1 / g:.4f}),format=yuv420p,eq=saturation={s},")
    if obs_tile_input is not None:
        # The recorder burned no observation inset (run died before writing its metadata);
        # the tile comes from the controller's own saved masks instead of the video crop.
        from compose_closed_loop import TILE_H, TILE_W
        replaced = False
        for i, chain in enumerate(chains):
            if chain.endswith("[obsbase]"):
                chains[i] = (f"[{obs_tile_input}:v]"
                             f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
                             f"scale={TILE_H}:{TILE_H}:flags=lanczos,"
                             f"pad={TILE_W}:{TILE_H}:{(TILE_W - TILE_H) // 2}:0:color=black,setsar=1,"
                             "drawbox=x=0:y=0:w=iw:h=ih:color=0x34A853:t=4[obsbase]")
                replaced = True
        if not replaced:
            raise SystemExit("Could not substitute the observation tile: no [obsbase] chain")
    for i, chain in enumerate(chains):
        if chain.startswith("[1:v]crop=iw*0.75:ih:iw*0.25:0,hflip,vflip,"):
            chains[i] = chain.replace("hflip,vflip,", "hflip,vflip," + tone(d455_tone), 1)
        elif chain.startswith("[2:v]trim=start="):
            chains[i] = chain.replace("crop=iw*0.75:ih:iw*0.05:0,", "crop=iw*0.75:ih:iw*0.05:0," + tone(webcam_tone), 1)
    return ";".join(chains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="spiral trial directory (.../sceneN/trialK)")
    parser.add_argument("output", type=Path, help="new MP4 to create")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--exposure", choices=("measured", "fixed"), default="measured",
                        help="measured: exposure.video_transfer; fixed: legacy constant gamma")
    parser.add_argument("--d455-gamma", type=float, default=1.8)
    parser.add_argument("--webcam-gamma", type=float, default=1.3)
    args = parser.parse_args()

    trial = args.trial.resolve()
    annotated_d455, raw_d455 = trial / "d455_topdown.mp4", trial / "d455_topdown_raw.mp4"
    raw_webcam, webcam_recorder = third_person(trial)
    for source in (annotated_d455, raw_d455, raw_webcam):
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")
    phases = load_json(trial / "metadata" / "phases.json")
    timing = load_json(trial / "metadata" / "real_timing.json")
    global PROFILE
    if timing.get("controller") not in PROFILES:
        raise SystemExit(f"Not a spiral or PMBS trial: controller={timing.get('controller')}")
    PROFILE = PROFILES[timing["controller"]]
    stop = timing.get("stop_reason")
    oow = stop if stop in OOW_STOPS else None
    arc_cm = round(100 * float(timing.get("arc_step_m", 0.01)), 1)

    total = min(duration(annotated_d455), args.limit or float("inf"))
    raw_duration, webcam_duration = duration(raw_d455), duration(raw_webcam)
    raw_end = min(raw_duration, total)
    recorder = load_json(trial / "metadata" / "d455_topdown_rec.json")
    video_zero_offset = max(0.0, float(recorder.get("duration_s", raw_duration)) - raw_duration)
    webcam_zero_offset = max(0.0, float(webcam_recorder.get("duration_s", webcam_duration)) - webcam_duration)
    webcam_advance = max(0.0, video_zero_offset - webcam_zero_offset)

    def video_event(phase):
        value = event_offset(phases, phase)
        return max(0.0, value - video_zero_offset) if value is not None else None

    obs_start = (video_event(PROFILE["observe"]) or video_event("spiral") or 0.0) + 0.1
    done_event = video_event("done")
    gn_start = video_event("grasp") or video_event("no-grasp") or raw_end
    obs_hold = max(obs_start, gn_start - 0.3)
    end_slide_start = done_event if done_event is not None else max(gn_start, total - 4.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing output: {args.output}")
    scene_match = re.search(r"scene[-_ ]*0*(\d+)",
                            f"{trial.parent.name}/{trial.name}", re.IGNORECASE)
    scene_name = f"Scene {int(scene_match.group(1)):02d}" if scene_match else "Scene"

    with tempfile.TemporaryDirectory(prefix="spiral_compose_") as temporary:
        tmp = Path(temporary)
        if oow:
            evidence = render_oow_evidence(trial, timing, tmp / "oow_evidence.png")
            tile_source, slide_source, tile_label = evidence, evidence, "OOW\nFailure"
        else:
            candidates = sorted((trial / "perception").glob("*_gn16.png"),
                                key=lambda p: int(re.search(r"dump(\d+)", p.name).group(1)))
            if not candidates:
                raise SystemExit("No grasp-network panel found")
            slide_source = tile_source = candidates[-1]
            tile_label = "Grasp\nEvaluation"
            if video_event("grasp") is not None and timing.get("real_grasp"):
                selected = int(timing["real_grasp"]["rotation_idx"])
                tile_source = slide_source.parent / f"{slide_source.stem}_tiles/bin_{selected:02d}.png"
        obs_tile = None
        if not observation_inset_present(annotated_d455, obs_start + 1.0):
            obs_tile = render_observation_tile(trial, tmp / "observation_tile.png")
            print("annotated video has no observation inset; "
                  + ("tile rebuilt from the saved observation masks" if obs_tile else
                     "no saved masks either, tile left as recorded"))
        steps = step_timings(phases)
        render_dashboard_video(tmp / "dashboard.mp4", total, video_zero_offset, steps, phases,
                               scene_name, arc_cm, oow)
        render_speed_badge(tmp / "speed.png")
        render_text_badge(tmp / "d455_label.png", "Orthographic Camera", LABEL_D455_W, SPEED_BADGE_H)
        render_text_badge(tmp / "obs_label.png", PROFILE["obs_label"], LABEL_OBS_W, TILE_LABEL_H)
        render_text_badge(tmp / "gn_label.png", tile_label, LABEL_GN_W, TILE_LABEL_H)
        render_fitted_image(tile_source, tmp / "selected_gn.png", TILE_W, TILE_H)
        render_fitted_image(slide_source, tmp / "gn_slide.png", CANVAS_W, CANVAS_H)
        from compose_closed_loop_memory import render_state_video
        if (trial / "metadata" / "trajectory.json").exists():
            pushes = (pmbs_pushes if PROFILE is PROFILES["pmbs"] else spiral_pushes)(trial, phases, video_zero_offset)
        else:
            # A run that died before writing its trace still has a valid camera record; the
            # executed-push overlay is simply omitted (noted in the render log).
            pushes = []
            print("no metadata/trajectory.json: rendering without the executed-push overlay")
        to_panel = camera_projection(trial)
        if PROFILE is PROFILES["pmbs"]:
            render_state_video(tmp / "trace.mov", total, 10, (MAIN_W, MAIN_H),
                               lambda ts: pmbs_pose_state(ts, pushes),
                               lambda st: draw_pmbs_pose(st, pushes, to_panel), alpha=True)
            render_pmbs_pose_legend(tmp / "trace_legend.png")
        else:
            render_state_video(tmp / "trace.mov", total, 10, (MAIN_W, MAIN_H),
                               lambda ts: trace_state(ts, pushes),
                               lambda st: draw_trace(st, pushes, to_panel), alpha=True)
            render_trace_legend(tmp / "trace_legend.png")
        trace_start = pushes[0][0] if pushes else total

        loop = lambda path: ["-loop", "1", "-framerate", "30", "-i", str(path)]
        d455_setting, webcam_setting = (args.d455_gamma, 1.08), (args.webcam_gamma, 1.04)
        if args.exposure == "measured":
            d455_setting, d455_meta = exposure.video_transfer(
                raw_d455, "d455", crop="crop=iw*0.75:ih:iw*0.25:0,hflip,vflip", saturation=1.08)
            webcam_setting, webcam_meta = exposure.video_transfer(
                raw_webcam, "webcam", crop="crop=iw*0.75:ih:iw*0.05:0", saturation=1.04)
            exposure.write_metadata(args.output.with_suffix(".exposure.json"),
                                    {"d455": d455_meta, "webcam": webcam_meta})
            print(f"exposure: D455 gain {d455_meta['brightness_gain']:.3f}, "
                  f"webcam gain {webcam_meta['brightness_gain']:.3f}")
        command = [
            find_encoder(), "-hide_banner", "-loglevel", "warning", "-nostdin", "-stats", "-y",
            "-i", str(annotated_d455), "-i", str(raw_d455), "-i", str(raw_webcam),
            "-i", str(tmp / "dashboard.mp4"),
            *loop(tmp / "speed.png"), *loop(tmp / "d455_label.png"),
            *loop(tmp / "obs_label.png"), *loop(tmp / "gn_label.png"),
            *loop(tmp / "selected_gn.png"), *loop(tmp / "gn_slide.png"),
            "-i", str(tmp / "trace.mov"), *loop(tmp / "trace_legend.png"),
            *(loop(obs_tile) if obs_tile else []),
            "-filter_complex", build_filter(total, raw_end, obs_start, obs_hold, gn_start,
                                            end_slide_start, webcam_advance, trace_start, args.preview,
                                            d455_setting, webcam_setting,
                                            obs_tile_input=13 if obs_tile else None),
            "-map", "[out]", "-an", "-r", "15" if args.preview else "30", "-t", f"{total:.6f}",
            "-c:v", "libx264", "-preset", "veryfast" if args.preview else args.preset,
            "-crf", "32" if args.preview else str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        print("Creating", args.output)
        action_label = "PMBS pushes" if PROFILE is PROFILES["pmbs"] else "spiral steps"
        print(f"Duration {total:.2f} s; stop {stop}; {len(pushes)} {action_label}; "
              f"observation tile {obs_start:.2f}-{obs_hold:.2f} s; final tile from {gn_start:.2f} s")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
