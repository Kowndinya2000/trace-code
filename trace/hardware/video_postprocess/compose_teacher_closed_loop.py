#!/usr/bin/env python3
"""Compose one teacher closed-loop trial into the master_v4 review layout.

Same canvas, palette, borders, badges, rotations and end slide as
compose_closed_loop.py (master_v4). What differs follows the teacher pipeline:
there is no digital twin, so the dashboard is PREPARE / PERCEIVE / SOLVE ON THE
ROBOT, the top-left tile is the re-sensed observation, the webcam tile sits
centred under the two top tiles where the simulator tile used to be, and the D455
panel carries the teacher's EEF push trace in pink, growing push by push.
Source recordings are read only; the output is always a new file.
"""

from __future__ import annotations

import argparse
from camera_sources import third_person
import json
import re
import subprocess
import tempfile
from pathlib import Path

import exposure

from compose_closed_loop import (
    CANVAS_H, CANVAS_W, DASH_H, DASH_W, DASH_Y, LABEL_D455_W, LABEL_GN_W, LABEL_OBS_W,
    MAIN_H, MAIN_W, MAIN_X, MAIN_Y, MARGIN, SPEED_BADGE_H, TILE_H, TILE_LABEL_H, TILE_W,
    TILE_X1, TILE_X2, TILE_Y1, TILE_Y2, duration, event_offset, find_encoder, load_json,
    render_fitted_image, render_speed_badge, render_text_badge,
)

# Re-sensed observation tile in the 1280 x 720 annotated teacher recording,
# below its 36 px title strip.
OBS_X, OBS_Y, OBS_SIZE = 14, 194, 332
WEBCAM_X = (TILE_X1 + TILE_X2 + TILE_W) // 2 - TILE_W // 2

TITLE_W, PREPARE_W, PERCEIVE_W, SOLVE_W = 320, 360, 340, 520
TOTAL_W = DASH_W - TITLE_W - PREPARE_W - PERCEIVE_W - SOLVE_W
LATO = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
LATO_SEMI = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
GREEN, BLUE, GRAY, WHITE, PANEL = "#34A853", "#4285F4", "#9AA0A6", "#F1F3F4", "#202124"


def status_text(detail: str) -> str:
    """Live sub-line under the loop row, in the wording agreed for the teacher video."""
    text = re.sub(r"returning to saved push pose for action \d+", "resuming push", detail)
    text = re.sub(r"original grasp network", "grasp network", text, flags=re.IGNORECASE)
    return text.replace("primitive", "action")


def detail_at(phases: dict, ts: float) -> str:
    current = ""
    for event in phases["events"]:
        if event["t"] - phases["t0"] > ts:
            break
        current = str(event.get("detail", ""))
    return "" if current.startswith("internal:") else current


def step_timings(phases: dict) -> dict:
    """Phase-clock seconds, using record_cameras.build_steps' first-occurrence rule."""
    first = lambda phase: event_offset(phases, phase)
    grasp = first("grasp")
    end_loop = grasp if grasp is not None else first("no-grasp")
    done = first("done")
    return {
        # The move to experiment home is not shown (0.04 s on every hardware
        # trial); model loading runs until the first observation so rows still tile.
        "models": (first("teacher-load"), first("teacher-sense")),
        "segmentation": (first("teacher-sense"), first("teacher-gn")),
        "loop": (first("teacher-gn"), end_loop if end_loop is not None else done),
        "grasp": (grasp, done if grasp is not None else None),
        "start": first("teacher-load"),
        "done": done,
    }


def state(span: tuple, ts: float) -> tuple[str, float]:
    start, end = span
    if start is None or ts < start:
        return "pending", 0.0
    if end is None or ts < end:
        return "active", ts - start
    return "done", end - start


def draw_dashboard(ts: float, steps: dict, phases: dict, scene_name: str):
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGB", (DASH_W, DASH_H), PANEL)
    draw = ImageDraw.Draw(panel)
    head = ImageFont.truetype(LATO_SEMI, 17)
    body = ImageFont.truetype(LATO, 18)
    big_body = ImageFont.truetype(LATO, 20)
    small = ImageFont.truetype(LATO, 16)
    status = ImageFont.truetype(LATO, 15)
    color = {"pending": GRAY, "active": BLUE, "done": GREEN}

    def group_color(names):
        states = [state(steps[name], ts)[0] for name in names]
        if all(s == "done" for s in states):
            return GREEN
        return BLUE if any(s != "pending" for s in states) else GRAY

    def dot(x, y, phase_state):
        if phase_state == "pending":
            draw.ellipse((x, y + 5, x + 10, y + 15), outline=GRAY, width=2)
        else:
            draw.ellipse((x, y + 5, x + 10, y + 15), fill=color[phase_state])

    def step_row(x0, width, y, label, name, font=body):
        phase_state, elapsed = state(steps[name], ts)
        fill = color[phase_state]
        dot(x0 + 18, y, phase_state)
        draw.text((x0 + 36, y), label, font=font, fill=fill)
        value = "-" if phase_state == "pending" else f"{elapsed:.1f} s"
        draw.text((x0 + width - draw.textlength(value, font=font) - 18, y), value,
                  font=font, fill=fill)
        return phase_state

    # Title
    title_font = ImageFont.truetype(LATO_SEMI, 25)
    policy_font = ImageFont.truetype(LATO_SEMI, 20)
    draw.text((18, 22), scene_name, font=title_font, fill=WHITE)
    draw.text((18, 79), "Online Teacher", font=policy_font, fill=WHITE)
    draw.text((18, 118), "Closed-Loop", font=ImageFont.truetype(LATO_SEMI, 19), fill=WHITE)

    # PREPARE
    x = TITLE_W
    draw.text((x + 18, 18), "PREPARE", font=head, fill=group_color(("models",)))
    step_row(x, PREPARE_W, 67, "Teacher + perception models", "models")

    # PERCEIVE, laid out like master_v4's two-line segmentation entry
    x += PREPARE_W
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
    draw.text((x + PERCEIVE_W - draw.textlength(value, font=small) - 18, 136), value,
              font=small, fill=seg_color)

    # SOLVE ON THE ROBOT
    x += PERCEIVE_W
    draw.text((x + 18, 18), "SOLVE ON THE ROBOT   re-sensed before every action", font=head,
              fill=group_color(("loop", "grasp")))
    if step_row(x, SOLVE_W, 67, "Grasp check + teacher push", "loop") == "active":
        live = status_text(detail_at(phases, ts))
        if live:
            draw.text((x + 48, 97), live, font=status, fill=GRAY)
    step_row(x, SOLVE_W, 143, "Grasp + retrieve", "grasp")

    # Total Time: record_cameras' clock, first step start to "done"
    x += SOLVE_W
    start, done = steps["start"], steps["done"]
    total = 0.0 if start is None else max(0.0, (min(ts, done) if done is not None else ts) - start)
    draw.text((x + 22, 52), "Total Time:", font=ImageFont.truetype(LATO_SEMI, 20), fill=WHITE)
    draw.text((x + 22, 88), f"{total:.1f} s", font=ImageFont.truetype(LATO_SEMI, 32), fill=WHITE)

    for boundary in (TITLE_W, TITLE_W + PREPARE_W, TITLE_W + PREPARE_W + PERCEIVE_W,
                     TITLE_W + PREPARE_W + PERCEIVE_W + SOLVE_W):
        draw.line((boundary, 0, boundary, DASH_H), fill=GRAY, width=2)
    return panel


def render_dashboard_video(path: Path, total: float, zero_offset: float, steps: dict,
                           phases: dict, scene_name: str) -> None:
    fps = 10
    command = [
        find_encoder(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{DASH_W}x{DASH_H}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_number in range(round(total * fps) + 1):
            # Dashboard state is on the phase clock; the video starts zero_offset later.
            ts = frame_number / fps + zero_offset
            process.stdin.write(draw_dashboard(ts, steps, phases, scene_name).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


# ------------------------------------------------------------ EEF trace
PINK = (255, 64, 160)
PUSH_Z = 0.020               # push height used by the student overlay as well
RAW_W, RAW_H, RAW_CROP_X = 1280, 720, 320


def teacher_pushes(trial: Path, phases: dict, video_zero_offset: float) -> list:
    """[(t_start, t_end, [eef, wp1, wp2])] in video seconds, one entry per executed push."""
    traj = load_json(trial / "metadata" / "trajectory.json")
    t0 = phases["t0"]
    push_t = [e["t"] - t0 - video_zero_offset for e in phases["events"] if e["phase"] == "teacher-push"]
    retract_t = [e["t"] - t0 - video_zero_offset for e in phases["events"] if e["phase"] == "teacher-retract"]
    pushes = []
    for k, (dense, wp) in enumerate(zip(traj["dense"], traj["waypoints"])):
        if k >= len(push_t):
            break
        end = retract_t[k] if k < len(retract_t) else push_t[k] + 0.55
        pushes.append((push_t[k], end, [dense["eef"][:2], wp["wp1"][:2], wp["wp2"][:2]]))
    return pushes


def camera_projection(trial: Path):
    import numpy as np

    records = sorted((trial / "perception").glob("*_gn16_predictions.npz"))
    with np.load(records[-1], allow_pickle=False) as data:
        b2c = np.linalg.inv(data["camera_to_base"])
    K = load_json(trial / "perception" / "d455_topdown_dump0_K.json")
    scale = MAIN_W / (RAW_W - RAW_CROP_X)

    def to_panel(x_sim, y_sim):
        p = b2c @ np.array([y_sim, -x_sim, PUSH_Z, 1.0])     # record_cameras._project
        u, v = p[0] * K["fx"] / p[2] + K["cx"], p[1] * K["fy"] / p[2] + K["cy"]
        return ((RAW_W - 1 - u) * scale, (RAW_H - 1 - v) * scale)   # crop 75 %, flip, scale
    return to_panel


def trace_state(ts: float, pushes: list):
    """(pushes fully drawn, fraction of the push in progress), stepped to 0.1."""
    done, frac = 0, 0.0
    for start, end, _ in pushes:
        if ts >= end:
            done += 1
        elif ts >= start:
            frac = round(min(1.0, (ts - start) / max(1e-3, end - start)), 1)
            break
        else:
            break
    return done, frac


def draw_trace(state, pushes: list, to_panel, supersample: int = 2):
    """Pink EEF trace of the teacher's pushes, growing push by push; latest push haloed."""
    from PIL import Image, ImageDraw

    S = supersample
    done, frac = state
    layer = Image.new("RGBA", (MAIN_W * S, MAIN_H * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    segments = []
    for k in range(min(len(pushes), done + (1 if frac > 0 else 0))):
        pts = [tuple(c * S for c in to_panel(*p)) for p in pushes[k][2]]
        if k == done:            # push in progress: grow along eef -> wp1 -> wp2
            grown, left = [pts[0]], frac * 2.0
            for a, b in zip(pts, pts[1:]):
                f = min(1.0, left); left -= f
                grown.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                if left <= 0:
                    break
            pts = grown
        segments.append(pts)
    if not segments:
        return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)
    path = [segments[0][0]]
    for seg in segments:
        path.extend(seg[1:] if path[-1] == seg[0] else seg)
    lens = 24 * S
    latest = segments[-1]
    if len(latest) > 1:
        draw.line(latest, fill=PINK + (85,), width=lens, joint="curve")
    for x, y in (latest[0], latest[-1]):
        draw.ellipse((x - lens / 2, y - lens / 2, x + lens / 2, y + lens / 2), fill=PINK + (85,))
    if len(path) > 1:
        draw.line(path, fill=PINK + (255,), width=4 * S, joint="curve")
    for seg in segments:
        for x, y in seg:
            r = 4 * S
            draw.ellipse((x - r, y - r, x + r, y + r), fill=PINK + (255,),
                         outline=(255, 255, 255, 255), width=S)
    return layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS)


def render_trace_legend(path: Path) -> None:
    """Small badge-style legend for the trace, centred at the bottom of the D455 panel."""
    from PIL import Image, ImageDraw, ImageFont

    S, H = 2, SPEED_BADGE_H
    font = ImageFont.truetype(LATO_SEMI, 13 * S)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    labels = ("Executed EEF path", "Latest push")
    widths = [probe.textlength(t, font=font) for t in labels]
    icon, gap, pad = 30 * S, 16 * S, 10 * S
    width = int(pad + icon + 6 * S + widths[0] + gap + icon + 6 * S + widths[1] + pad)
    badge = Image.new("RGBA", (width, H * S), (32, 33, 36, 224))
    draw = ImageDraw.Draw(badge)
    cy = H * S // 2
    x = pad
    draw.line((x, cy + 3 * S, x + 12 * S, cy - 3 * S, x + icon, cy), fill=PINK + (255,), width=3 * S)
    for px, py in ((x, cy + 3 * S), (x + 12 * S, cy - 3 * S), (x + icon, cy)):
        draw.ellipse((px - 3 * S, py - 3 * S, px + 3 * S, py + 3 * S), fill=PINK + (255,),
                     outline=(255, 255, 255, 255), width=S)
    x += icon + 6 * S
    draw.text((x, 6 * S), labels[0], font=font, fill="#F1F3F4")
    x += widths[0] + gap
    draw.line((x + 6 * S, cy, x + icon - 6 * S, cy), fill=PINK + (90,), width=14 * S)
    for px in (x + 6 * S, x + icon - 6 * S):
        draw.ellipse((px - 7 * S, cy - 7 * S, px + 7 * S, cy + 7 * S), fill=PINK + (90,))
    draw.line((x + 6 * S, cy, x + icon - 6 * S, cy), fill=PINK + (255,), width=3 * S)
    x += icon + 6 * S
    draw.text((x, 6 * S), labels[1], font=font, fill="#F1F3F4")
    badge.resize((width // S, H), Image.Resampling.LANCZOS).save(path)


def build_filter(total: float, raw_end: float, obs_start: float, obs_hold: float,
                 gn_start: float, end_slide_start: float, webcam_advance: float,
                 trace_start: float, preview: bool,
                 d455_tone: tuple[float, float] | str = (1.0, 1.0),
                 webcam_tone: tuple[float, float] | str = (1.0, 1.0)) -> str:
    def tone(setting: tuple[float, float] | str) -> str:
        if isinstance(setting, str):
            return setting
        gamma, saturation = setting
        if gamma == 1.0 and saturation == 1.0:
            return ""
        return (f"format=gbrp,lutrgb=r=gammaval({1 / gamma:.4f}):"
                f"g=gammaval({1 / gamma:.4f}):b=gammaval({1 / gamma:.4f}),"
                f"format=yuv420p,eq=saturation={saturation},")

    tail = max(0.0, total - raw_end + 0.25)
    obs_tail = max(0.0, total - obs_hold + 0.25)
    obs_xpad = (TILE_W - TILE_H) // 2
    chains = [
        f"color=c=0x111820:s={CANVAS_W}x{CANVAS_H}:r=30:d={total:.6f}[bg]",
        (f"[1:v]crop=iw*0.75:ih:iw*0.25:0,hflip,vflip,"
         f"{tone(d455_tone)}"
         f"tpad=stop_mode=clone:stop_duration={tail:.6f},"
         f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         f"scale={MAIN_W}:{MAIN_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         "setsar=1[d455raw]"),
        f"[10:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,format=rgba[tracelayer]",
        ("[d455raw][tracelayer]overlay=0:0:format=auto,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xE91E63:t=6[d455base]"),
        (f"[0:v]crop={OBS_SIZE}:{OBS_SIZE}:{OBS_X}:{OBS_Y},hflip,vflip,"
         f"trim=end={obs_hold:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={obs_tail:.6f},"
         f"scale={TILE_H}:{TILE_H}:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:{obs_xpad}:0:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x34A853:t=4[obsbase]"),
        f"[3:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1[dashboard]",
        "[4:v]format=rgba,split=2[badge455][badgecam]",
        "[5:v]format=rgba[label455]",
        "[6:v]format=rgba[labelobs]",
        "[7:v]format=rgba[labelgn]",
        "[d455base][badge455]overlay=12:H-h-12[d455speed]",
        "[d455speed][label455]overlay=W-w-12:H-h-12[d455labelled]",
        "[11:v]format=rgba[tracelegend]",
        f"[d455labelled][tracelegend]overlay=(W-w)/2:H-h-12:enable='gte(t,{trace_start:.6f})'[d455]",
        "[obsbase][labelobs]overlay=W-w-12:H-h-12[obs]",
        (f"[2:v]trim=start={webcam_advance:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={tail + webcam_advance + 2.0:.6f},"
         f"trim=duration={total:.6f},crop=iw*0.75:ih:iw*0.05:0,"
         f"{tone(webcam_tone)}"
         f"scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x4285F4:t=4[webcambase]"),
        "[webcambase][badgecam]overlay=12:H-h-12[webcam]",
        (f"[8:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xFB8C00:t=4[gnselected]"),
        "[gnselected][labelgn]overlay=W-w-12:H-h-12[gn]",
        f"[9:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1[gnslide]",
        f"[bg][dashboard]overlay={MARGIN}:{DASH_Y}[c0]",
        f"[c0][d455]overlay={MAIN_X}:{MAIN_Y}[c1]",
        f"[c1][obs]overlay={TILE_X1}:{TILE_Y1}:enable='gte(t,{obs_start:.6f})'[c2]",
        f"[c2][gn]overlay={TILE_X2}:{TILE_Y1}:enable='gte(t,{gn_start:.6f})'[c3]",
        f"[c3][webcam]overlay={WEBCAM_X}:{TILE_Y2}[c4]",
        f"[c4][gnslide]overlay=0:0:enable='gte(t,{end_slide_start:.6f})'[c5]",
    ]
    chains.append("[c5]scale=960:540:flags=lanczos,format=yuv420p[out]" if preview
                  else "[c5]format=yuv420p[out]")
    return ";".join(chains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="teacher trial directory (…/sceneN/trial1)")
    parser.add_argument("output", type=Path, help="new MP4 to create")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--limit", type=float, help="render only the first N seconds")
    parser.add_argument("--preview", action="store_true",
                        help="fast low-size check render: 960x540, 15 fps, crf 32, veryfast")
    parser.add_argument("--exposure", choices=("measured", "fixed"), default="measured",
                        help="measured: exposure.video_transfer; fixed: legacy constant gamma")
    parser.add_argument("--d455-gamma", type=float, default=1.8)
    parser.add_argument("--webcam-gamma", type=float, default=1.3)
    args = parser.parse_args()

    trial = args.trial.resolve()
    annotated_d455 = trial / "d455_topdown.mp4"
    raw_d455 = trial / "d455_topdown_raw.mp4"
    raw_webcam, webcam_recorder = third_person(trial)
    for source in (annotated_d455, raw_d455, raw_webcam):
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")

    phases = load_json(trial / "metadata" / "phases.json")
    timing = load_json(trial / "metadata" / "real_timing.json")
    if timing.get("controller") != "teacher_full_observation":
        raise SystemExit(f"Not a teacher closed-loop trial: controller={timing.get('controller')}")
    total = min(duration(annotated_d455), args.limit or float("inf"))
    raw_duration = duration(raw_d455)
    webcam_duration = duration(raw_webcam)
    raw_end = min(raw_duration, total)
    recorder = load_json(trial / "metadata" / "d455_topdown_rec.json")
    # Same clock alignment as compose_closed_loop.py.
    video_zero_offset = max(0.0, float(recorder.get("duration_s", raw_duration)) - raw_duration)
    webcam_zero_offset = max(
        0.0, float(webcam_recorder.get("duration_s", webcam_duration)) - webcam_duration)
    webcam_advance = max(0.0, video_zero_offset - webcam_zero_offset)

    def video_event(phase: str) -> float | None:
        value = event_offset(phases, phase)
        return max(0.0, value - video_zero_offset) if value is not None else None

    # The annotated tile appears with the first grasp-network evaluation. From the
    # grasp on, the same screen region carries the grasp panel instead, so hold the
    # last re-sensed observation from just before the grasp.
    obs_start = (video_event("teacher-gn") or 0.0) + 0.1
    done_event = video_event("done")
    gn_start = video_event("grasp") or video_event("no-grasp") or raw_end
    obs_hold = max(obs_start, gn_start - 0.3)
    end_slide_start = done_event if done_event is not None else max(gn_start, total - 4.0)

    gn_candidates = sorted((trial / "perception").glob("*_gn16.png"),
                           key=lambda p: int(re.search(r"dump(\d+)", p.name).group(1)))
    if not gn_candidates:
        raise SystemExit(f"No grasp-network panel found under {trial / 'perception'}")
    gn_panel = gn_candidates[-1]
    selected_gn_panel = gn_panel
    if video_event("grasp") is not None:
        import numpy as np

        predictions_path = gn_panel.with_name(gn_panel.stem + "_predictions.npz")
        with np.load(predictions_path, allow_pickle=False) as predictions:
            metadata = json.loads(str(predictions["metadata"].item()))
        selected_bin = int(metadata["grasp"]["rotation_idx"])
        selected_gn_panel = gn_panel.parent / f"{gn_panel.stem}_tiles/bin_{selected_bin:02d}.png"
        if not selected_gn_panel.is_file():
            raise SystemExit(f"Missing selected grasp bin: {selected_gn_panel}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing output: {args.output}")

    d455_setting: tuple[float, float] | str = (args.d455_gamma, 1.08)
    webcam_setting: tuple[float, float] | str = (args.webcam_gamma, 1.04)
    if args.exposure == "measured":
        d455_setting, d455_meta = exposure.video_transfer(
            raw_d455, "d455", crop="crop=iw*0.75:ih:iw*0.25:0,hflip,vflip",
            saturation=1.08)
        webcam_setting, webcam_meta = exposure.video_transfer(
            raw_webcam, "webcam", crop="crop=iw*0.75:ih:iw*0.05:0",
            saturation=1.04)
        exposure.write_metadata(args.output.with_suffix(".exposure.json"),
                                {"d455": d455_meta, "webcam": webcam_meta})
        print(f"exposure: D455 gain {d455_meta['brightness_gain']:.3f}, "
              f"webcam gain {webcam_meta['brightness_gain']:.3f}")

    scene_match = re.search(r"scene\s*0*(\d+)", trial.parent.name, re.IGNORECASE)
    scene_name = f"Scene {int(scene_match.group(1)):02d}" if scene_match else "Scene"

    with tempfile.TemporaryDirectory(prefix="teacher_compose_") as temporary:
        tmp = Path(temporary)
        render_dashboard_video(tmp / "dashboard.mp4", total, video_zero_offset,
                               step_timings(phases), phases, scene_name)
        render_speed_badge(tmp / "speed.png")
        render_text_badge(tmp / "d455_label.png", "Orthographic Camera", LABEL_D455_W, SPEED_BADGE_H)
        render_text_badge(tmp / "obs_label.png", "Re-sensed\nObservation", LABEL_OBS_W, TILE_LABEL_H)
        render_text_badge(tmp / "gn_label.png", "Grasp\nEvaluation", LABEL_GN_W, TILE_LABEL_H)
        render_fitted_image(selected_gn_panel, tmp / "selected_gn.png", TILE_W, TILE_H)
        render_fitted_image(gn_panel, tmp / "gn_slide.png", CANVAS_W, CANVAS_H)
        from compose_closed_loop_memory import render_state_video
        pushes = teacher_pushes(trial, phases, video_zero_offset)
        to_panel = camera_projection(trial)
        render_state_video(tmp / "trace.mov", total, 10, (MAIN_W, MAIN_H),
                           lambda ts: trace_state(ts, pushes),
                           lambda st: draw_trace(st, pushes, to_panel), alpha=True)
        render_trace_legend(tmp / "trace_legend.png")
        trace_start = pushes[0][0] if pushes else total

        loop = lambda path: ["-loop", "1", "-framerate", "30", "-i", str(path)]
        command = [
            find_encoder(), "-hide_banner", "-loglevel", "warning", "-nostdin", "-stats", "-y",
            "-i", str(annotated_d455), "-i", str(raw_d455), "-i", str(raw_webcam),
            "-i", str(tmp / "dashboard.mp4"),
            *loop(tmp / "speed.png"), *loop(tmp / "d455_label.png"),
            *loop(tmp / "obs_label.png"), *loop(tmp / "gn_label.png"),
            *loop(tmp / "selected_gn.png"), *loop(tmp / "gn_slide.png"),
            "-i", str(tmp / "trace.mov"), *loop(tmp / "trace_legend.png"),
            "-filter_complex", build_filter(total, raw_end, obs_start, obs_hold, gn_start,
                                            end_slide_start, webcam_advance, trace_start,
                                            args.preview, d455_setting, webcam_setting),
            "-map", "[out]", "-an", "-r", "15" if args.preview else "30", "-t", f"{total:.6f}",
            "-c:v", "libx264",
            "-preset", "veryfast" if args.preview else args.preset,
            "-crf", "32" if args.preview else str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        print("Creating", args.output)
        print(f"Duration {total:.2f} s; webcam advance {webcam_advance:.3f} s; "
              f"observation tile {obs_start:.2f}-{obs_hold:.2f} s; grasp tile from {gn_start:.2f} s")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
