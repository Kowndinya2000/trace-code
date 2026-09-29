#!/usr/bin/env python3
"""Shared layout for the annotated trial videos: canvas, palette, panels, traces, end slide.

The D455 camera image is rotated 180 degrees while its dashboard and diagnostic tiles
remain upright and outside the camera image. Source recordings are read only; the output
is always a new file.

This module is imported by the four composers that are actually run:

    compose_closed_loop_memory.py     TRACE          (adds the robot-observation panel)
    compose_teacher_closed_loop.py    Online Teacher
    compose_open_loop.py              Teacher Replay
    compose_spiral_closed_loop.py     Spiral and PMBS

Its own `main` is an earlier TRACE entry point that predates the measured exposure
transfer, so running it directly would produce an uncorrected video. It refuses instead;
use compose_closed_loop_memory.py for TRACE trials.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


CANVAS_W = 1920
CANVAS_H = 1080
MARGIN = 10
DASH_W = 1900
DASH_H = 260
DASH_Y = 10
CONTENT_Y = 300
MAIN_X = 30
MAIN_Y = 360
MAIN_W = 800
MAIN_H = 600
TILE_W = 500
TILE_H = 350
TILE_X1 = 860
TILE_X2 = 1380
TILE_Y1 = 300
TILE_Y2 = 670
SIM_CONTENT_SIZE = 350

# Coordinates in the original 1280 x 720 annotated D455 recording.
STANDARD_BAND_H = 152
EXPANDED_BAND_H = 178
OBS_X, OBS_Y, OBS_W, OBS_H = 14, 158, 333, 371
GRASP_X, GRASP_Y, GRASP_W, GRASP_H = 14, 172, 333, 386

# The simulator render is 3840 square. Its yellow workspace frame runs from
# about 570 through 3270; use the clean interior so rotating the camera view
# cannot rotate the border or any surrounding labels.
SIM_INNER_X = 583
SIM_INNER_Y = 583
SIM_INNER_SIZE = 2676
SIM_WORKSPACE_LABEL_H = 270
TITLE_OUT_W = 320
PERCEPTION_OUT_W = 360
SOLVE_OUT_W = 340
EXECUTE_OUT_W = 410
TOTAL_OUT_W = 470
PERCEPTION_OUT_X = TITLE_OUT_W
SOLVE_OUT_X = PERCEPTION_OUT_X + PERCEPTION_OUT_W
EXECUTE_OUT_X = SOLVE_OUT_X + SOLVE_OUT_W
TOTAL_OUT_X = EXECUTE_OUT_X + EXECUTE_OUT_W
SOURCE_PERCEPTION_X = 295
SOURCE_SOLVE_X = 573
SOURCE_SOLVE_W = 193
SOURCE_EXECUTE_X = 766
SOURCE_EXECUTE_W = 240
SOURCE_TOTAL_X = 1006
SOURCE_TOTAL_W = 274
OBS_LABEL_H = 36
SPEED_BADGE_W = 104
LABEL_D455_W = 144
LABEL_OBS_W = 97
LABEL_GN_W = 80
LABEL_SIM_W = 57
SPEED_BADGE_H = 28
TILE_LABEL_H = 31
ATLAS_W = 720
ATLAS_H = 551


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    return float(subprocess.check_output(command, text=True).strip())


def event_offset(phases: dict, phase: str, *, first: bool = True) -> float | None:
    matches = [event["t"] - phases["t0"] for event in phases["events"]
               if event.get("phase") == phase]
    if not matches:
        return None
    return matches[0] if first else matches[-1]


def find_encoder() -> str:
    """Find an FFmpeg build with libx264 (the PATH build may be decode-only)."""
    candidates = [os.environ.get("FFMPEG_BIN"), shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        result = subprocess.run([candidate, "-hide_banner", "-encoders"],
                                text=True, capture_output=True)
        if result.returncode == 0 and "libx264" in result.stdout:
            return candidate
    raise SystemExit("No FFmpeg build with libx264 was found")


def legacy_solver_breakdown(phases: dict, sim_timing: dict) -> dict | None:
    """Recover the honest portion of a legacy coarse solve annotation."""
    twin = event_offset(phases, "twin")
    solve = event_offset(phases, "solve")
    done = event_offset(phases, "solve-done")
    steps = sim_timing.get("num_steps")
    control_hz = sim_timing.get("fps") or sim_timing.get("control_hz")
    if None in (twin, solve, done) or not steps or not control_hz:
        return None
    simulated = float(steps) / float(control_hz)
    combined = done - solve
    return {
        "schema_version": 1,
        "source_schema": "legacy outer solve markers",
        "isaac_gym_and_teacher_load_wall_seconds": solve - twin,
        "ppo_rollout_simulated_seconds": simulated,
        "legacy_combined_solve_wall_seconds": combined,
        "solver_accounting_remainder_seconds": combined - simulated,
        "solver_accounting_remainder_contains": [
            "scene settling", "PPO wall-time minus simulated-time difference",
            "final 16-orientation grasp evaluation", "trajectory and JSON export"],
        "caveat": ("The legacy run did not timestamp the internal solve boundaries. "
                   "The accounting remainder reconciles the old wall interval against "
                   "simulated rollout time; its components cannot be recovered exactly."),
    }


def draw_solver_panel(breakdown: dict, timestamp: float, twin_start: float,
                      solve_start: float):
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGB", (SOLVE_OUT_W, DASH_H), "#202124")
    draw = ImageDraw.Draw(panel)
    regular = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
    semibold = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
    head = ImageFont.truetype(semibold, 17)
    body = ImageFont.truetype(regular, 15)
    small = ImageFont.truetype(regular, 11)
    green, blue, gray = "#34A853", "#4285F4", "#9AA0A6"
    load_duration = breakdown["isaac_gym_and_teacher_load_wall_seconds"]
    rollout_duration = breakdown["ppo_rollout_simulated_seconds"]
    overhead_duration = breakdown["solver_accounting_remainder_seconds"]
    rollout_end = solve_start + rollout_duration
    overhead_end = rollout_end + overhead_duration

    def state(start: float, end: float) -> tuple[str, float]:
        if timestamp < start:
            return "pending", 0.0
        if timestamp < end:
            return "active", timestamp - start
        return "done", end - start

    states = [
        state(twin_start, solve_start),
        state(solve_start, rollout_end),
        state(rollout_end, overhead_end),
    ]
    heading_color = gray if timestamp < twin_start else (
        green if timestamp >= overhead_end else blue)
    draw.line((0, 0, 0, DASH_H), fill=gray, width=2)
    draw.text((18, 18), "SOLVE   digital twin", font=head, fill=heading_color)
    rows = [
        ("Isaac Gym + teacher", breakdown["isaac_gym_and_teacher_load_wall_seconds"], "wall"),
        ("PPO rollout", breakdown["ppo_rollout_simulated_seconds"], "sim"),
        ("Plan Post-Processing", breakdown["solver_accounting_remainder_seconds"], "wall"),
    ]
    for y, (label, final_value, suffix), (phase_state, elapsed) in zip(
            (61, 112, 163), rows, states):
        color = {"pending": gray, "active": blue, "done": green}[phase_state]
        if phase_state == "pending":
            draw.ellipse((18, y + 5, 28, y + 15), outline=gray, width=2)
        else:
            draw.ellipse((18, y + 5, 28, y + 15), fill=color)
        draw.text((36, y), label, font=body, fill=color)
        value_text = "-" if phase_state == "pending" else f"{min(elapsed, final_value):.1f} s"
        value_w = draw.textlength(value_text, font=body)
        draw.text((SOLVE_OUT_W - value_w - 16, y + 21), value_text,
                  font=body, fill=color)
        draw.text((36, y + 21), suffix, font=small, fill=gray)
    return panel


def render_solver_panel_video(path: Path, breakdown: dict, total: float,
                              twin_start: float, solve_start: float) -> None:
    fps = 10
    command = [
        find_encoder(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
        f"{SOLVE_OUT_W}x{DASH_H}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_number in range(round(total * fps) + 1):
            timestamp = frame_number / fps
            frame = draw_solver_panel(breakdown, timestamp, twin_start, solve_start)
            process.stdin.write(frame.convert("RGB").tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def draw_execution_and_time_panel(timestamp: float, timings: dict):
    """Draw the EXECUTE and Total Time columns in the shared Lato family."""
    from PIL import Image, ImageDraw, ImageFont

    width = EXECUTE_OUT_W + TOTAL_OUT_W
    panel = Image.new("RGB", (width, DASH_H), "#202124")
    draw = ImageDraw.Draw(panel)
    regular = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
    semibold = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
    head = ImageFont.truetype(semibold, 17)
    body = ImageFont.truetype(regular, 18)
    time_head = ImageFont.truetype(semibold, 20)
    time_value = ImageFont.truetype(semibold, 32)
    green, blue, gray, white = "#34A853", "#4285F4", "#9AA0A6", "#F1F3F4"

    draw.line((0, 0, 0, DASH_H), fill=gray, width=2)
    draw.text((18, 18), "EXECUTE   real robot, closed loop", font=head,
              fill=blue if timestamp >= timings["policy_start"] else gray)

    def state(start: float | None, end: float | None) -> tuple[str, float]:
        if start is None or timestamp < start:
            return "pending", 0.0
        if end is None or timestamp < end:
            return "active", timestamp - start
        return "done", end - start

    rows = [
        ("Policy Initialization", timings["policy_start"], timings["policy_end"]),
        ("Push trajectory", timings["push_start"], timings["push_end"]),
        ("Grasp evaluation", timings["evaluation_start"], timings["evaluation_end"]),
        ("Grasp + retrieve", timings["grasp_start"], timings["grasp_end"]),
    ]
    for y, (label, start, end) in zip((67, 105, 143, 181), rows):
        phase_state, elapsed = state(start, end)
        color = {"pending": gray, "active": blue, "done": green}[phase_state]
        if phase_state == "pending":
            draw.ellipse((18, y + 5, 28, y + 15), outline=gray, width=2)
        else:
            draw.ellipse((18, y + 5, 28, y + 15), fill=color)
        draw.text((36, y), label, font=body, fill=color)
        value = "-" if phase_state == "pending" else f"{elapsed:.1f} s"
        value_w = draw.textlength(value, font=body)
        draw.text((EXECUTE_OUT_W - value_w - 18, y), value,
                  font=body, fill=color)

    total_x = EXECUTE_OUT_W
    draw.line((total_x, 0, total_x, DASH_H), fill=gray, width=2)
    draw.text((total_x + 22, 52), "Total Time:", font=time_head, fill=white)
    elapsed_total = max(0.0, timestamp - timings["total_time_offset"])
    draw.text((total_x + 22, 88), f"{elapsed_total:.1f} s",
              font=time_value, fill=white)
    return panel


def render_execution_and_time_video(path: Path, total: float,
                                    timings: dict) -> None:
    fps = 10
    width = EXECUTE_OUT_W + TOTAL_OUT_W
    command = [
        find_encoder(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{DASH_H}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_number in range(round(total * fps) + 1):
            timestamp = frame_number / fps
            frame = draw_execution_and_time_panel(timestamp, timings)
            process.stdin.write(frame.convert("RGB").tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def render_title_panel(path: Path, scene_name: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGBA", (TITLE_OUT_W, DASH_H), (32, 33, 36, 255))
    draw = ImageDraw.Draw(panel)
    font_path = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
    scene_font = ImageFont.truetype(font_path, 25)
    policy_text = "TRACE"
    size = 20
    policy_font = ImageFont.truetype(font_path, size)
    while draw.textlength(policy_text, font=policy_font) > TITLE_OUT_W - 36 and size > 14:
        size -= 1
        policy_font = ImageFont.truetype(font_path, size)
    closed_font = ImageFont.truetype(font_path, 19)
    draw.text((18, 22), scene_name, font=scene_font, fill="#F1F3F4")
    draw.text((18, 79), policy_text, font=policy_font, fill="#F1F3F4")
    draw.text((18, 118), "Closed-Loop", font=closed_font, fill="#F1F3F4")
    draw.line((TITLE_OUT_W - 1, 0, TITLE_OUT_W - 1, DASH_H), fill="#9AA0A6", width=2)
    panel.save(path)


def render_perception_panel(path: Path, *, complete: bool, seconds: float) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.new("RGBA", (PERCEPTION_OUT_W, DASH_H), (32, 33, 36, 255))
    draw = ImageDraw.Draw(panel)
    regular = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"
    semibold = "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf"
    head = ImageFont.truetype(semibold, 17)
    body = ImageFont.truetype(regular, 20)
    small = ImageFont.truetype(regular, 16)
    green, blue, gray = "#34A853", "#4285F4", "#9AA0A6"
    color = green if complete else blue
    draw.text((18, 18), "PERCEIVE   real camera", font=head,
              fill=green if complete else blue)
    draw.ellipse((18, 71, 29, 82), fill=color)
    draw.text((38, 64), "Segmentation", font=body, fill=color)
    draw.text((38, 101), "+ Pose estimation", font=body, fill=color)
    value = f"{seconds:.1f} s" if complete else "-"
    value_w = draw.textlength(value, font=small)
    draw.text((PERCEPTION_OUT_W - value_w - 18, 136), value, font=small,
              fill=color if complete else gray)
    draw.line((PERCEPTION_OUT_W - 1, 0, PERCEPTION_OUT_W - 1, DASH_H),
              fill=gray, width=2)
    panel.save(path)


def render_text_badge(path: Path, text: str, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Semibold.ttf", 13)
    badge = Image.new("RGBA", (width, height), (32, 33, 36, 224))
    draw = ImageDraw.Draw(badge)
    lines = text.split("\n")
    if len(lines) == 1:
        draw.text((10, 6), text, font=font, fill="#F1F3F4")
    else:
        draw.multiline_text((10, 3), text, font=font, fill="#F1F3F4",
                            spacing=0)
    badge.save(path)


def render_asset_atlas(path: Path, title: Path,
                       perception_active: Path, perception_done: Path,
                       speed: Path, d455_label: Path, obs_label: Path,
                       gn_label: Path, sim_label: Path) -> None:
    from PIL import Image

    atlas = Image.new("RGBA", (ATLAS_W, ATLAS_H), (0, 0, 0, 0))
    for source, xy in (
        (title, (0, 0)),
        (perception_active, (0, DASH_H)),
        (perception_done, (PERCEPTION_OUT_W, DASH_H)),
        (speed, (0, DASH_H * 2)),
        (d455_label, (110, DASH_H * 2)),
        (gn_label, (296, DASH_H * 2)),
        (sim_label, (452, DASH_H * 2)),
        (obs_label, (550, DASH_H * 2)),
    ):
        atlas.alpha_composite(Image.open(source).convert("RGBA"), dest=xy)
    atlas.save(path)


def render_speed_badge(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    badge = Image.new("RGBA", (104, 28), (32, 33, 36, 224))
    draw = ImageDraw.Draw(badge)
    font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Semibold.ttf", 13)
    draw.text((10, 6), "Speed: 1x", font=font, fill="#F1F3F4")
    badge.save(path)


def render_fitted_image(source: Path, output: Path, width: int, height: int) -> None:
    """Resize a static source once instead of asking FFmpeg to do it per frame."""
    from PIL import Image

    image = Image.open(source).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    canvas.save(output)


def build_filter(
    total: float,
    raw_d455_end: float,
    overlay_start: float,
    obs_hold: float,
    dashboard_hold: float,
    solve_start: float,
    gn_start: float,
    end_slide_start: float,
    webcam_advance: float,
    rollout_duration: float,
    solver_panel: bool,
) -> str:
    tail = max(0.0, total - raw_d455_end + 0.25)
    dashboard_tail = max(0.0, total - dashboard_hold + 0.25)
    obs_tail = max(0.0, total - obs_hold + 0.25)
    obs_inner = TILE_H
    obs_xpad = (TILE_W - obs_inner) // 2
    chains = [
        f"color=c=0x111820:s={CANVAS_W}x{CANVAS_H}:r=30:d={total:.6f}[bg]",
        (f"[1:v]crop=iw*0.75:ih:iw*0.25:0,hflip,vflip,"
         f"tpad=stop_mode=clone:stop_duration={tail:.6f},"
         f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         f"scale={MAIN_W}:{MAIN_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         "setsar=1,"
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
         f"{OBS_X}:{OBS_Y + OBS_LABEL_H},hflip,vflip,"
         f"trim=end={obs_hold:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={obs_tail:.6f},"
         f"scale={obs_inner}:{obs_inner}:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:{obs_xpad}:0:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x34A853:t=4[obsbase]"),
        (f"[4:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,format=rgba,"
         "split=8[atitle][apactive][apdone][aspeed]"
         "[ad455label][aobslabel][agnlabel][asimlabel]"
         ),
        f"[atitle]crop={TITLE_OUT_W}:{DASH_H}:0:0[titlepanel]",
        (f"[apactive]crop={PERCEPTION_OUT_W}:{DASH_H}:"
         "0:260[perceptionactive]"),
        (f"[apdone]crop={PERCEPTION_OUT_W}:{DASH_H}:"
         f"{PERCEPTION_OUT_W}:260[perceptiondone]"),
        f"[aspeed]crop={SPEED_BADGE_W}:{SPEED_BADGE_H}:0:520[speedbadge]",
        (f"[ad455label]crop={LABEL_D455_W}:{SPEED_BADGE_H}:"
         "110:520[label455]"),
        f"[aobslabel]crop={LABEL_OBS_W}:{TILE_LABEL_H}:550:520[labelobs]",
        f"[agnlabel]crop={LABEL_GN_W}:{TILE_LABEL_H}:296:520[labelgn]",
        f"[asimlabel]crop={LABEL_SIM_W}:{TILE_LABEL_H}:452:520[labelsim]",
    ]
    if solver_panel:
        chains.append(
            f"[5:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
            f"scale={SOLVE_OUT_W}:{DASH_H}:flags=lanczos,setsar=1[solverdynamic]"
        )
    else:
        chains.append("[5:v]nullsink")
    chains.append(
        f"[6:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
        f"scale={EXECUTE_OUT_W + TOTAL_OUT_W}:{DASH_H}:flags=lanczos,"
        "setsar=1[executiondynamic]"
    )
    chains += [
        f"color=c=0x202124:s={DASH_W}x{DASH_H}:r=30:d={total:.6f}[headerbg]",
        "[headerbg][titlepanel]overlay=0:0[h0]",
        f"[h0][perceptionactive]overlay={PERCEPTION_OUT_X}:0[h1]",
        (f"[h1][perceptiondone]overlay={PERCEPTION_OUT_X}:0:"
         f"enable='gte(t,{overlay_start:.6f})'[h2]"),
        f"[h2][solveoriginal]overlay={SOLVE_OUT_X}:0[h3]",
    ]
    if solver_panel:
        chains.append(
            f"[h3][solverdynamic]overlay={SOLVE_OUT_X}:0:eof_action=repeat[h4]"
        )
    else:
        chains.append("[h3]null[h4]")
    chains += [
        f"[h4][executiondynamic]overlay={EXECUTE_OUT_X}:0[h5]",
        f"[h5][progress]overlay=0:{DASH_H - 8}[dashboard]",
        "[speedbadge]split=2[badge455][badge415]",
        "[d455base][badge455]overlay=12:H-h-12[d455speed]",
        "[d455speed][label455]overlay=W-w-12:H-h-12[d455]",
        "[obsbase][labelobs]overlay=W-w-12:H-h-12[obs]",
        (f"[2:v]trim=start={webcam_advance:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={tail + webcam_advance + 2.0:.6f},"
         f"trim=duration={total:.6f},"
         "crop=iw*0.75:ih:iw*0.05:0,"
         f"scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x4285F4:t=4[webcam]"),
        "[webcam][badge415]overlay=12:H-h-12[d415]",
        (f"[7:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xFB8C00:t=4[gnselected]"),
        "[gnselected][labelgn]overlay=W-w-12:H-h-12[gn]",
        (f"[8:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,setsar=1[gnslide]"),
        (f"[3:v]trim=duration={rollout_duration:.6f},setpts=PTS-STARTPTS,"
         "split=3[simbase][siminner][simlabels]"),
        (f"[siminner]crop={SIM_INNER_SIZE}:{SIM_INNER_SIZE}:"
         f"{SIM_INNER_X}:{SIM_INNER_Y},"
         f"drawbox=x=0:y=0:w={SIM_INNER_SIZE}:h={SIM_WORKSPACE_LABEL_H}:"
         "color=black:t=fill,transpose=2[simrotated]"),
        (f"[simbase][simrotated]overlay={SIM_INNER_X}:{SIM_INNER_Y}"
         "[simscene]"),
        (f"[simlabels]crop={SIM_INNER_SIZE}:{SIM_WORKSPACE_LABEL_H}:"
         f"{SIM_INNER_X}:{SIM_INNER_Y}[simlabelstrip]"),
        (f"[simscene][simlabelstrip]overlay={SIM_INNER_X}:{SIM_INNER_Y},"
         f"scale={SIM_CONTENT_SIZE}:{SIM_CONTENT_SIZE}:flags=lanczos,setsar=1,"
         f"setpts=PTS+{solve_start:.6f}/TB[simvideo]"),
        (f"color=c=black:s={SIM_CONTENT_SIZE}x{SIM_CONTENT_SIZE}:r=30:d={total:.6f},"
         "drawtext=fontfile=/usr/share/fonts/truetype/lato/Lato-Semibold.ttf:"
         "text='Loading Isaac Gym...':fontcolor=0xDADCE0:fontsize=20:"
         "x=(w-text_w)/2:y=(h-text_h)/2[simblank]"),
        ("[simblank][simvideo]overlay=0:0:eof_action=repeat:shortest=0,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xFBBC04:t=4[simbase]"),
        "[simbase][labelsim]overlay=W-w-12:H-h-12[sim]",
        f"[bg][dashboard]overlay={MARGIN}:{DASH_Y}[c0]",
        f"[c0][d455]overlay={MAIN_X}:{MAIN_Y}[c1]",
        (f"[c1][obs]overlay={TILE_X1}:{TILE_Y1}:"
         f"enable='gte(t,{overlay_start:.6f})'[c2]"),
        f"[c2][gn]overlay={TILE_X2}:{TILE_Y1}:enable='gte(t,{gn_start:.6f})'[c3]",
        f"[c3][d415]overlay={TILE_X1}:{TILE_Y2}[c4]",
        f"[c4][sim]overlay={TILE_X2}:{TILE_Y2}[c5]",
        (f"[c5][gnslide]overlay=0:0:"
         f"enable='gte(t,{end_slide_start:.6f})'[c6]"),
    ]
    chains.append("[c6]format=yuv420p[out]")
    return ";".join(chains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="trial directory containing the three recordings")
    parser.add_argument("output", type=Path, help="new MP4 to create")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--limit", type=float, help="render only the first N seconds")
    args = parser.parse_args()

    trial = args.trial.resolve()
    annotated_d455 = trial / "d455_topdown.mp4"
    raw_d455 = trial / "d455_topdown_raw.mp4"
    raw_webcam = trial / "webcam_scene_raw.mp4"
    sim_candidates = sorted((trial / "sim").glob("*_closed-loop.mp4"))
    if not sim_candidates:
        raise SystemExit(f"No simulator video found under {trial / 'sim'}")
    sim = sim_candidates[0]
    for source in (annotated_d455, raw_d455, raw_webcam, sim):
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")

    phases = load_json(trial / "metadata" / "phases.json")
    timing = load_json(trial / "metadata" / "real_timing.json")
    sim_timing_candidates = sorted((trial / "sim").glob("*_sim_timing.json"))
    sim_timing = load_json(sim_timing_candidates[0]) if sim_timing_candidates else {}
    total = min(duration(annotated_d455), args.limit or float("inf"))
    sim_duration = duration(sim)
    raw_duration = duration(raw_d455)
    webcam_duration = duration(raw_webcam)
    raw_end = min(raw_duration, total)
    recorder = load_json(trial / "metadata" / "d455_topdown_rec.json")
    webcam_recorder = load_json(trial / "metadata" / "webcam_scene_rec.json")
    # phases.json starts before the first encoded camera frame. The recorder's
    # timestamp span includes that startup gap; video PTS does not.
    video_zero_offset = max(0.0, float(recorder.get("duration_s", raw_duration)) - raw_duration)
    webcam_zero_offset = max(
        0.0, float(webcam_recorder.get("duration_s", webcam_duration)) - webcam_duration)
    # Both recorders share t0_epoch, but their encoded files omit different
    # amounts of startup time. Advance the earlier webcam stream to the D455
    # video origin so visible robot motion lines up across the two panels.
    webcam_advance = max(0.0, video_zero_offset - webcam_zero_offset)

    def video_event(phase: str) -> float | None:
        value = event_offset(phases, phase)
        return max(0.0, value - video_zero_offset) if value is not None else None

    overlay_start = video_event("twin") or 0.0
    overlay_stop = video_event("home") or raw_end
    solve_start = video_event("solve") or 0.0
    gn_start = video_event("grasp") or video_event("no-grasp") or raw_end
    obs_hold = max(overlay_start, overlay_stop - 0.5)
    dashboard_hold = min(raw_end, gn_start + 1.2)
    gn_candidates = sorted((trial / "perception").glob("*_gn16.png"))
    if not gn_candidates:
        raise SystemExit(f"No grasp-network panel found under {trial / 'perception'}")
    gn_panel = gn_candidates[-1]
    feasible_grasp = video_event("grasp") is not None
    selected_gn_panel = gn_panel
    if feasible_grasp:
        prediction_candidates = sorted(
            (trial / "perception").glob("*_gn16_predictions.npz"))
        if not prediction_candidates:
            raise SystemExit(
                f"No grasp-network predictions found under {trial / 'perception'}")
        import numpy as np

        with np.load(prediction_candidates[-1], allow_pickle=False) as predictions:
            prediction_metadata = json.loads(str(predictions["metadata"].item()))
        selected_bin = int(prediction_metadata["grasp"]["rotation_idx"])
        selected_candidate = (
            trial / "perception" /
            f"{gn_panel.stem}_tiles/bin_{selected_bin:02d}.png"
        )
        if not selected_candidate.is_file():
            raise SystemExit(f"Missing selected grasp bin: {selected_candidate}")
        selected_gn_panel = selected_candidate
    done_event = video_event("done")
    end_slide_start = done_event if done_event is not None else max(gn_start, total - 4.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing output: {args.output}")

    breakdown = None
    if not {"twin-settle", "solve-export"} <= {
            event.get("phase") for event in phases.get("events", [])}:
        breakdown = legacy_solver_breakdown(phases, sim_timing)

    with tempfile.TemporaryDirectory(prefix="closed_loop_compose_") as temporary:
        panel_path = Path(temporary) / "solver_timing.mp4"
        title_path = Path(temporary) / "title.png"
        perception_active_path = Path(temporary) / "perception_active.png"
        perception_done_path = Path(temporary) / "perception_done.png"
        badge_path = Path(temporary) / "speed.png"
        d455_label_path = Path(temporary) / "d455_label.png"
        obs_label_path = Path(temporary) / "obs_label.png"
        gn_label_path = Path(temporary) / "gn_label.png"
        sim_label_path = Path(temporary) / "sim_label.png"
        selected_gn_render_path = Path(temporary) / "selected_gn.png"
        gn_slide_path = Path(temporary) / "gn_slide.png"
        atlas_path = Path(temporary) / "assets.png"
        execution_panel_path = Path(temporary) / "execution_timing.mp4"
        scene_match = __import__("re").search(r"scene\s*0*(\d+)", trial.parent.name,
                                               __import__("re").IGNORECASE)
        scene_name = f"Scene {int(scene_match.group(1)):02d}" if scene_match else "Scene"
        first_perceive = event_offset(phases, "perceive") or 0.0
        twin_event = event_offset(phases, "twin") or first_perceive
        perception_seconds = max(0.0, twin_event - first_perceive)
        render_title_panel(title_path, scene_name)
        render_perception_panel(perception_active_path, complete=False,
                                seconds=perception_seconds)
        render_perception_panel(perception_done_path, complete=True,
                                seconds=perception_seconds)
        render_speed_badge(badge_path)
        render_text_badge(d455_label_path, "Orthographic Camera", LABEL_D455_W,
                          SPEED_BADGE_H)
        render_text_badge(obs_label_path, "Occluded\nObservations", LABEL_OBS_W,
                          TILE_LABEL_H)
        render_text_badge(gn_label_path, "Grasp\nEvaluation", LABEL_GN_W,
                          TILE_LABEL_H)
        render_text_badge(sim_label_path, "Digital\nTwin", LABEL_SIM_W,
                          TILE_LABEL_H)
        render_fitted_image(selected_gn_panel, selected_gn_render_path,
                            TILE_W, TILE_H)
        render_fitted_image(gn_panel, gn_slide_path, CANVAS_W, CANVAS_H)
        rollout_duration = (breakdown["ppo_rollout_simulated_seconds"]
                            if breakdown else sim_duration)
        if breakdown:
            render_solver_panel_video(panel_path, breakdown, total,
                                      overlay_start, solve_start)
        execution_timings = {
            "policy_start": video_event("student-load"),
            "policy_end": video_event("student"),
            "push_start": video_event("student"),
            "push_end": video_event("home"),
            "evaluation_start": video_event("re-sense"),
            "evaluation_end": video_event("grasp") or video_event("no-grasp"),
            "grasp_start": video_event("grasp"),
            "grasp_end": video_event("done") if video_event("grasp") else None,
            # The source overlay begins its clock just under half a second after
            # the first encoded dashboard frame.
            "total_time_offset": 0.45,
        }
        render_execution_and_time_video(execution_panel_path, total,
                                        execution_timings)
        render_asset_atlas(atlas_path, title_path,
                           perception_active_path, perception_done_path,
                           badge_path, d455_label_path, obs_label_path,
                           gn_label_path, sim_label_path)
        command = [
            find_encoder(), "-hide_banner", "-loglevel", "warning", "-nostdin", "-stats", "-y",
            "-hwaccel", "none", "-i", str(annotated_d455),
            "-hwaccel", "none", "-i", str(raw_d455),
            "-hwaccel", "none", "-i", str(raw_webcam),
            "-hwaccel", "none", "-i", str(sim),
            "-loop", "1", "-framerate", "30", "-i", str(atlas_path),
        ]
        if breakdown:
            command += ["-hwaccel", "none", "-i", str(panel_path)]
        else:
            command += ["-loop", "1", "-framerate", "30", "-i", str(atlas_path)]
        command += ["-hwaccel", "none", "-i", str(execution_panel_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(selected_gn_render_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(gn_slide_path)]
        command += [
            "-filter_complex", build_filter(total, raw_end, overlay_start, obs_hold,
                                            dashboard_hold, solve_start, gn_start,
                                            end_slide_start, webcam_advance,
                                            rollout_duration,
                                            bool(breakdown)),
            "-map", "[out]", "-an", "-r", "30", "-t", f"{total:.6f}",
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        print("Creating", args.output)
        print(f"Duration {total:.2f} s; D455 {MAIN_W}x{MAIN_H}; canvas {CANVAS_W}x{CANVAS_H}")
        print(f"Webcam advance {webcam_advance:.3f} s from recorder clock alignment")
        subprocess.run(command, check=True)
    if breakdown:
        report = args.output.with_suffix(".solver_timing.json")
        report.write_text(json.dumps(breakdown, indent=2) + "\n")
        print("Solver timing ->", report)


if __name__ == "__main__":
    raise SystemExit(
        "compose_closed_loop.py is the shared layout module, not a composer entry point.\n"
        "Its renderer predates the measured exposure transfer, so it would write an\n"
        "uncorrected video. Use one of:\n"
        "  compose_closed_loop_memory.py   TRACE\n"
        "  compose_teacher_closed_loop.py  Online Teacher\n"
        "  compose_open_loop.py            Teacher Replay\n"
        "  compose_spiral_closed_loop.py   Spiral and PMBS")
