#!/usr/bin/env python3
"""Compose one open-loop trial into the locked multi-panel review layout.

The D455 recording has a burned-in source dashboard, so the camera region is
cropped out before it is rotated. Source recordings are read only; the output
is always a new file.
"""

from __future__ import annotations

import argparse
from camera_sources import third_person
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import exposure


CANVAS_W = 1920
CANVAS_H = 1080
MARGIN = 10
DASH_W = 1900
DASH_H = 260
DASH_Y = 10
CONTENT_Y = 300
MAIN_X = 30
MAIN_Y = 413
MAIN_W = 800
MAIN_H = 494
TILE_W = 500
TILE_H = 350
TILE_X1 = 860
TILE_X2 = 1380
TILE_Y1 = 300
TILE_Y2 = 670
SIM_CONTENT_SIZE = 350
ENDCARD_X = 10
ENDCARD_Y = CONTENT_Y
ENDCARD_W = 1900
ENDCARD_H = CANVAS_H - ENDCARD_Y - 10

# Coordinates in the original 1280 x 720 annotated D455 recording.
STANDARD_BAND_H = 152
OPEN_CAMERA_Y = 126
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
SIM_INNER_OUT = 244
SIM_INNER_OUT_X = 53
SIM_WORKSPACE_LABEL_OUT_H = 25
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
TRACE_PINK = (255, 64, 160)
TRACE_Z = 0.020
RAW_W, RAW_H, RAW_CROP_X = 1280, 720, 320
SIM_FRAME_LO, SIM_FRAME_SPAN = 570, 2700
# D455 panel window in the raw (header-free) recording: 570 rows, shifted 9%
# toward the far mat edge (the bottom of the panel after the 180 degree turn).
D455_VIEW_H = 570
D455_VIEW_Y0 = RAW_H - D455_VIEW_H - round(0.09 * D455_VIEW_H)
D455_VIEW_CROP = (f"crop=iw:{D455_VIEW_H}:0:{D455_VIEW_Y0},"
                  "crop=iw*0.75:ih:iw*0.25:0,hflip,vflip")
# Webcam crop width (fraction of the frame, from x = 5%). The annotated webcam's
# burned-in dashboard starts near x = 74%, so the wide grasp crop needs the raw file.
# First annotated-D455 row that is always camera: the burned-in band grows to
# row 145 during execution, with its blue separator on rows 146-149.
ANNOTATED_D455_FIRST_ROW = 150
WEBCAM_NO_GRASP_FRACTION = 0.6354
WEBCAM_GRASP_FRACTION = 0.7135
WEBCAM_ANNOTATED_MAX_FRACTION = 0.68
# Twin replays in preference order; closed-loop is skipped when its replay
# never reached graspable (sim_timing per_mode_time_to_graspable is null).
SIM_MODES = ("closed-loop", "joint", "cartesian_feedforward", "cartesian")


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


def open_loop_solver_breakdown(phases: dict, sim_duration: float) -> dict | None:
    """Split the recorded solve wall interval around the visible rollout."""
    twin = event_offset(phases, "twin")
    solve = event_offset(phases, "solve")
    execution = event_offset(phases, "execution")
    if None in (twin, solve, execution):
        return None
    post = max(0.0, execution - solve - sim_duration)
    return {
        "schema_version": 1,
        "source_schema": "open-loop phase markers plus simulator video PTS",
        "isaac_gym_and_teacher_load_wall_seconds": solve - twin,
        "ppo_rollout_simulated_seconds": sim_duration,
        "legacy_combined_solve_wall_seconds": execution - solve,
        "solver_accounting_remainder_seconds": post,
        "solver_accounting_remainder_contains": [
            "rollout wall-time minus simulator PTS", "trajectory extraction",
            "result logging", "hardware-plan handoff"],
        "caveat": ("Plan Post-Processing reconciles the recorded solve-to-execution "
                   "wall interval after subtracting the simulator video's true PTS."),
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
    green, blue, gray, white, red = "#34A853", "#4285F4", "#9AA0A6", "#F1F3F4", "#EA4335"
    status = ImageFont.truetype(regular, 14)
    failure = timings.get("failure")
    failed = failure is not None and timestamp >= failure["at"]

    draw.line((0, 0, 0, DASH_H), fill=gray, width=2)
    draw.text((18, 18), "EXECUTE   real robot, open loop", font=head,
              fill=red if failed else (blue if timestamp >= timings["policy_start"] else gray))

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
        if failed and label == "Grasp + retrieve":
            # Target not graspable (or OOW) after the replay: no grasp is attempted.
            draw.ellipse((18, y + 5, 28, y + 15), fill=red)
            draw.text((36, y), label, font=body, fill=red)
            value_w = draw.textlength(failure["label"], font=body)
            draw.text((EXECUTE_OUT_W - value_w - 18, y), failure["label"], font=body, fill=red)
            draw.text((36, y + 27), failure["reason"], font=status, fill=red)
            continue
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
              font=time_value, fill=red if failed else white)
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
    policy_text = "Teacher Replay"
    size = 20
    policy_font = ImageFont.truetype(font_path, size)
    while draw.textlength(policy_text, font=policy_font) > TITLE_OUT_W - 36 and size > 14:
        size -= 1
        policy_font = ImageFont.truetype(font_path, size)
    closed_font = ImageFont.truetype(font_path, 19)
    draw.text((18, 22), scene_name, font=scene_font, fill="#F1F3F4")
    draw.text((18, 79), policy_text, font=policy_font, fill="#F1F3F4")
    draw.text((18, 118), "Open-Loop", font=closed_font, fill="#F1F3F4")
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


def render_source_tile(path: Path, source: Path, border: str, *, rotate: bool) -> None:
    """Pre-scale a still diagnostic once instead of resizing it every frame."""
    from PIL import Image, ImageDraw

    content = Image.open(source).convert("RGB")
    if rotate:
        content = content.rotate(180)
    content.thumbnail((TILE_W, TILE_H), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (TILE_W, TILE_H), "black")
    tile.paste(content, ((TILE_W - content.width) // 2,
                         (TILE_H - content.height) // 2))
    ImageDraw.Draw(tile).rectangle((1, 1, TILE_W - 2, TILE_H - 2),
                                   outline=border, width=4)
    tile.save(path)


def render_grasp_endcard(path: Path, source: Path) -> None:
    """Fit the complete grasp-orientation grid into the final content area."""
    from PIL import Image, ImageDraw

    content = Image.open(source).convert("RGB")
    content.thumbnail((ENDCARD_W, ENDCARD_H), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (ENDCARD_W, ENDCARD_H), "black")
    card.paste(content, ((ENDCARD_W - content.width) // 2,
                         (ENDCARD_H - content.height) // 2))
    ImageDraw.Draw(card).rectangle((1, 1, ENDCARD_W - 2, ENDCARD_H - 2),
                                   outline="#FB8C00", width=4)
    card.save(path)


ISAACGYMENVS_ROOT = Path(__file__).resolve().parents[2] / "sim"


def regenerate_grasp_panel(panel: Path, output: Path) -> Path:
    """Redraw the GN grid from its saved predictions with the current renderer.

    The recorded PNG may carry a jaw footprint on a sub-threshold network best;
    the current renderer draws it only for an executable grasp. Falls back to the
    recorded PNG when the predictions or the renderer are unavailable.
    """
    record = panel.with_name(panel.stem + "_predictions.npz")
    if not record.is_file():
        return panel
    result = subprocess.run(
        [sys.executable, "-m", "isaacgymenvs.open_loop.render_saved_grasp",
         str(record), "--output", str(output)],
        cwd=ISAACGYMENVS_ROOT, capture_output=True, text=True)
    if result.returncode != 0 or not output.is_file():
        print(f"grasp panel: kept recorded {panel.name} (regeneration failed: "
              f"{(result.stderr or result.stdout).strip().splitlines()[-1:]})")
        return panel
    print(f"grasp panel: regenerated from {record.name}")
    return output


def rollout_trace(trial: Path) -> list[list[float]]:
    trajectory = load_json(trial / "trajectory.json")
    points = [trajectory["start_eef"][:2]]
    points.extend(step["eef"][:2] for step in trajectory["dense"])
    return points


def camera_projection(trial: Path):
    """Map sim-frame EEF positions into the cropped, rotated D455 panel."""
    import numpy as np

    records = sorted(trial.glob("*_gn16_predictions.npz"))
    if not records:
        raise SystemExit(f"No grasp-network camera calibration under {trial}")
    with np.load(records[-1], allow_pickle=False) as data:
        base_to_camera = np.linalg.inv(data["camera_to_base"])
    intrinsics = load_json(trial / "d455_topdown_dump0_K.json")

    def to_panel(x_sim: float, y_sim: float) -> tuple[float, float]:
        point = base_to_camera @ np.array([y_sim, -x_sim, TRACE_Z, 1.0])
        u = point[0] * intrinsics["fx"] / point[2] + intrinsics["cx"]
        v = point[1] * intrinsics["fy"] / point[2] + intrinsics["cy"]
        # D455_VIEW_CROP window, right 75%, rotated 180 degrees, scaled to the panel.
        x = (RAW_W - 1 - u) * MAIN_W / (RAW_W - RAW_CROP_X)
        y = (D455_VIEW_Y0 + D455_VIEW_H - 1 - v) * MAIN_H / D455_VIEW_H
        return x, y

    return to_panel


def render_d455_trace(path: Path, trace: list[list[float]], to_panel) -> None:
    from PIL import Image, ImageDraw

    scale = 2
    layer = Image.new("RGBA", (MAIN_W * scale, MAIN_H * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    points = [tuple(value * scale for value in to_panel(*point)) for point in trace]
    if len(points) > 1:
        draw.line(points, fill=TRACE_PINK + (80,), width=14 * scale, joint="curve")
        draw.line(points, fill=TRACE_PINK + (255,), width=5 * scale, joint="curve")
    for x, y in points:
        radius = 4 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=TRACE_PINK + (255,), outline=(255, 255, 255, 255), width=scale)
    layer.resize((MAIN_W, MAIN_H), Image.Resampling.LANCZOS).save(path)


def render_sim_trace(path: Path, trace: list[list[float]]) -> None:
    """Map the complete teacher EEF rollout through the simulator-tile transform."""
    from PIL import Image, ImageDraw

    scale = 2
    layer = Image.new("RGBA", (TILE_W * scale, TILE_H * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    content_scale = SIM_CONTENT_SIZE / 3840.0
    x_pad = (TILE_W - SIM_CONTENT_SIZE) // 2

    def to_tile(x_sim: float, y_sim: float) -> tuple[float, float]:
        u = (x_sim - 0.276) / 0.448
        v = (y_sim + 0.224) / 0.448
        col = SIM_FRAME_LO + v * SIM_FRAME_SPAN
        row = SIM_FRAME_LO + u * SIM_FRAME_SPAN
        inner_col, inner_row = col - SIM_INNER_X, row - SIM_INNER_Y
        final_x = SIM_INNER_X + inner_row
        final_y = SIM_INNER_Y + (SIM_INNER_SIZE - 1 - inner_col)
        return ((x_pad + final_x * content_scale) * scale,
                final_y * content_scale * scale)

    points = [to_tile(*point) for point in trace]
    if len(points) > 1:
        draw.line(points, fill=TRACE_PINK + (90,), width=11 * scale, joint="curve")
        draw.line(points, fill=TRACE_PINK + (255,), width=4 * scale, joint="curve")
    for x, y in points:
        radius = 3 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=TRACE_PINK + (255,), outline=(255, 255, 255, 255), width=scale)
    layer.resize((TILE_W, TILE_H), Image.Resampling.LANCZOS).save(path)


def build_filter(
    total: float,
    raw_d455_end: float,
    overlay_start: float,
    obs_start: float,
    dashboard_hold: float,
    solve_start: float,
    gn_start: float,
    rollout_duration: float,
    trace_start: float,
    sim_trace_start: float,
    solver_panel: bool,
    webcam_crop_fraction: float,
    d455_tone: tuple[float, float] | str = (1.0, 1.0),
    webcam_tone: tuple[float, float] | str = (1.0, 1.0),
) -> str:
    def tone(setting: tuple[float, float] | str) -> str:
        """Return a measured transfer chain or the legacy fixed-gamma chain."""
        if isinstance(setting, str):
            return setting
        gamma, saturation = setting
        if gamma == 1.0 and saturation == 1.0:
            return ""
        g = 1.0 / gamma
        return (f"format=gbrp,lutrgb=r=gammaval({g:.4f}):g=gammaval({g:.4f}):b=gammaval({g:.4f}),"
                f"format=yuv420p,eq=saturation={saturation},")

    tail = max(0.0, total - raw_d455_end + 0.25)
    dashboard_tail = max(0.0, total - dashboard_hold + 0.25)
    chains = [
        f"color=c=0x111820:s={CANVAS_W}x{CANVAS_H}:r=30:d={total:.6f}[bg]",
        (f"[1:v]trim=end={raw_d455_end:.6f},setpts=PTS-STARTPTS,"
         f"{D455_VIEW_CROP},"
         f"{tone(d455_tone)}"
         f"tpad=stop_mode=clone:stop_duration={tail:.6f},"
         f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         f"scale={MAIN_W}:{MAIN_H}:flags=lanczos,"
         "setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xE91E63:t=6[d455raw]"),
        "[10:v]format=rgba[d455trace]",
        (f"[d455raw][d455trace]overlay=0:0:format=auto:"
         f"enable='gte(t,{trace_start:.6f})'[d455base]"),
        (f"[0:v]trim=end={dashboard_hold:.6f},"
         f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={dashboard_tail:.6f}"
         "[headerheld]"),
        (f"[headerheld]crop=1280:5:0:{OPEN_CAMERA_Y - 5},"
         f"scale={DASH_W}:8:flags=lanczos,"
         "setsar=1[progress]"),
        (f"[7:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         "setsar=1[obsbase]"),
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
    ]
    if solver_panel:
        chains.append(
            f"[h2][solverdynamic]overlay={SOLVE_OUT_X}:0:eof_action=repeat[h4]"
        )
    else:
        chains.append("[h2]null[h4]")
    chains += [
        f"[h4][executiondynamic]overlay={EXECUTE_OUT_X}:0[h5]",
        f"[h5][progress]overlay=0:{DASH_H - 8}[dashboard]",
        "[speedbadge]split=2[badge455][badge415]",
        "[d455base][badge455]overlay=12:H-h-12[d455speed]",
        "[d455speed][label455]overlay=W-w-12:H-h-12[d455]",
        "[obsbase][labelobs]overlay=W-w-12:H-h-12[obs]",
        (f"[2:v]trim=end={raw_d455_end:.6f},setpts=PTS-STARTPTS,"
         f"tpad=stop_mode=clone:stop_duration={tail + 2.0:.6f},"
         f"trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         f"crop=iw*{webcam_crop_fraction:.4f}:ih:iw*0.05:0,"
         f"{tone(webcam_tone)}"
         f"scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0x4285F4:t=4[webcam]"),
        "[webcam][badge415]overlay=12:H-h-12[d415]",
        (f"[8:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         "setsar=1[gnbase]"),
        "[gnbase][labelgn]overlay=W-w-12:H-h-12[gn]",
        (f"[9:v]trim=duration={total:.6f},setpts=PTS-STARTPTS,"
         "setsar=1[gnendcard]"),
        (f"[3:v]trim=duration={rollout_duration:.6f},setpts=PTS-STARTPTS,"
         "split=3[simbase][siminner][simlabels]"),
        (f"[simbase]scale={SIM_CONTENT_SIZE}:{SIM_CONTENT_SIZE}:flags=lanczos"
         "[simsmallbase]"),
        (f"[siminner]crop={SIM_INNER_SIZE}:{SIM_INNER_SIZE}:"
         f"{SIM_INNER_X}:{SIM_INNER_Y},scale={SIM_INNER_OUT}:{SIM_INNER_OUT}:"
         "flags=lanczos,"
         f"drawbox=x=0:y=0:w={SIM_INNER_OUT}:h={SIM_WORKSPACE_LABEL_OUT_H}:"
         "color=black:t=fill,transpose=2[simrotated]"),
        (f"[simsmallbase][simrotated]overlay={SIM_INNER_OUT_X}:"
         f"{SIM_INNER_OUT_X}"
         "[simscene]"),
        (f"[simlabels]crop={SIM_INNER_SIZE}:{SIM_WORKSPACE_LABEL_H}:"
         f"{SIM_INNER_X}:{SIM_INNER_Y},scale={SIM_INNER_OUT}:"
         f"{SIM_WORKSPACE_LABEL_OUT_H}:flags=lanczos[simlabelstrip]"),
        (f"[simscene][simlabelstrip]overlay={SIM_INNER_OUT_X}:"
         f"{SIM_INNER_OUT_X},setsar=1,"
         f"setpts=PTS+{solve_start:.6f}/TB[simvideo]"),
        (f"color=c=black:s={SIM_CONTENT_SIZE}x{SIM_CONTENT_SIZE}:r=30:d={total:.6f},"
         "drawtext=fontfile=/usr/share/fonts/truetype/lato/Lato-Semibold.ttf:"
         "text='Loading Isaac Gym...':fontcolor=0xDADCE0:fontsize=20:"
         "x=(w-text_w)/2:y=(h-text_h)/2[simblank]"),
        ("[simblank][simvideo]overlay=0:0:eof_action=repeat:shortest=0,"
         f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
         "drawbox=x=0:y=0:w=iw:h=ih:color=0xFBBC04:t=4[simpadded]"),
        "[11:v]format=rgba[simtrace]",
        (f"[simpadded][simtrace]overlay=0:0:format=auto:"
         f"enable='gte(t,{sim_trace_start:.6f})'[simtraced]"),
        "[simtraced][labelsim]overlay=W-w-12:H-h-12[sim]",
        f"[bg][dashboard]overlay={MARGIN}:{DASH_Y}[c0]",
        f"[c0][d455]overlay={MAIN_X}:{MAIN_Y}[c1]",
        (f"[c1][obs]overlay={TILE_X1}:{TILE_Y1}:"
         f"enable='gte(t,{obs_start:.6f})'[c2]"),
        f"[c2][gn]overlay={TILE_X2}:{TILE_Y1}:enable='gte(t,{gn_start:.6f})'[c3]",
        f"[c3][d415]overlay={TILE_X1}:{TILE_Y2}[c4]",
        f"[c4][sim]overlay={TILE_X2}:{TILE_Y2}[c5]",
        (f"[c5][gnendcard]overlay={ENDCARD_X}:{ENDCARD_Y}:"
         f"enable='gte(t,{gn_start:.6f})'[c6]"),
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
    parser.add_argument("--exposure", choices=("measured", "fixed"), default="measured",
                        help="measured: exposure.video_transfer; fixed: legacy constant gamma")
    parser.add_argument("--d455-gamma", type=float, default=1.8,
                        help="brighten the top-down D455 view (1 = unchanged)")
    parser.add_argument("--d455-saturation", type=float, default=1.08)
    parser.add_argument("--webcam-gamma", type=float, default=1.3,
                        help="brighten the webcam view (1 = unchanged)")
    parser.add_argument("--webcam-saturation", type=float, default=1.04)
    args = parser.parse_args()

    trial = args.trial.resolve()
    annotated_d455 = trial / "d455_topdown.mp4"
    # The raw recording is frame-aligned with the annotated one but has no
    # burned-in header over the far mat edge.
    global D455_VIEW_Y0, D455_VIEW_CROP
    raw_d455 = trial / "d455_topdown_raw.mp4"
    if not raw_d455.is_file():
        raw_d455 = annotated_d455
        # Without the raw file, keep the window below the burned-in header.
        if D455_VIEW_Y0 < ANNOTATED_D455_FIRST_ROW:
            D455_VIEW_CROP = D455_VIEW_CROP.replace(
                f":0:{D455_VIEW_Y0},", f":0:{ANNOTATED_D455_FIRST_ROW},", 1)
            D455_VIEW_Y0 = ANNOTATED_D455_FIRST_ROW
        print(f"no d455_topdown_raw.mp4: annotated D455, window from row {D455_VIEW_Y0}")
    # Same for the webcam: its burned-in dashboard reaches into the grasp crop.
    webcam_max_fraction = 1.0
    if any((trial / f"{camera}_raw.mp4").is_file() for camera in ("webcam_scene", "d415_scene")):
        raw_webcam, _ = third_person(trial)
    else:
        raw_webcam, _ = third_person(trial, raw=False)
        webcam_max_fraction = WEBCAM_ANNOTATED_MAX_FRACTION
        print("no raw third-person video: annotated webcam, crop kept clear of its dashboard")
    sim_timing_candidates = sorted((trial / "sim").glob("*_sim_timing.json"))
    sim_timing = load_json(sim_timing_candidates[0]) if sim_timing_candidates else {}
    reached = (sim_timing.get("per_mode_time_to_graspable")
               or sim_timing.get("per_mode_replay_compute_wall_seconds_to_graspable") or {})
    sim_videos = {mode: next(iter(sorted((trial / "sim").glob(f"*_{mode}.mp4"))), None)
                  for mode in SIM_MODES}
    solved_modes = [m for m in SIM_MODES if sim_videos[m] and reached.get(m) is not None]
    sim_mode = (solved_modes or [m for m in SIM_MODES if sim_videos[m]] or [None])[0]
    if sim_mode is None:
        raise SystemExit(f"No simulator video found under {trial / 'sim'}")
    sim = sim_videos[sim_mode]
    print(f"twin replay: {sim.name}")
    for source in (annotated_d455, raw_d455, raw_webcam, sim):
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")

    phases = load_json(trial / "phases.json")
    timing = load_json(trial / "real_timing.json")
    total = min(duration(annotated_d455), args.limit or float("inf"))
    sim_duration = duration(sim)
    raw_duration = duration(raw_d455)
    # The archived annotated camera videos append a full-frame GN end card.
    # Freeze the live camera views shortly before that transition; the GN panel
    # is already presented in its own tile.
    raw_end = min(max(0.0, (event_offset(phases, "done") or raw_duration) - 1.5),
                  total, raw_duration)

    def video_event(phase: str) -> float | None:
        return event_offset(phases, phase)

    overlay_start = video_event("twin") or 0.0
    solve_start = video_event("solve") or 0.0
    obs_start = video_event("grasp") or video_event("no-grasp") or raw_end
    gn_start = obs_start
    dashboard_hold = raw_end
    obs_candidates = sorted(trial.glob("*_heightmap.png"))
    if not obs_candidates:
        raise SystemExit(f"No post-push observation panel found under {trial}")
    obs_panel = obs_candidates[-1]
    gn_candidates = sorted(trial.glob("*_gn16.png"))
    if not gn_candidates:
        raise SystemExit(f"No grasp-network panel found under {trial}")
    gn_panel = gn_candidates[-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing output: {args.output}")

    webcam_fraction = min(WEBCAM_NO_GRASP_FRACTION if video_event("no-grasp")
                          else WEBCAM_GRASP_FRACTION, webcam_max_fraction)
    d455_setting: tuple[float, float] | str = (args.d455_gamma, args.d455_saturation)
    webcam_setting: tuple[float, float] | str = (args.webcam_gamma, args.webcam_saturation)
    if args.exposure == "measured":
        d455_setting, d455_meta = exposure.video_transfer(
            raw_d455, "d455",
            crop=D455_VIEW_CROP,
            saturation=args.d455_saturation)
        webcam_setting, webcam_meta = exposure.video_transfer(
            raw_webcam, "webcam",
            crop=f"crop=iw*{webcam_fraction:.4f}:ih:iw*0.05:0",
            saturation=args.webcam_saturation)
        exposure.write_metadata(args.output.with_suffix(".exposure.json"),
                                {"d455": d455_meta, "webcam": webcam_meta})
        print(f"exposure: D455 gain {d455_meta['brightness_gain']:.3f}, "
              f"webcam gain {webcam_meta['brightness_gain']:.3f}")

    breakdown = open_loop_solver_breakdown(phases, sim_duration)

    with tempfile.TemporaryDirectory(prefix="open_loop_compose_") as temporary:
        panel_path = Path(temporary) / "solver_timing.mp4"
        title_path = Path(temporary) / "title.png"
        perception_active_path = Path(temporary) / "perception_active.png"
        perception_done_path = Path(temporary) / "perception_done.png"
        badge_path = Path(temporary) / "speed.png"
        d455_label_path = Path(temporary) / "d455_label.png"
        obs_label_path = Path(temporary) / "obs_label.png"
        gn_label_path = Path(temporary) / "gn_label.png"
        sim_label_path = Path(temporary) / "sim_label.png"
        atlas_path = Path(temporary) / "assets.png"
        execution_panel_path = Path(temporary) / "execution_timing.mp4"
        obs_tile_path = Path(temporary) / "observation_tile.png"
        gn_tile_path = Path(temporary) / "grasp_tile.png"
        gn_endcard_path = Path(temporary) / "grasp_endcard.png"
        d455_trace_path = Path(temporary) / "d455_trace.png"
        sim_trace_path = Path(temporary) / "sim_trace.png"
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
        render_source_tile(obs_tile_path, obs_panel, "#34A853", rotate=True)
        gn_panel = regenerate_grasp_panel(gn_panel, Path(temporary) / "grasp_panel.png")
        render_source_tile(gn_tile_path, gn_panel, "#FB8C00", rotate=False)
        render_grasp_endcard(gn_endcard_path, gn_panel)
        trace = rollout_trace(trial)
        render_d455_trace(d455_trace_path, trace, camera_projection(trial))
        render_sim_trace(sim_trace_path, trace)
        trace_start = video_event("solve-done") or (solve_start + sim_duration)
        motion_seconds = float(sim_timing.get("trajectory_simulated_seconds") or
                               (float(sim_timing.get("num_steps") or 0) /
                                float(sim_timing.get("control_hz") or 15.06)) or
                               sim_duration)
        sim_trace_start = solve_start + min(sim_duration, motion_seconds)
        rollout_duration = (breakdown["ppo_rollout_simulated_seconds"]
                            if breakdown else sim_duration)
        if breakdown:
            render_solver_panel_video(panel_path, breakdown, total,
                                      overlay_start, solve_start)
        execution_timings = {
            "policy_start": video_event("execution"),
            "policy_end": video_event("approach"),
            "push_start": video_event("approach"),
            "push_end": video_event("home"),
            "evaluation_start": video_event("re-sense"),
            "evaluation_end": video_event("grasp") or video_event("no-grasp"),
            "grasp_start": video_event("grasp"),
            "grasp_end": video_event("done") if video_event("grasp") else None,
            # The source overlay begins its clock just under half a second after
            # the first encoded dashboard frame.
            "total_time_offset": 0.20,
            "failure": None,
        }
        no_grasp = video_event("no-grasp")
        if no_grasp is not None:
            threshold = timing.get("stock_gn_threshold", 0.70)
            q = (timing.get("real_grasp") or {}).get("q")
            if timing.get("oow_failure"):
                reason = {"off_mat": "part of an object left the black mat",
                          "object_missing": "an object is no longer on the mat"}.get(
                              timing.get("stop_reason"), "an object left the workspace")
                execution_timings["failure"] = {"at": no_grasp, "label": "OOW failure",
                                                "reason": reason}
            else:
                reason = (f"grasp score {q:.2f} < {threshold:.2f}, no grasp attempted"
                          if q is not None else "no grasp attempted")
                execution_timings["failure"] = {"at": no_grasp, "label": "Not graspable",
                                                "reason": reason}
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
        command += ["-loop", "1", "-framerate", "30", "-i", str(obs_tile_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(gn_tile_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(gn_endcard_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(d455_trace_path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(sim_trace_path)]
        command += [
            "-filter_complex", build_filter(total, raw_end, overlay_start, obs_start,
                                            dashboard_hold, solve_start, gn_start,
                                            rollout_duration,
                                            trace_start, sim_trace_start,
                                            bool(breakdown),
                                            webcam_fraction,
                                            d455_setting,
                                            webcam_setting),
            "-map", "[out]", "-an", "-r", "30", "-t", f"{total:.6f}",
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        print("Creating", args.output)
        print(f"Duration {total:.2f} s; D455 {MAIN_W}x{MAIN_H}; canvas {CANVAS_W}x{CANVAS_H}")
        subprocess.run(command, check=True)
    if breakdown:
        report = args.output.with_suffix(".solver_timing.json")
        report.write_text(json.dumps(breakdown, indent=2) + "\n")
        print("Solver timing ->", report)


if __name__ == "__main__":
    main()
