"""Standalone RealSense recorder for real-robot runs.

Runs as its OWN PROCESS so that grabbing frames never adds latency to
execute_trajectory: the executor touches control files, it never blocks on a
camera. Frames are spooled to disk as they arrive and encoded to mp4 only at
stop, so encoding cannot steal time from the robot either.

It also solves a hard constraint: a RealSense device can be opened by exactly
one process. The perception camera (D455) therefore cannot be recorded AND
opened by perceive_scene at the same time. This recorder owns the device and
serves the re-sense: touch <ctrl>.dump and it writes color/depth/intrinsics
next to the video for whoever needs a frame.

  # third-person view of the robot
  python record_cameras.py --serial "$TRACE_D415_SERIAL" --name d415_scene --out runs/real1
  # top-down perception camera, also serving frames to the re-sense
  python record_cameras.py --serial "$TRACE_D455_SERIAL" --name d455_topdown --out runs/real1 --depth

Control files inside <out>/<name>.ctrl/:
  ready  written by the recorder once frames are flowing
  dump   touch it -> recorder writes dump_<n>_color.png/_depth.npy/_K.json
  stop   touch it -> recorder finishes, encodes, exits
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path
import time

import numpy as np
import cv2


def load_phases(out_dir):
    """Timestamped phase events from execute_trajectory, if it wrote any."""
    p = os.path.join(out_dir, "phases.json")
    if not os.path.exists(p):
        p = os.path.join(out_dir, "metadata", "phases.json")
    if not os.path.exists(p):
        return []
    try:
        return json.load(open(p)).get("events", [])
    except (OSError, ValueError):
        return []


# The pipeline as an audience sees it. Several logged phases collapse into one
# step: approach/push/home are all "the robot executing the plan".
# The pipeline as an audience sees it, in three groups that mirror the method:
# perceive once, solve entirely in the twin, replay open-loop on the robot.
# Simulator startup and the policy are separate rows on purpose -- the first is
# fixed process overhead, the second is the actual computation, and merged the
# interesting number hides inside the boring one.
LEGACY_GROUPS = [
    ("PERCEIVE   real camera", [
        ("Segmentation + pose estimation", ("perceive",))]),
    ("SOLVE   digital twin", [
        ("Simulator load",                 ("twin",)),
        ("PPO policy rollout",             ("solve",))]),
    ("EXECUTE   real robot, open loop", [
        ("Controller load",                 ("student-load", "spiral-load")),
        # "student": the closed-loop driver (open_loop/run_student.py) marks one
        # event per executed primitive under this phase.
        ("Push trajectory",                ("approach", "push", "student", "spiral",
                                             "spiral-retract")),
        ("Grasp evaluation",               ("home", "re-sense")),
        ("Grasp + retrieve",               ("grasp", "no-grasp"))]),
]
GROUPS = [
    ("PERCEIVE   real camera", [
        ("Segmentation + pose estimation", ("perceive",))]),
    ("SOLVE   digital twin", [
        ("Isaac Gym + teacher load",       ("twin",)),
        ("Scene settling",                 ("twin-settle",)),
        ("PPO rollout",                    ("solve",)),
        ("Plan finalization",              ("solve-export",))]),
    ("EXECUTE   real robot, open loop", [
        ("Controller load",                 ("student-load", "spiral-load")),
        ("Push trajectory",                ("approach", "push", "student", "spiral",
                                             "spiral-retract")),
        ("Grasp evaluation",               ("home", "re-sense")),
        ("Grasp + retrieve",               ("grasp", "no-grasp"))]),
]
STEP_END_PHASES = {
    "Isaac Gym + teacher load": "twin-settle",
    "Scene settling": "solve",
    "PPO rollout": "solve-export",
    "Plan finalization": "solve-done",
}
# The teacher baseline has no digital twin: no simulator load, no offline
# rollout, no replay. Its first observation is the initial scene and every
# later one is a re-sense, so the loop cannot be split into per-stage rows --
# build_steps keys on the FIRST occurrence of a phase, and these phases repeat
# once per primitive. One row for the loop is the honest presentation.
TEACHER_FULL_OBS_GROUPS = [
    ("PREPARE", [
        ("Teacher + perception models", ("teacher-load",)),
        ("Robot to experiment home",    ("initial-home",))]),
    ("PERCEIVE   real camera", [
        ("Initial scene segmentation",  ("teacher-retract", "teacher-sense"))]),
    ("SOLVE ON THE ROBOT   re-sensed before every primitive", [
        ("Grasp check + teacher push",  ("teacher-gn", "teacher-policy",
                                         "teacher-return", "teacher-push")),
        ("Grasp + retrieve",            ("grasp", "no-grasp"))]),
]
# The spiral baseline has no digital twin either. It perceives the clean scene
# once, then observes the (partially occluded) scene before every 1 cm spiral
# command, runs the grasp network in place, and retracts only to re-sense a
# hidden target or to confirm a positive grasp check. Like the teacher, its
# per-step phases repeat, so the loop is one row.
SPIRAL_GROUPS = [
    ("PERCEIVE   real camera", [
        ("Initial scene segmentation",  ("perceive",))]),
    ("PREPARE", [
        ("Perception + grasp models",   ("spiral-load",)),
        ("Robot to experiment home",    ("initial-home",))]),
    ("SPIRAL ON THE ROBOT   observed before every 1 cm step", [
        ("Grasp check + spiral push",   ("spiral", "spiral-observe", "spiral-sense",
                                         "spiral-gn", "spiral-retract", "spiral-return",
                                         "re-sense")),
        ("Grasp + retrieve",            ("grasp", "no-grasp"))]),
]
# PMBS re-senses the full scene from PMBS home before every decision and plans one
# push with parallel MCTS in its own simulator; the loop is one row as well.
PMBS_GROUPS = [
    ("PERCEIVE   real camera", [
        ("Initial scene segmentation",  ("perceive",))]),
    ("PREPARE", [
        ("PMBS models + simulator",     ("pmbs-load",)),
        ("Robot to experiment home",    ("initial-home",))]),
    ("PMBS ON THE ROBOT   re-sensed before every push", [
        ("Grasp check + MCTS push",     ("pmbs-observe", "pmbs-gn", "pmbs-search", "pmbs-push",
                                         "pmbs-retract", "re-sense")),
        ("Grasp + retrieve",            ("grasp", "no-grasp"))]),
]

# Google News palette (partnermarketinghub.withgoogle.com, "fifteen core
# colors"), as BGR. The page assigns no roles, so: LIGHT variants carry text --
# they are the ones that sit on a dark ground legibly -- and MEDIUM variants
# mark graphics (bullets, accent edge, rules), where a saturated hue reads well
# at small size. Black #202124 is the panel itself and Grey #9AA0A6 is anything
# not yet reached.
C_PANEL = (36, 33, 32)        # #202124 black
C_RULE = (166, 160, 154)      # #9AA0A6 grey
C_TEXT = (244, 243, 241)      # #F1F3F4 light grey
# Measured against #202124: the LIGHT variants clear AA by miles (12.4:1) but
# read as white -- done and in-progress stopped being distinguishable at a
# glance. The MEDIUM ones keep their hue and still clear AA (green 5.27:1,
# blue 4.52:1), so state text uses those. Medium red is the one that does not
# (4.10:1), so a failed row takes light red for text and medium red for its
# mark.
C_DONE = (83, 168, 52)        # #34A853 medium green
C_DONE_MARK = (83, 168, 52)
C_NOW = (244, 133, 66)        # #4285F4 medium blue
C_NOW_MARK = (244, 133, 66)
C_WAIT = (166, 160, 154)      # #9AA0A6 grey            6.10:1
C_SKIP = (207, 210, 250)      # #FAD2CF light red
C_SKIP_MARK = (53, 67, 234)   # #EA4335 medium red
C_ACCENT = (4, 188, 251)      # #FBBC04 yellow


def build_steps(events, groups=GROUPS):
    """[{name, t0, t1, skipped}] -- when each step started and finished.

    A step ends when the NEXT step that actually ran begins (or at "done" for
    the last one), so the durations tile the run with no gaps and no overlap.
    """
    first = {}
    for e in events:
        first.setdefault(e["phase"], e["t"])
    seq = []
    definitions = [step for _, steps in groups for step in steps]
    for name, phases in definitions:
        ts = [first[p] for p in phases if p in first]
        seq.append({"name": name, "t0": min(ts) if ts else None, "t1": None,
                    "skipped": "no-grasp" in first and "Grasp +" in name})
    nxt = first.get("done")
    for (name, phases), st in zip(reversed(definitions), reversed(seq)):
        if st["t0"] is None:
            continue
        # an explicit "-done" event wins: without one a step runs until the
        # NEXT step starts and swallows whatever happens in between -- the
        # policy-solve row was reporting the executor's process startup too
        exact_end = first.get(STEP_END_PHASES.get(name, ""))
        ends = [first[p + "-done"] for p in phases if p + "-done" in first]
        st["t1"] = exact_end if exact_end is not None else max(ends) if ends else nxt
        nxt = st["t0"]
    return seq


def step_state(st, ts):
    """-> ('done'|'now'|'wait', seconds_to_show or None)"""
    if st["t0"] is None or ts < st["t0"]:
        return "wait", None
    if st["t1"] is not None and ts >= st["t1"]:
        return "done", st["t1"] - st["t0"]
    return "now", ts - st["t0"]


def phase_at(events, ts):
    """The phase in force at wall-clock ts."""
    cur = ("", "")
    for e in events:
        if e["t"] <= ts:
            cur = (e["phase"], e.get("detail", ""))
        else:
            break
    # Internal controller events keep the annotation clock aligned without
    # exposing model diagnostics in the rendered video.
    if str(cur[1]).startswith("internal:"):
        return cur[0], ""
    return cur


def model_score_at(events, ts):
    """Persist the terminal original-GN score through retract/grasp/home."""
    score = ""
    for event in events:
        if event["t"] > ts:
            break
        if event.get("phase") in ("teacher-sense", "spiral-sense"):
            score = ""              # a new full observation is being acquired
        detail = str(event.get("detail", ""))
        if detail.startswith("Original grasp network score:"):
            score = detail
    return score


def rollout_clocks(events):
    """Return simulated and wall seconds recorded at the rollout boundary."""
    for event in events:
        if event.get("phase") != "solve-export":
            continue
        detail = str(event.get("detail", ""))
        sim = re.search(r"(?:^|;\s*)simulated_s=([0-9.]+)", detail)
        wall = re.search(r"(?:^|;\s*)wall_s=([0-9.]+)", detail)
        if sim:
            return float(sim.group(1)), float(wall.group(1)) if wall else None
    return None, None


def numeric_path_key(path):
    """Sort dump2 before dump10 while retaining ordinary lexical grouping."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", os.path.basename(path))]


def workspace_polygon(calib_path):
    """The 44.8 cm workspace square projected into this camera's image.

    Only meaningful for the calibrated top-down D455. Drawing the same boundary
    the sim videos draw is what makes the real top-down directly comparable to
    the digital twin footage.
    """
    import json as _json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    from isaacgymenvs.open_loop import frames
    cam2base = np.loadtxt(calib_path)
    b2c = np.linalg.inv(cam2base)
    return frames, b2c


def _project(frames, b2c, K, x_sim, y_sim, z=0.0):
    p = b2c @ np.array([y_sim, -x_sim, z, 1.0])
    return (int(round(p[0] * K["fx"] / p[2] + K["cx"])),
            int(round(p[1] * K["fy"] / p[2] + K["cy"])))


def _primitive_waypoints(xy, action, total=0.04):
    """Two simulator primitive-library waypoints, used only for video traces."""
    s, ds = total / 2.0, total / (2.0 * np.sqrt(2.0))
    card = np.asarray([[0, s], [s, 0], [0, -s], [-s, 0]], float)
    if action < 4:
        d1 = d2 = card[action]
    else:
        k = action - 4
        direction, mode = k // 3, k % 3
        diag = np.asarray([[ds, ds], [-ds, ds], [ds, -ds], [-ds, -ds]])
        first = np.asarray([[s, 0], [-s, 0], [s, 0], [-s, 0]])
        second = np.asarray([[0, s], [0, s], [0, -s], [0, -s]])
        if mode == 0:
            d1 = d2 = diag[direction]
        elif mode == 1:
            d1, d2 = first[direction], second[direction]
        else:
            d1, d2 = second[direction], first[direction]
    xy = np.asarray(xy, float)
    return xy + d1, xy + d1 + d2


def load_closed_loop_overlay(out_dir, events):
    """Load post-filter masks and teacher/student EEF paths for D455 video."""
    traj_path = os.path.join(out_dir, "trajectory.json")
    timing_path = os.path.join(out_dir, "real_timing.json")
    mask_paths = sorted(glob.glob(os.path.join(
        out_dir, "observations", "camera_masks_step_*.npz")))
    if not os.path.exists(traj_path):
        traj_path = os.path.join(out_dir, "metadata", "trajectory.json")
    if not os.path.exists(timing_path):
        timing_path = os.path.join(out_dir, "metadata", "real_timing.json")
    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(
            out_dir, "perception", "observations", "camera_masks_step_*.npz")))
    # Read older completed trials without perpetuating their stale folder name.
    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(
            out_dir, "dual_gn", "camera_masks_step_*.npz")))
    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(
            out_dir, "perception", "dual_gn", "camera_masks_step_*.npz")))
    if not (os.path.exists(traj_path) and os.path.exists(timing_path) and mask_paths):
        return None
    try:
        traj = json.load(open(traj_path))
        timing = json.load(open(timing_path))
        intrinsics_path = os.path.join(out_dir, "d455_topdown_dump0_K.json")
        if not os.path.exists(intrinsics_path):
            intrinsics_path = os.path.join(
                out_dir, "perception", "d455_topdown_dump0_K.json")
        K = json.load(open(intrinsics_path))
        records = sorted(glob.glob(os.path.join(out_dir, "*_gn16_predictions.npz")),
                         key=numeric_path_key)
        if not records:
            records = sorted(glob.glob(os.path.join(
                out_dir, "perception", "*_gn16_predictions.npz")),
                key=numeric_path_key)
        cam2base = (np.load(records[-1], allow_pickle=False)["camera_to_base"] if records
                    else np.loadtxt(os.environ.get("TRACE_CALIB", "data/calibration/camera_to_base.txt")))
        b2c = np.linalg.inv(cam2base)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
        from isaacgymenvs.open_loop import frames as frame_defs

        start = np.asarray(traj["start_eef"][:2], float)
        # The teacher prior is its recorded dense EEF state sequence. Connecting
        # stored wp2 directly to the next primitive's wp1 also connects across
        # the simulator's settle offset and creates artificial zig-zag teeth.
        teacher = [start] + [np.asarray(d["eef"][:2], float)
                             for d in traj.get("dense", [])]

        controller = timing.get("controller")
        if controller in ("circular_spiral", "pmbs"):
            observe_phase = "spiral-observe" if controller == "circular_spiral" else "pmbs-observe"
            observe = [e["t"] for e in events if e.get("phase") == observe_phase]
            kinds = [o.get("kind", "partial") for o in timing.get("observations", [])]
            titles = {"initial": "Initial observation", "partial": "Partial observation",
                      "clean": "Re-sensed observation"}
            n = min(len(observe), len(mask_paths))
            start_t = observe[0] if observe else 0.0
            stop_t = next((e["t"] for e in events
                           if e.get("phase") in ("grasp", "no-grasp", "done") and e["t"] >= start_t),
                          observe[-1] + 1.0 if observe else 0.0)
            return {"frames": frame_defs, "b2c": b2c, "K": K,
                    "teacher": np.zeros((0, 2)), "student": np.zeros((0, 2)), "segments": [],
                    "step_times": observe[:n], "start_t": start_t, "stop_t": stop_t,
                    "mask_paths": mask_paths[:n], "mask_cache": {},
                    "observation_titles": [titles.get(k, "Observation") for k in kinds[:n]],
                    "observation_title": "Partial observation", "traces": False}
        primitive_kinds = ("primitive", "decision") if controller == "teacher_full_observation" else ("primitive",)
        primitives = [x for x in timing.get("log", [])
                      if x.get("kind") in primitive_kinds and "action" in x]
        student_segments, student = [], [start]
        for rec in primitives:
            p0 = np.asarray(rec["eef_sim"][:2], float)
            p1, p2 = _primitive_waypoints(p0, int(rec["action"]))
            student_segments.append((p0, p1, p2))
            student += [p1, p2]

        # One internal step event is written immediately before each commanded
        # primitive. Keep the legacy score prefix readable for old trials.
        if controller == "teacher_full_observation":
            step_events = [e["t"] for e in events
                           if e.get("phase") == "teacher-gn" and
                           str(e.get("detail", "")).startswith("evaluating ")]
        else:
            step_events = [e["t"] for e in events if e.get("phase") == "student" and
                           (str(e.get("detail", "")).startswith("internal: student step;") or
                            str(e.get("detail", "")).startswith(
                                "Occlusion-aware grasp classifier score:"))]
            step_events = step_events[:len(student_segments)]
        # The first twin marker is emitted immediately after the clean initial
        # pose estimate completes. The initial segmentation is therefore ready
        # to display while model warmup and twin solving continue in parallel.
        if controller == "teacher_full_observation":
            overlay_start_t = step_events[0] if step_events else 0.0
            stop_t = next((e["t"] for e in events
                           if e.get("phase") in ("grasp", "no-grasp", "return-home", "done")
                           and e["t"] >= overlay_start_t),
                          step_events[-1] + 1.0 if step_events else 0.0)
        else:
            overlay_start_t = next((e["t"] for e in events if e.get("phase") == "twin"),
                                   step_events[0] if step_events else 0.0)
            stop_t = next((e["t"] for e in events if e.get("phase") == "home"),
                          step_events[-1] + 1.0 if step_events else 0.0)
        return {"frames": frame_defs, "b2c": b2c, "K": K,
                "teacher": np.asarray(teacher), "student": np.asarray(student),
                "segments": student_segments, "step_times": step_events,
                "start_t": overlay_start_t, "stop_t": stop_t,
                "mask_paths": mask_paths, "mask_cache": {},
                "observation_title": ("Re-sensed observation"
                                      if controller == "teacher_full_observation"
                                      else "Occluded observations")}
    except Exception as exc:
        print(f"[rec] closed-loop overlay unavailable: {exc}")
        return None


def draw_closed_loop_overlay(img, overlay, ts, show_rollout_traces=False):
    """Draw the controller's own observations and, when requested, rollout paths.

    Student: arm-occluded observations taken mid-push. Teacher full-observation:
    clean re-sensed scenes taken at PMBS home, one per primitive.
    """
    times = overlay["step_times"]
    # Do not carry the last observation into the terminal grasp evaluation,
    # where it is stale -- for the student it is also arm-occluded and can make
    # a graspable layout look blocked.
    if not times or ts < overlay["start_t"] or ts >= overlay["stop_t"]:
        return
    student_started = ts >= times[0]
    step = min(max(0, np.searchsorted(times, ts, side="right") - 1),
               len(overlay["mask_paths"]) - 1)
    cache = overlay["mask_cache"]
    if step not in cache:
        with np.load(overlay["mask_paths"][step], allow_pickle=False) as data:
            kept = data["kept"].astype(bool)
            rejected = data["rejected"].astype(bool)
            scene_rgb = (data["scene_rgb"].copy() if "scene_rgb" in data
                         else np.zeros((320, 320, 3), np.uint8))
            cache[step] = (kept, rejected, scene_rgb)
    kept, rejected, scene_rgb = cache[step]

    # Render the resolved canonical segmented scene directly. Instance masks
    # can overlap in camera space; compositing them produced mixed colours near
    # the target. build_real_heightmap has already resolved each retained mask
    # into the exact partial scene consumed by the original grasp network.
    off = int(overlay["frames"].CANVAS_WS_OFFSET)
    crop = scene_rgb[off:-off, off:-off, ::-1]  # workspace only; stored RGB -> BGR
    # The canonical policy raster uses sim axes, while the D455 video uses the
    # physical top-down camera axes. Rotate the inset into the same viewing
    # orientation as the main image so object directions compare directly.
    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    tile_w, tile_h = 332, 332
    canvas = fit_on_canvas(crop, tile_w, tile_h)
    box_x = 14
    box_y = max(150, (img.shape[0] - tile_h) // 2)
    cv2.rectangle(img, (box_x, box_y - 36),
                  (box_x + tile_w, box_y + tile_h + 1), C_PANEL, -1)
    titles = overlay.get("observation_titles")
    title = (titles[step] if titles and step < len(titles)
             else overlay.get("observation_title", "Occluded observations"))
    cv2.putText(img, title,
                (box_x + 12, box_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, C_TEXT, 1, cv2.LINE_AA)
    img[box_y:box_y + tile_h, box_x:box_x + tile_w] = canvas
    if not show_rollout_traces or overlay.get("traces") is False:
        return
    # The teacher/student paths become meaningful only when execution starts;
    # before that, show the available initial segmentation by itself.
    if not student_started:
        return

    def projected(points):
        return np.asarray([_project(overlay["frames"], overlay["b2c"], overlay["K"],
                                    float(p[0]), float(p[1]), z=0.020)
                           for p in points], np.int32)

    teacher_px = projected(overlay["teacher"])
    if len(teacher_px) > 1:
        cv2.polylines(img, [teacher_px], False, (4, 188, 251), 3, cv2.LINE_AA)

    # Grow the student path according to wall-clock progress through its current
    # primitive. The trace therefore changes as the policy changes its actions.
    complete = min(step, len(overlay["segments"]))
    student_points = [overlay["student"][0]]
    for i in range(complete):
        student_points.extend(overlay["segments"][i][1:])
    if step < len(overlay["segments"]):
        t0 = times[step]
        t1 = times[step + 1] if step + 1 < len(times) else overlay["stop_t"]
        frac = float(np.clip((ts - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
        p0, p1, p2 = overlay["segments"][step]
        if frac <= 0.5:
            student_points.append(p0 + (p1 - p0) * (2.0 * frac))
        else:
            student_points.extend([p1, p1 + (p2 - p1) * (2.0 * frac - 1.0)])
    student_px = projected(student_points)
    if len(student_px) > 1:
        cv2.polylines(img, [student_px], False, (255, 210, 70), 4, cv2.LINE_AA)
        cv2.circle(img, tuple(student_px[-1]), 6, (255, 210, 70), -1, cv2.LINE_AA)

    # Compact trace legend at the lower left.
    ly = img.shape[0] - 82
    cv2.rectangle(img, (14, ly), (400, ly + 42), C_PANEL, -1)
    cv2.line(img, (28, ly + 20), (68, ly + 20), (4, 188, 251), 3, cv2.LINE_AA)
    cv2.putText(img, "teacher rollout", (78, ly + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, C_TEXT, 1, cv2.LINE_AA)
    cv2.line(img, (225, ly + 20), (265, ly + 20), (255, 210, 70), 4, cv2.LINE_AA)
    cv2.putText(img, "student", (275, ly + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, C_TEXT, 1, cv2.LINE_AA)


def load_selected_grasp_overlay(out_dir, events, timing):
    """Load the executed GN bin for the successful grasp/retrieval interval."""
    grasp = timing.get("real_grasp") or {}
    if not grasp.get("graspable") or grasp.get("rotation_idx") is None:
        return None
    start_t = next((e["t"] for e in events if e.get("phase") == "grasp"), None)
    if start_t is None:
        return None
    selected = int(grasp["rotation_idx"])
    grids = sorted(glob.glob(os.path.join(out_dir, "*_gn16.png")), key=numeric_path_key)
    if not grids:
        grids = sorted(glob.glob(os.path.join(out_dir, "perception", "*_gn16.png")))
        grids.sort(key=numeric_path_key)
    if not grids:
        return None
    grid = os.path.splitext(grids[-1])[0]
    tile_path = os.path.join(grid + "_tiles", f"bin_{selected:02d}.png")
    tile = cv2.imread(tile_path)
    if tile is None:
        return None

    # Keep the chosen orientation visible throughout approach, closure, lift,
    # and retrieval. The terminal marker is the point at which grasping has
    # finished; failed evaluations never enter this function.
    stop_t = next((e["t"] for e in events
                   if e.get("phase") == "done" and e["t"] >= start_t), None)
    if stop_t is None:
        stop_t = start_t + max(1.0, float(timing.get("grasp_seconds", 0.0)))
    if stop_t <= start_t:
        return None
    return {"image": tile, "bin": selected, "start_t": start_t, "stop_t": stop_t}


def draw_selected_grasp_overlay(img, overlay, ts):
    """Show only the executed orientation through successful retrieval."""
    if overlay is None or not overlay["start_t"] <= ts < overlay["stop_t"]:
        return
    tile_w, tile_h = 332, 332
    canvas = fit_on_canvas(overlay["image"], tile_w, tile_h)
    box_x = 14
    # The grasp phase adds a score detail row to the top band, making it taller
    # than during student execution. Keep this title below that expanded band.
    box_y = max(205, (img.shape[0] - tile_h) // 2 + 30)
    cv2.rectangle(img, (box_x, box_y - 52),
                  (box_x + tile_w, box_y + tile_h + 1), C_PANEL, -1)
    cv2.putText(img, "Selected grasp orientation", (box_x + 12, box_y - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, f"Bin {overlay['bin']:02d} of 16",
                (box_x + 12, box_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, C_DONE, 1, cv2.LINE_AA)
    img[box_y:box_y + tile_h, box_x:box_x + tile_w] = canvas


def draw_band(img, u, TXT, label, subtitle, body, total):
    th_ = max(1, int(u))
    """Full-width strip across the TOP of the frame.

    For the top-down D455 the mat sits low and centred, so the whole upper strip
    is bare table -- a corner panel there wastes the one layout the frame is
    actually offering. The same rows are laid out in columns instead of a
    stack: identity on the left, one column per pipeline group, total on the
    right. The D415 keeps the corner panel; its frame has no such empty band.
    """
    h, w = img.shape[:2]
    pad, dot, gap = int(14 * u), int(13 * u), int(26 * u)
    cols = [[(label, 0.46, C_TEXT, None),
             (subtitle, 0.32, C_RULE, None),
             ("Speed 1x  (real time, unedited)", 0.28, C_RULE, None)]]
    cur = None
    for kind, name, tail, extra, col, state in body:
        if kind == "head":
            cur = [(name, 0.33, col, None)]
            cols.append(cur)
        else:
            cur.append((name, 0.37, col, (state, col, tail)))
            if extra:
                cur.append((extra, 0.28, C_RULE, None))
    if total:
        cols.append([("", 0.33, C_WAIT, None), (total[0], 0.38, total[2], None),
                     (total[1], 0.56, total[2], None)])

    widths = []
    for c in cols:
        wmax = 0
        for txt, sc, _, b in c:
            tw = TXT.w(txt, sc * u)
            vw = TXT.w(b[2], 0.37 * u) + int(12 * u) if b else 0
            wmax = max(wmax, (dot if b else 0) + tw + vw)
        widths.append(wmax)
    nrow = max(len(c) for c in cols)
    hei = int(20 * u) + nrow * int(23 * u) + pad

    img[0:hei] = np.array(C_PANEL, np.uint8)          # opaque, see draw corner
    cv2.rectangle(img, (0, hei - max(2, int(3 * u))), (w, hei), C_NOW_MARK, -1)

    x = pad
    for ci, (c, cw) in enumerate(zip(cols, widths)):
        if ci:
            cv2.line(img, (x - gap // 2, int(10 * u)), (x - gap // 2, hei - int(10 * u)),
                     C_RULE, 1)
        y = int(20 * u)
        for txt, sc, col, b in c:
            if b:
                state, bcol, tail = b
                cx, cy, r = x + dot // 2, y - int(4 * u), int(4.5 * u)
                mk = {C_DONE: C_DONE_MARK, C_NOW: C_NOW_MARK,
                      C_SKIP: C_SKIP_MARK}.get(bcol, bcol)
                cv2.circle(img, (cx, cy), r, mk, th_ if state == "wait" else -1)
                if state == "done":
                    cv2.line(img, (cx - r // 2, cy), (cx, cy + r // 2), C_PANEL, th_)
                    cv2.line(img, (cx, cy + r // 2), (cx + r, cy - r), C_PANEL, th_)
                TXT.put(txt, (x + dot, y), sc * u, col)
                TXT.put(tail, (x + cw - TXT.w(tail, 0.37 * u),
                                        y), 0.37 * u, col)
            elif txt:
                TXT.put(txt, (x, y), sc * u, col, bold=(sc >= 0.38))
            y += int(23 * u)
        x += cw + gap
    return hei



# --- text rendering ---------------------------------------------------------
# cv2.putText draws HERSHEY fonts: 1960s single-weight vector strokes with no
# hinting, kerning or real typeface design. That, not the capture resolution or
# the x264 settings, is what made the overlays look cheap. Pillow + FreeType
# gives real glyphs, a bold weight for headings, and correct metrics -- and
# because layout is measured with the SAME engine that draws, nothing shifts.
#
# Text is queued and flushed in one PIL pass per frame; converting per call
# would cost a round trip each time.
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_FONTS = {}


def _font(px, bold):
    key = (int(px), bool(bold))
    if key not in _FONTS:
        from PIL import ImageFont
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            _FONTS[key] = ImageFont.truetype(os.path.join(_FONT_DIR, name), max(6, int(px)))
        except OSError:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


class Text:
    """Queue text, measure it, then draw it all in one Pillow pass.

    Sizes are still given in the cv2 font-scale units the layout code was
    written against; PX_PER_SCALE converts. 30 was picked so DejaVu's cap
    height matches HERSHEY_SIMPLEX at the same scale, which is what keeps the
    existing spacing constants valid.
    """
    PX_PER_SCALE = 33.0

    def __init__(self):
        self.q = []

    def w(self, text, sc, bold=False):
        if not text:
            return 0
        return int(round(_font(sc * self.PX_PER_SCALE, bold).getlength(text)))

    def put(self, text, xy, sc, color, bold=False):
        if text:
            self.q.append((text, xy, sc, color, bold))

    def flush(self, img):
        if not self.q:
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        for text, (x, y), sc, color, bold in self.q:
            fnt = _font(sc * self.PX_PER_SCALE, bold)
            # xy is a BASELINE in the cv2 convention; PIL anchors "ls" the same
            d.text((x, y), text, font=fnt, fill=tuple(int(c) for c in color[::-1]),
                   anchor="ls")
        self.q.clear()
        img[:, :, :] = np.asarray(pil)[:, :, ::-1]
        return img


def load_timing(out_mp4):
    """Load real_timing.json beside the videos or in organized metadata."""
    try:
        directory = os.path.dirname(out_mp4)
        path = os.path.join(directory, "real_timing.json")
        if not os.path.exists(path):
            path = os.path.join(directory, "metadata", "real_timing.json")
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def fit_on_canvas(img, w, h):
    """Letterbox img into a w x h frame, preserving aspect."""
    sc = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def encode(spool, stamps, out_mp4, fps, t0, label, events=(), subtitle="",
           calib=None, intr=None, endcard=None, endcard_s=4.0, crf=14, preset="slow",
           layout="corner", ext="png", exec_label=None, write_raw=True,
           show_rollout_traces=False):
    """Overlay a TOP-LEFT info panel on each spooled frame and write the mp4.

    Top-right corner, not a bottom band: the workspace sits low and left of
    centre in both views, so a footer covers the part of the scene the video
    exists to show and the left corner crowds the clutter.
    """
    if not stamps:
        print("[rec] no frames captured")
        return None
    h, w = cv2.imread(os.path.join(spool, "%06d." % 0 + ext)).shape[:2]
    raw = out_mp4.replace(".mp4", "_annotated_tmp.mp4")   # mp4v intermediate, replaced by the x264 cut
    # Encode at the rate frames were ACTUALLY captured, not the nominal stream
    # rate: spooling a PNG per frame drops a few percent (d455 measured 28.6 of
    # a nominal 30), and encoding at the nominal rate plays the result ~5% fast.
    # Only a measured rate makes the "speed 1x" caption below true.
    # A measured AVERAGE rate is honest only when capture was uniform. When the
    # disk stalls the spool -- a concurrent encode on the same drive can starve
    # it down to 12.6 fps against a nominal 30 -- the idle stretches keep up and
    # the busy ones drop, so ONE average rate plays the quiet parts long and the
    # busy parts fast, and an event 48 s in can land 115 s into the video.
    # Put the frames back on a uniform clock:
    # sample the spool at a constant rate and repeat the last frame captured
    # before each tick, so one second of video is one second of the run
    # everywhere, dropped frames included.
    stamps = np.asarray(stamps, dtype=np.float64)
    measured = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                if len(stamps) > 1 else fps)
    if len(stamps) > 1:
        times = np.arange(stamps[0], stamps[-1], 1.0 / fps)
        order = np.clip(np.searchsorted(stamps, times, side="right") - 1,
                        0, len(stamps) - 1)
    else:
        times, order = stamps.copy(), np.zeros(1, dtype=int)
    print(f"[rec] capture measured {measured:.2f} fps; encoding on a uniform "
          f"{fps:.2f} fps clock ({len(times)} frames, {len(stamps)} captured)")
    # Unannotated copy straight from the spool, same rate and codec settings:
    # the overlay is opaque, so nothing under it survives in the annotated cut.
    try:
        if not write_raw:
            raise RuntimeError("raw copy already exists")
        import subprocess, imageio_ffmpeg
        clean = out_mp4.replace(".mp4", "_raw.mp4")
        # Same uniform clock as the annotated cut: a concat list naming one
        # spooled frame per tick. %06d input would re-impose the average rate.
        listing = os.path.join(spool, "uniform_clock.txt")
        with open(listing, "w") as fh:
            for idx in order:
                fh.write("file '%s'\n" % os.path.join(spool, "%06d." % idx + ext))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-r", f"{fps:.4f}", "-i", listing,
                        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-tune", "film",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", clean], check=True)
        print(f"[rec] raw (unannotated) video -> {clean}")
    except Exception as e:
        if write_raw:
            print(f"[rec] raw video failed ({e}); annotated cut only")
    teacher_full_obs = bool(exec_label and "teacher full-observation" in exec_label.lower())
    spiral_run = bool(exec_label and "(spiral)" in exec_label.lower())
    pmbs_run = bool(exec_label and "(pmbs)" in exec_label.lower())
    phase_names = {event.get("phase") for event in events}
    if teacher_full_obs:
        annotation_groups = TEACHER_FULL_OBS_GROUPS
    elif spiral_run:
        annotation_groups = SPIRAL_GROUPS
    elif pmbs_run:
        annotation_groups = PMBS_GROUPS
    elif phase_names.intersection({'twin-settle', 'solve-export'}):
        annotation_groups = GROUPS
    else:
        annotation_groups = LEGACY_GROUPS
    steps = build_steps(events, groups=annotation_groups)
    rollout_sim_s, _ = rollout_clocks(events)
    timing_record = load_timing(out_mp4) or {}
    stock_threshold = float(timing_record.get("stock_gn_threshold", 0.70))
    terminal_grasp = timing_record.get("real_grasp") or {}
    oow_failure = bool(timing_record.get("oow_failure")) or \
        timing_record.get("stop_reason") in ("out_of_workspace", "off_mat", "object_missing")
    terminal_network_graspable = bool(
        terminal_grasp and terminal_grasp.get(
            "network_graspable",
            float(terminal_grasp.get("q", 0)) >= stock_threshold))
    # Closed-loop graphics are opt-in through the wrapper's explicit execution
    # label. Open-loop also uses the band layout, so artifact presence alone
    # must never make it inherit the student mask/trace presentation.
    is_closed_loop = bool(exec_label and "closed loop" in exec_label.lower())
    closed_loop_overlay = (
        load_closed_loop_overlay(os.path.dirname(out_mp4), events)
        if layout == "band" and (is_closed_loop or teacher_full_obs) else None)
    selected_grasp_overlay = (
        load_selected_grasp_overlay(os.path.dirname(out_mp4), events, timing_record)
        if layout == "band" and (is_closed_loop or teacher_full_obs) else None)
    TXT, TXTC = Text(), Text()          # frame text, end-card text
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = cv2.FONT_HERSHEY_SIMPLEX
    u = max(1.0, w / 1280.0)
    PH_COL = {"approach": (200, 200, 200), "push": (250, 190, 110),
              "student-load": (244, 133, 66), "spiral-load": (244, 133, 66),
              "spiral": (250, 190, 110), "spiral-retract": (200, 160, 240),
              "spiral-observe": (110, 220, 255), "spiral-sense": (110, 220, 255),
              "spiral-gn": (110, 220, 255), "spiral-return": (200, 160, 240),
              "initial-home": (200, 160, 240),
              "pmbs-load": (244, 133, 66), "pmbs-observe": (110, 220, 255),
              "pmbs-gn": (110, 220, 255), "pmbs-search": (244, 133, 66),
              "pmbs-push": (250, 190, 110), "pmbs-retract": (200, 160, 240),
              "teacher-load": (244, 133, 66), "teacher-retract": (200, 160, 240),
              "teacher-sense": (110, 220, 255), "teacher-gn": (110, 220, 255),
              "teacher-policy": (244, 133, 66), "teacher-return": (200, 160, 240),
              "teacher-push": (250, 190, 110),
              "home": (200, 160, 240), "re-sense": (110, 220, 255),
              "grasp": (120, 255, 120), "no-grasp": (120, 120, 255),
              "done": (180, 180, 180)}
    for ts, i in zip(times, order):
        img = cv2.imread(os.path.join(spool, "%06d." % i + ext))
        if img is None:
            continue
        if calib is not None and intr is not None:
            try:
                frames, b2c = calib
                (wx0, wx1), (wy0, wy1) = (frames.SIM_WORKSPACE_LIMITS[0],
                                          frames.SIM_WORKSPACE_LIMITS[1])
                pts = np.array([_project(frames, b2c, intr, a, b) for a, b in
                                ((wx0, wy0), (wx1, wy0), (wx1, wy1), (wx0, wy1))], np.int32)
                cv2.polylines(img, [pts], True, C_ACCENT, max(1, int(2 * u)), cv2.LINE_AA)
                TXT.put("Workspace  44.8 x 44.8 cm", (pts[0][0] + int(8 * u), pts[0][1] + int(24 * u)), 0.45 * u, C_ACCENT)
            except Exception:
                pass
        if closed_loop_overlay is not None:
            draw_closed_loop_overlay(img, closed_loop_overlay, ts,
                                     show_rollout_traces=show_rollout_traces)
        draw_selected_grasp_overlay(img, selected_grasp_overlay, ts)
        ph, det = phase_at(events, ts) if events else ("", "")
        if ph in ("student", "spiral", "spiral-retract", "spiral-observe", "spiral-sense",
                  "spiral-gn", "spiral-return", "pmbs-observe", "pmbs-gn",
                  "pmbs-search", "pmbs-push", "pmbs-retract", "teacher-retract",
                  "teacher-sense", "teacher-gn", "teacher-policy", "teacher-return",
                  "teacher-push", "home", "re-sense",
                  "grasp", "no-grasp"):
            if not (ph == "no-grasp" and str(det).startswith("OOW failure")):
                det = model_score_at(events, ts) or det
        th_ = max(1, int(u))
        pad, indent, dot = int(9 * u), int(12 * u), int(15 * u)

        # Build the panel as ROWS, each measured on its own, then back each one
        # with a rectangle only as wide as it needs. The result tapers -- the
        # title and the checkpoint line are the wide part, the step rows below
        # are narrow -- instead of one block sized to the widest line, which
        # covered scene the narrow rows never used.
        # Rows carry a BLOCK id. Width is taken per block, not per row, so the
        # panel steps once per section instead of once per line -- a per-row
        # taper left the group captions floating on their own narrow strips
        # while their own steps ran wider underneath them.
        rows = []            # (height, parts, bullet, block)

        def row(hgt, parts, bullet=None, block=0):
            rows.append((int(hgt * u), parts, bullet, block))

        row(24, [(0, label, 0.46, C_TEXT)])
        if subtitle:
            row(19, [(0, subtitle, 0.34, C_RULE)])
        row(21, [(0, "Speed 1x  (real time, unedited)", 0.30, C_RULE)])

        # measure the step columns first so name/value align across all groups
        body = []
        k = 0
        groups = [(f"EXECUTE   {exec_label}" if exec_label and g.startswith("EXECUTE")
                   and not teacher_full_obs else g, st)
                  for g, st in annotation_groups]
        for gname, gsteps in groups:
            gst = steps[k:k + len(gsteps)]; k += len(gsteps)
            visible = [(definition, state) for definition, state in zip(gsteps, gst)
                       if not (definition[0] == "Student policy load" and
                               state["t0"] is None)]
            visible_states = [state for _, state in visible]
            gdone = all(step_state(x, ts)[0] == "done" for x in visible_states)
            gnow = any(step_state(x, ts)[0] == "now" for x in visible_states)
            body.append(("head", gname, "", "",
                         C_DONE if gdone else C_NOW if gnow else C_WAIT, None))
            for definition, st in visible:
                state, dur = step_state(st, ts)
                col = {"done": C_DONE, "now": C_NOW, "wait": C_WAIT}[state]
                if st["skipped"] and state != "wait":
                    col = C_SKIP
                    tail = ("OOW failure" if oow_failure
                            else "Grasp withheld" if terminal_network_graspable
                            else "Not graspable")
                else:
                    tail = f"{dur:.1f} s" if dur is not None else "-"
                    if (definition[0] == "PPO rollout" and state == "done"
                            and rollout_sim_s is not None):
                        tail = f"{rollout_sim_s:.1f} s sim"
                extra = det if (state == "now" and det) else ""
                body.append(("step", st["name"], tail, extra, col, state))
        nw = max(TXT.w(r[1], 0.38 * u)
                 for r in body if r[0] == "step")
        vx = indent + dot + nw + int(14 * u)          # value column

        blk = 0
        for kind, name, tail, extra, col, state in body:
            if kind == "head":
                blk += 1
                row(21, [(0, name, 0.33, col)], block=blk)
            else:
                row(23, [(indent + dot, name, 0.38, col), (vx, tail, 0.38, col)],
                    (indent, col, state), block=blk)
                if extra:
                    # its own indented sub-line, not a third column: the live
                    # detail is the longest string on screen and putting it
                    # inline made the EXECUTE block the widest, which inverted
                    # the taper the panel is supposed to have
                    row(17, [(indent + dot + int(12 * u), extra, 0.29,
                              C_RULE)], block=blk)

        t_start = next((st["t0"] for st in steps if st["t0"] is not None), None)
        t_end = steps[-1]["t1"] if steps and steps[-1]["t1"] is not None else None
        total_row = None
        if t_start is not None:
            over = t_end is not None and ts >= t_end
            total = (t_end if over else max(ts, t_start)) - t_start
            tc = C_DONE if over else C_TEXT
            total_row = ("Total Time:", f"{total:.1f} s", tc)
            row(30, [(indent + dot, total_row[0], 0.42, tc),
                     (vx, total_row[1], 0.48, tc)], block=99)

        if layout == "band":
            draw_band(img, u, TXT, label, subtitle, body, total_row)
            TXT.flush(img)
            vw.write(img)
            continue

        own = [max(dx + TXT.w(t, sc * u)
                   for dx, t, sc, _ in parts) + 2 * pad for _, parts, _, _ in rows]
        # One uniform width. Per-block widths were tried and read as a ragged
        # left edge rather than a deliberate taper; the panel is small enough
        # now (the live detail moved to its own sub-line) that a clean rectangle
        # costs no more scene than the stepped version did.
        wid = max(own)
        widths = [wid] * len(rows)
        hei = sum(hgt for hgt, _, _, _ in rows) + pad

        # one blended pass over the union of the row rectangles, so the taper is
        # a single translucent shape rather than per-row seams
        x0 = max(0, w - wid)
        region = img[0:hei, x0:x0 + wid]
        # Opaque, not translucent. A see-through panel lets scene texture fight
        # the text, and its contrast changes as the scene does -- the arm
        # passing behind it visibly dims rows. A solid ground reads as a
        # designed caption rather than a screenshot artefact, and legibility
        # stops depending on what is underneath.
        tint = np.full_like(region, 1) * np.array(C_PANEL, np.uint8)
        mask = np.zeros(region.shape[:2], bool)
        yy = 0
        for (hgt, _, _, _), rw in zip(rows, widths):
            mask[yy:yy + hgt, wid - rw:] = True
            yy += hgt
        mask[yy:yy + pad, wid - widths[-1]:] = True
        region[mask] = tint[mask]
        yy = 0
        for (hgt, _, _, _), rw in zip(rows, widths):        # accent edge, stepped
            cv2.rectangle(img, (x0 + wid - rw, yy),
                          (x0 + wid - rw + max(2, int(3 * u)), yy + hgt), C_NOW, -1)
            yy += hgt

        yy = 0
        for (hgt, parts, bullet, _), rw in zip(rows, widths):
            bx = x0 + wid - rw + pad
            base = yy + hgt - int(7 * u)
            if bullet is not None:
                bdx, bcol, bstate = bullet
                cx, cy, r = bx + bdx + dot // 2, base - int(4 * u), int(4.5 * u)
                mk = {C_DONE: C_DONE_MARK, C_NOW: C_NOW_MARK,
                      C_SKIP: C_SKIP_MARK}.get(bcol, bcol)
                cv2.circle(img, (cx, cy), r, mk, th_ if bstate == "wait" else -1)
                if bstate == "done":
                    cv2.line(img, (cx - r // 2, cy), (cx, cy + r // 2), C_PANEL, th_)
                    cv2.line(img, (cx, cy + r // 2), (cx + r, cy - r), C_PANEL, th_)
            for dx, t, sc, col in parts:
                TXT.put(t, (bx + dx, base), sc * u, col, bold=(sc >= 0.42))
            yy += hgt
        TXT.flush(img)
        vw.write(img)

    # Held end-card: the grasp network's 16-rotation search, the one step the
    # live video cannot show because it happens at a single instant. Held, not
    # overlaid, so the wall-clock on the frames above stays honest -- the clock
    # is frozen here on purpose and the card says so.
    if endcard and os.path.exists(endcard):
        # The panel is roughly square and the video is 16:9, so it fits
        # height-limited and leaves a wide black margin either side. Put the
        # headline facts there at a readable size -- the panel's own captions
        # shrink to a few pixels at this scale and only the heat pattern
        # survives, which is the picture but not the point.
        bar = int(28 * u)
        card = fit_on_canvas(cv2.imread(endcard), w, h - bar)
        card = np.vstack([card, np.zeros((bar, w, 3), np.uint8) + np.array(C_PANEL, np.uint8)])
        pw = int(cv2.imread(endcard).shape[1] * min(w / cv2.imread(endcard).shape[1],
                                                    (h - bar) / cv2.imread(endcard).shape[0]))
        # Only the open-loop replay evaluates the grasp network once. Every
        # closed-loop controller here -- student, spiral, full-observation
        # teacher -- runs it before each primitive, and this card shows the
        # evaluation it accepted.
        TXTC.put("Held frame - the grasp network is evaluated before every primitive; "
                 "this is the accepted one"
                 if (is_closed_loop or teacher_full_obs) else
                 "Held frame - the grasp network is evaluated ONCE, after the pushes",
                 (int(10 * u), h - int(9 * u)), 0.42 * u, C_RULE)

        lines = [("Original grasp network", 0.52, C_TEXT),
                 ("16-rotation search", 0.40, C_RULE)]
        g = terminal_grasp
        if g:
            # Show the trained network's graspability score. A separate
            # geometric diagnostic may affect old execution records, but it
            # must not relabel an above-threshold network prediction on video.
            ok = bool(g.get("network_graspable",
                            float(g.get("q", 0)) >= stock_threshold))
            lines += [("", 0.30, (0, 0, 0)),
                      (f"Network Q {g.get('q', 0):.3f}", 0.60, C_DONE if ok else C_NOW),
                      ("Above threshold" if ok else "Below threshold", 0.42,
                       C_DONE if ok else C_NOW),
                      (f"Threshold {stock_threshold:.2f}", 0.34, C_RULE),
                      ("", 0.30, (0, 0, 0)),
                      ("Selected orientation" if ok else "Orientation", 0.34, C_RULE),
                      (f"bin {g.get('rotation_idx', '?')} of 16" if ok
                       else "None viable", 0.46,
                       C_DONE if ok else C_NOW)]
        x = (w - pw) // 2 + pw + int(14 * u)
        if x < w - int(120 * u):                       # only if the margin is real
            y = int(60 * u)
            for txt, sc, col in lines:
                if txt:
                    TXTC.put(txt, (x, y), sc * u, col, bold=(sc >= 0.46))
                y += int(30 * u * sc / 0.4)
        TXTC.flush(card)
        for _ in range(max(1, int(round(endcard_s * fps)))):
            vw.write(card)
        print(f"[rec] end-card appended ({endcard_s:.0f}s): {endcard}")
    vw.release()
    print(f"[rec] encoding x264 crf {crf} preset {preset}")
    try:
        import subprocess, imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                        "-y", "-i", raw, "-c:v", "libx264", "-crf", str(crf),
                        "-preset", preset, "-tune", "film",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4], check=True)
        os.remove(raw)
    except Exception as e:
        print(f"[rec] ffmpeg failed ({e}); keeping {raw}")
        out_mp4 = raw
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--name", required=True, help="output basename, e.g. d415_scene")
    ap.add_argument("--out", default="runs/real", help="output directory")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=0, help="0 = pick the best the link allows")
    # Frames spool as lossless PNG, so the ONLY quality loss in the pipeline is
    # this encode. It runs after the robot has stopped, so a slow preset costs
    # the run nothing. (The real ceiling is the capture: a camera on USB 2.1 is
    # limited to 1280x720 at 15 fps and no encoder setting recovers that.)
    ap.add_argument("--layout", choices=("corner", "band"), default="corner",
                    help="corner: panel in the top-right (D415). band: full-width "
                         "strip across the top, for the top-down view whose upper "
                         "strip is bare table (D455).")
    ap.add_argument("--crf", type=int, default=14, help="x264 quality, lower is better")
    ap.add_argument("--preset", default="slow", help="x264 preset")
    ap.add_argument("--endcard-seconds", type=float, default=4.0,
                    help="hold the grasp-network panel this long at the end (0 = off)")
    ap.add_argument("--exec-label", default=None,
                    help='EXECUTE group header, e.g. "real robot, closed loop (student)"')
    ap.add_argument("--rollout-traces", action="store_true",
                    help="overlay teacher and evolving student paths (default: off)")
    ap.add_argument("--depth", action="store_true", help="also stream depth (needed to serve re-sense)")
    ap.add_argument(
        "--fast-dumps", action="store_true",
        help="publish read-back-verified RGB-D dumps without forcing each payload to "
             "stable storage; avoids multi-second fsync stalls on external media")
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--t0", type=float, default=0.0, help="shared epoch start; 0 = first frame")
    ap.add_argument("--label", default="", help="title line burned into the panel")
    ap.add_argument("--subtitle", default="", help="second line, e.g. the checkpoint")
    ap.add_argument("--calib", default="", help="4x4 cam2base txt; draws the workspace box")
    args = ap.parse_args()

    import pyrealsense2 as rs
    os.makedirs(args.out, exist_ok=True)
    ctrl = os.path.join(args.out, args.name + ".ctrl")
    os.makedirs(ctrl, exist_ok=True)
    for f in ("ready", "stop", "dump"):
        p = os.path.join(ctrl, f)
        if os.path.exists(p):
            os.remove(p)
    spool = os.path.join(args.out, args.name + "_frames")
    os.makedirs(spool, exist_ok=True)

    # USB 2.1 links cannot do 720p30; fall back rather than fail
    fps = args.fps
    if fps == 0:
        ctx = rs.context()
        usb = "3"
        for d in ctx.query_devices():
            if d.get_info(rs.camera_info.serial_number) == args.serial:
                usb = d.get_info(rs.camera_info.usb_type_descriptor)
        fps = 15 if usb.startswith("2") else 30
        print(f"[rec] {args.name}: usb {usb} -> {args.width}x{args.height} @ {fps} fps")

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, fps)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, fps)
    profile = pipe.start(cfg)
    # SAME colour setup the perception path uses. Without it the D455 handed
    # back a red-cast stream (frame mean BGR 87/121/203) in which every
    # instance scored 0.000 purple, so the in-run re-sense reported "no target"
    # on a scene that graspability scored 1.10 through perceive_scene's own
    # capture. The dumps this recorder serves ARE the re-sense input, so they
    # have to come out of an identically configured sensor.
    try:
        from isaacgymenvs.open_loop.perceive_scene import settle_color_sensor
        settle_color_sensor(profile, pipe, warmup=30, tag="rec")
        print(f"[rec] {args.name}: colour sensor settled and frozen")
    except Exception as e:
        print(f"[rec] {args.name}: WARNING colour setup skipped ({e})")
    align = rs.align(rs.stream.color) if args.depth else None
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() if args.depth else 1.0

    # Start numbering after whatever is already on disk. Restarting at 0 makes a
    # re-run overwrite the previous run's dump0 under the same filename, which
    # hides the new dump from anything watching for one to appear.
    _prev = glob.glob(os.path.join(args.out, f"{args.name}_dump*.done"))
    n_dump = 1 + max([int(re.search(r"_dump(\d+)\.done$", f).group(1)) for f in _prev],
                     default=-1)
    stamps, t0 = [], None
    last_K = None
    open(os.path.join(ctrl, "ready"), "w").close()
    print(f"[rec] {args.name}: recording -> {spool}")
    if args.fast_dumps:
        print(f"[rec] {args.name}: fast RGB-D dump publication enabled "
              "(atomic rename + read-back verification, no per-file fsync)")
    try:
        while True:
            fs = pipe.wait_for_frames()
            if align is not None:
                fs = align.process(fs)
            cf = fs.get_color_frame()
            if not cf:
                continue
            now = time.time()
            if t0 is None:
                t0 = args.t0 if args.t0 > 0 else now
            color = np.asanyarray(cf.get_data())
            if last_K is None:
                _i = cf.profile.as_video_stream_profile().intrinsics
                last_K = {"fx": _i.fx, "fy": _i.fy, "cx": _i.ppx, "cy": _i.ppy}
            cv2.imwrite(os.path.join(spool, "%06d.png" % len(stamps)), color)
            stamps.append(now)

            if os.path.exists(os.path.join(ctrl, "dump")):
                base = os.path.join(args.out, f"{args.name}_dump{n_dump}")
                intr = cf.profile.as_video_stream_profile().intrinsics
                finals = [base + "_color.png", base + "_K.json"]
                temps = [base + "_color.tmp.png", base + "_K.tmp.json"]
                if args.depth:
                    finals.append(base + "_depth.npy")
                    temps.append(base + "_depth.tmp.npy")
                try:
                    for path in temps + finals + [base + ".done.tmp"]:
                        if os.path.exists(path):
                            os.remove(path)
                    if not cv2.imwrite(temps[0], color):
                        raise OSError("cv2.imwrite rejected the dump color PNG")
                    with open(temps[1], "w") as stream:
                        json.dump({"fx": intr.fx, "fy": intr.fy,
                                   "cx": intr.ppx, "cy": intr.ppy}, stream, indent=2)
                        stream.flush()
                        if not args.fast_dumps:
                            os.fsync(stream.fileno())
                    if args.depth:
                        df = fs.get_depth_frame()
                        with open(temps[2], "wb") as stream:
                            np.save(stream, np.asanyarray(df.get_data()).astype(np.float32)
                                    * depth_scale)
                            stream.flush()
                            if not args.fast_dumps:
                                os.fsync(stream.fileno())
                    if not args.fast_dumps:
                        with open(temps[0], "rb") as stream:
                            os.fsync(stream.fileno())
                    for temp, final in zip(temps, finals):
                        os.replace(temp, final)
                    check = cv2.imread(finals[0])
                    if check is None or check.shape != color.shape:
                        raise OSError("installed dump color PNG failed read-back")
                    if args.depth:
                        check_depth = np.load(finals[2])
                        if check_depth.ndim != 2 or check_depth.size == 0:
                            raise OSError("installed dump depth failed read-back")
                    with open(base + ".done.tmp", "w") as stream:
                        stream.write("complete\n")
                        stream.flush()
                        if not args.fast_dumps:
                            os.fsync(stream.fileno())
                    os.replace(base + ".done.tmp", base + ".done")
                    os.remove(os.path.join(ctrl, "dump"))
                    print(f"[rec] {args.name}: dumped verified frame -> {base}_*")
                    n_dump += 1
                except Exception as exc:
                    print(f"[rec] {args.name}: dump write failed, retrying ({exc})", flush=True)
                    for path in temps + finals + [base + ".done.tmp", base + ".done"]:
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            pass

            if os.path.exists(os.path.join(ctrl, "stop")) or (now - t0) > args.max_seconds:
                break
    finally:
        pipe.stop()

    out_mp4 = os.path.join(args.out, args.name + ".mp4")
    calib = workspace_polygon(args.calib) if args.calib else None
    # any run writes at most one of these; take the last if a re-sense repeated
    cards = sorted(glob.glob(os.path.join(args.out, "*_gn16.png")), key=numeric_path_key)
    if not cards:
        cards = sorted(glob.glob(os.path.join(args.out, "perception", "*_gn16.png")))
        cards.sort(key=numeric_path_key)
    written = encode(spool, stamps, out_mp4, fps, t0,
                     args.label or args.name.replace("_", " "),
                     events=load_phases(args.out), subtitle=args.subtitle,
                     calib=calib, intr=last_K,
                     endcard=cards[-1] if cards else None,
                     endcard_s=args.endcard_seconds,
                     crf=args.crf, preset=args.preset, layout=args.layout,
                     exec_label=args.exec_label,
                     show_rollout_traces=args.rollout_traces)
    meas_fps = ((len(stamps) - 1) / (stamps[-1] - stamps[0])) if len(stamps) > 1 else fps
    meta = {"name": args.name, "serial": args.serial, "frames": len(stamps),
            "stream_fps": fps, "encoded_fps": round(meas_fps, 3),
            "t0_epoch": t0, "duration_s": round((stamps[-1] - t0), 3) if stamps else 0.0,
            "video": written, "dumps": n_dump}
    json.dump(meta, open(os.path.join(args.out, args.name + "_rec.json"), "w"), indent=2)
    if not written or not os.path.isfile(written) or os.path.getsize(written) == 0:
        raise RuntimeError(f"final video missing or empty; preserving source frames in {spool}")
    shutil.rmtree(spool)
    print(f"[rec] {args.name}: {len(stamps)} frames, {meta['duration_s']} s -> {written}; source frames removed")


if __name__ == "__main__":
    main()
