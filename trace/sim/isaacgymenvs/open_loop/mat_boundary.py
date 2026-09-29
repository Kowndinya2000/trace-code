"""Black-mat boundary from the clean D455 frame and a footprint-on-mat test.

The mat is the large dark quadrilateral under the clutter. Its four image corners
are found once per trial on the clean initial frame, then intersected with the
mat plane (median base-frame height of the mat pixels) using the calibrated
camera pose, which gives the boundary in the sim frame. An object counts as off
the mat as soon as any vertex of its top-face footprint (the simulated mesh
outline, placed at the estimated pose) lies outside that quadrilateral.
"""
import numpy as np
import cv2

from isaacgymenvs.open_loop import frames, perceive_scene as ps


def detect_mat_polygon(color, depth, K, cam2base, dark_value=None):
    """-> dict(corners_sim (4, 2), corners_px (4, 2), plane_z) or raise RuntimeError.

    The dark/bright split is Otsu's threshold on the brightness channel unless given:
    recorder frames (mat V~40, wood ~90) and direct captures (mat ~80, wood ~180) differ
    in exposure, so no fixed value works for both.
    """
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    if dark_value is None:
        dark_value, _ = cv2.threshold(cv2.GaussianBlur(hsv[..., 2], (9, 9), 0), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (hsv[..., 2] < dark_value).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        raise RuntimeError("mat not found: no dark region")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    h, w = mask.shape
    if len(quad) != 4 or cv2.contourArea(quad) < 0.15 * h * w:
        raise RuntimeError(f"mat not found: {len(quad)}-gon, area {cv2.contourArea(quad):.0f} px")
    if (quad[:, 0] <= 1).any() or (quad[:, 0] >= w - 2).any() or (quad[:, 1] <= 1).any() \
            or (quad[:, 1] >= h - 2).any():
        raise RuntimeError("mat not found: dark region touches the image border")

    _, _, zb = ps.deproject_to_sim(depth, K, cam2base)
    interior = cv2.erode(mask, np.ones((31, 31), np.uint8)).astype(bool) & (depth > 0.05)
    plane_z = float(np.median(zb[interior]))
    origin = cam2base[:3, 3]
    corners = []
    for u, v in quad.astype(float):
        ray = cam2base[:3, :3] @ np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
        t = (plane_z - origin[2]) / ray[2]
        xb, yb = (origin + t * ray)[:2]
        corners.append(frames.real_to_sim(xb, yb))
    return {"corners_sim": np.asarray(corners, float), "corners_px": quad.astype(int),
            "plane_z": plane_z}


def footprint_sim(obj):
    """(N, 2) sim-frame top-face outline of a located object."""
    poly = ps.silhouette_polygon(obj["name"])
    c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
    return poly @ np.array([[c, s], [-s, c]]) + np.array([obj["x"], obj["y"]])


def footprint_off_mat(obj, corners_sim):
    """Largest distance (m) any footprint vertex lies outside the mat; 0.0 if fully on it."""
    contour = np.asarray(corners_sim, np.float32).reshape(-1, 1, 2)
    worst = 0.0
    for x, y in footprint_sim(obj):
        signed = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        worst = max(worst, -signed)
    return worst


def off_mat_counts(objects, corners_sim, tolerance_m=0.0):
    """{class name: count} of real objects with part of the footprint off the mat."""
    counts = {}
    for obj in objects:
        if obj.get("pad"):
            continue
        if footprint_off_mat(obj, corners_sim) > tolerance_m:
            counts[obj["name"]] = counts.get(obj["name"], 0) + 1
    return counts


def instance_mat_measurements(instances, depth, K, cam2base, corners_sim,
                              min_z=0.025, max_z=0.065, min_points=50, stride=3):
    """Per segmented block: measured top-face points vs the mat, independent of pose fitting.

    Block walls are vertical, so the deprojected top face has the footprint's xy.
    Points outside the block height band (table, robot links) are ignored, which
    also drops robot masks. Returns dicts with the class, number of points, the
    centre, and ``edge_mm``: the 2nd-percentile signed distance to the mat edge
    (positive inside, negative outside), robust to a few mixed-depth boundary pixels.
    """
    xs, ys, zb = ps.deproject_to_sim(depth, K, cam2base)
    band = (depth > 0.05) & (zb >= min_z) & (zb <= max_z)
    contour = np.asarray(corners_sim, np.float32).reshape(-1, 1, 2)
    out = []
    for inst in instances:
        sel = inst["mask"].astype(bool) & band
        n = int(sel.sum())
        if n < min_points:
            continue
        X, Y = xs[sel][::stride], ys[sel][::stride]
        signed = np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), True)
                           for x, y in zip(X, Y)])
        out.append({"name": ps.CLASS_ID_TO_NAME[inst["class"]], "score": float(inst["score"]),
                    "points": n, "center": [round(float(X.mean()), 4), round(float(Y.mean()), 4)],
                    "edge_mm": round(1000 * float(np.percentile(signed, 2)), 1)})
    return out


def measured_off_mat_counts(measurements, tolerance_mm=3.0):
    """{class: count} of measured blocks with any footprint part beyond the mat edge."""
    counts = {}
    for m in measurements:
        if m["edge_mm"] < -tolerance_mm:
            counts[m["name"]] = counts.get(m["name"], 0) + 1
    return counts


def measured_block_count(measurements, reach_m=0.10, corners_sim=None):
    """Blocks whose centre is on the mat or within ``reach_m`` of it."""
    if corners_sim is None:
        return len(measurements)
    contour = np.asarray(corners_sim, np.float32).reshape(-1, 1, 2)
    return sum(1 for m in measurements
               if cv2.pointPolygonTest(contour, tuple(float(v) for v in m["center"]), True)
               >= -reach_m)
