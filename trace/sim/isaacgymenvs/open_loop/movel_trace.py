"""Geometric fidelity metrics for measured UR moveL TCP traces."""
import numpy as np
from scipy.spatial.transform import Rotation


def point_to_polyline_distance(points, polyline):
    """Shortest Euclidean distance from each point to a piecewise-linear path."""
    points = np.asarray(points, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or polyline.ndim != 2 or points.shape[1] != polyline.shape[1]:
        raise ValueError("points and polyline must be two-dimensional with matching coordinates")
    if len(polyline) == 0:
        raise ValueError("polyline is empty")
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    start, delta = polyline[:-1], np.diff(polyline, axis=0)
    denom = np.sum(delta * delta, axis=1)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.sum(offset * delta[None, :, :], axis=2) / np.maximum(denom, 1e-20)
    projection = start[None, :, :] + np.clip(fraction, 0.0, 1.0)[..., None] * delta[None, :, :]
    return np.linalg.norm(points[:, None, :] - projection, axis=2).min(axis=1)


def summarize(reference_poses, samples, teacher_duration_s=None):
    """Compare measured TCP samples with a fixed-orientation moveL reference."""
    reference = np.asarray(reference_poses, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 6 or len(reference) < 2:
        raise ValueError("reference_poses must contain at least two 6D poses")
    if len(samples) < 2:
        raise ValueError("at least two measured TCP samples are required")
    actual = np.asarray([row["tcp_pose"] for row in samples], dtype=float)
    speed = np.asarray([row["tcp_speed"] for row in samples], dtype=float)
    times = np.asarray([row["time_s"] for row in samples], dtype=float)
    if actual.shape != (len(samples), 6) or speed.shape != (len(samples), 6):
        raise ValueError("invalid TCP sample shape")
    if not np.isfinite(np.r_[reference.ravel(), actual.ravel(), speed.ravel(), times]).all():
        raise ValueError("non-finite moveL trace")

    actual_to_reference = point_to_polyline_distance(actual[:, :2], reference[:, :2])
    reference_to_actual = point_to_polyline_distance(reference[:, :2], actual[:, :2])
    symmetric = np.r_[actual_to_reference, reference_to_actual]
    reference_length = np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1).sum()
    actual_length = np.linalg.norm(np.diff(actual[:, :2], axis=0), axis=1).sum()
    orientation_error = np.rad2deg(
        (Rotation.from_rotvec(actual[:, 3:])
         * Rotation.from_rotvec(reference[0, 3:]).inv()).magnitude())
    linear_speed = np.linalg.norm(speed[:, :3], axis=1)
    dt = np.diff(times)
    valid_dt = dt > 1e-6
    acceleration = (np.linalg.norm(np.diff(speed[:, :3], axis=0)[valid_dt], axis=1)
                    / dt[valid_dt]) if valid_dt.any() else np.asarray([])
    duration = float(times[-1] - times[0])

    result = {
        "comparison_basis": "measured TCP geometry against the commanded teacher polyline",
        "samples": int(len(samples)),
        "reference_points": int(len(reference)),
        "hardware_duration_s": duration,
        "teacher_duration_s": None if teacher_duration_s is None else float(teacher_duration_s),
        "duration_ratio_hardware_over_teacher": (
            None if not teacher_duration_s else duration / float(teacher_duration_s)),
        "reference_xy_path_cm": float(reference_length * 100),
        "measured_xy_path_cm": float(actual_length * 100),
        "path_length_ratio_percent": float(100 * actual_length / reference_length),
        "rms_symmetric_xy_error_mm": float(np.sqrt(np.mean(symmetric * symmetric)) * 1000),
        "max_actual_to_reference_xy_error_mm": float(actual_to_reference.max() * 1000),
        "max_reference_to_actual_xy_error_mm": float(reference_to_actual.max() * 1000),
        "final_xy_error_mm": float(np.linalg.norm(actual[-1, :2] - reference[-1, :2]) * 1000),
        "rms_z_error_mm": float(np.sqrt(np.mean((actual[:, 2] - reference[0, 2]) ** 2)) * 1000),
        "max_z_error_mm": float(np.abs(actual[:, 2] - reference[0, 2]).max() * 1000),
        "rms_orientation_error_deg": float(np.sqrt(np.mean(orientation_error ** 2))),
        "max_orientation_error_deg": float(orientation_error.max()),
        "max_measured_linear_speed_m_s": float(linear_speed.max()),
        "p95_measured_linear_speed_m_s": float(np.percentile(linear_speed, 95)),
        "p95_measured_linear_acceleration_m_s2": (
            float(np.percentile(acceleration, 95)) if len(acceleration) else None),
        "actual_samples_within_1mm_percent": float(100 * np.mean(actual_to_reference <= .001)),
        "reference_path_within_1mm_percent": float(100 * np.mean(reference_to_actual <= .001)),
    }
    result["geometry_passed"] = bool(
        result["rms_symmetric_xy_error_mm"] <= 1.0
        and result["max_actual_to_reference_xy_error_mm"] <= 2.0
        and result["max_reference_to_actual_xy_error_mm"] <= 2.0
        and result["final_xy_error_mm"] <= 1.0
        and 99.0 <= result["path_length_ratio_percent"] <= 101.0)
    result["perpendicular_passed"] = bool(result["max_orientation_error_deg"] <= .5)
    return result
