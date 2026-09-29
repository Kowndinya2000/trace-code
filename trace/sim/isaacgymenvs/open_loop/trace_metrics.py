"""Timing-aware physics-trace comparison, without simulator dependencies."""
import numpy as np


def compare_physics(reference, replay):
    if abs(reference["dt"] - replay["dt"]) > 1e-9:
        raise ValueError("Cannot compare traces on different physics clocks")
    a, b = reference["samples"], replay["samples"]
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("Cannot compare empty traces")
    ea, eb = [np.asarray([s["eef_state"] for s in rows[:n]]) for rows in (a, b)]
    p = np.linalg.norm(ea[:, :3] - eb[:, :3], axis=1) * 1000
    # atan2 handles exactly equal float32 quaternions without acos roundoff.
    qa, qb = ea[:, 3:7], eb[:, 3:7]
    qa = qa / np.linalg.norm(qa, axis=1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=1, keepdims=True)
    qb *= np.where(np.sum(qa * qb, axis=1, keepdims=True) < 0, -1, 1)
    angle = 4 * np.arctan2(np.linalg.norm(qa - qb, axis=1), np.linalg.norm(qa + qb, axis=1))
    result = {"reference_ticks": len(a), "replay_ticks": len(b),
              "complete": len(a) == len(b), "compared_ticks": n,
              "duration_error_s": (len(b) - len(a)) * reference["dt"],
              "max_eef_position_mm": float(p.max()),
              "rms_eef_position_mm": float(np.sqrt(np.mean(p ** 2))),
              "final_eef_position_mm": float(p[-1]),
              "max_eef_orientation_deg": float(np.rad2deg(angle).max()),
              "max_linear_velocity_error_mm_s": float(np.linalg.norm(ea[:, 7:10] - eb[:, 7:10], axis=1).max()*1000),
              "eef_state_exact": len(a) == len(b) and bool(np.array_equal(ea, eb))}
    for key in ("joint_pos", "joint_targets", "block_state"):
        if key not in a[0] or key not in b[0]:
            continue
        va, vb = [np.asarray([s[key] for s in rows[:n]]) for rows in (a, b)]
        result[key + "_exact"] = len(a) == len(b) and bool(np.array_equal(va, vb))
        if key == "block_state":
            result["max_object_position_mm"] = float(np.linalg.norm(va[..., :3] - vb[..., :3], axis=-1).max()*1000)
        elif key == "joint_pos":
            result["max_joint_position_error_rad"] = float(np.abs(va-vb).max())
    for key in ("eef_state", "joint_pos", "joint_vel", "block_state"):
        va, vb = np.asarray(reference["initial"][key]), np.asarray(replay["initial"][key])
        result["initial_" + key + "_exact"] = bool(np.array_equal(va, vb))
    return result
