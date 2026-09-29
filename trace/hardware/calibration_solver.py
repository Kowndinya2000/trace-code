"""ChArUco observations and eye-to-hand refinement; no hardware or GUI access.

Transforms use column vectors: X = base_T_camera, E = base_T_tcp,
Y = tcp_T_board. Predicted board corners in the camera are inv(X) @ E @ Y @ p.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def make_board(square_m=0.025, marker_m=0.01875):
    if not (np.isfinite(square_m) and np.isfinite(marker_m)
            and 0 < marker_m < square_m):
        raise ValueError("Board lengths must satisfy 0 < marker < square")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((4, 4), square_m, marker_m, dictionary)
    board.setLegacyPattern(True)
    return board, cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def pose_matrix(pose):
    """UR TCP pose: xyz metres followed by a rotation vector in radians, not RPY."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected six finite pose coordinates")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_parameters(matrix):
    return np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]


def parameter_matrix(parameters):
    return pose_matrix(np.r_[parameters[3:6], parameters[:3]])


def mean_transform(matrices):
    matrices = np.asarray(matrices)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(matrices[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return result


def board_pose(ids, corners, board, camera_matrix, distortion):
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(ids) < 6 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Need at least six distinct ChArUco corners")
    points = np.asarray(board.getChessboardCorners(), dtype=float)[ids]
    if np.linalg.matrix_rank(points[:, :2] - points[:, :2].mean(axis=0)) < 2:
        raise ValueError("ChArUco corners are collinear")
    ok, rvec, tvec = cv2.solvePnP(points, corners, camera_matrix, distortion,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Board pose estimation failed")
    rvec, tvec = cv2.solvePnPRefineLM(points, corners, camera_matrix, distortion, rvec, tvec)
    transform = pose_matrix(np.r_[tvec.ravel(), rvec.ravel()])
    if not np.isfinite(transform).all() or np.any((points @ transform[:3, :3].T
                                                 + transform[:3, 3])[:, 2] <= 0):
        raise ValueError("Invalid board pose or board behind camera")
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
    error = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
    return transform, float(np.sqrt(np.mean(error**2)))


def detect_observation(bgr, board, detector, camera_matrix, distortion, max_rms_px=1.0):
    """Retain board-specific IDs and refined chessboard corners, not marker origins."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    if marker_ids is None:
        raise ValueError("No board markers detected")
    valid = np.isin(marker_ids.ravel(), board.getIds().ravel())
    marker_corners = [corner for corner, keep in zip(marker_corners, valid) if keep]
    marker_ids = marker_ids[valid]
    if len(marker_ids) < 2 or len(np.unique(marker_ids)) != len(marker_ids):
        raise ValueError("Too few board markers or duplicate marker IDs")
    marker_corners, marker_ids, _, _ = detector.refineDetectedMarkers(
        gray, board, marker_corners, marker_ids, rejected,
        cameraMatrix=camera_matrix, distCoeffs=distortion)
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board,
        cameraMatrix=camera_matrix, distCoeffs=distortion)
    if ids is None or corners is None:
        raise ValueError("No ChArUco corners detected")
    transform, rms = board_pose(ids, corners, board, camera_matrix, distortion)
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"Board reprojection RMS {rms:.3f} px exceeds {max_rms_px:.3f} px")
    return {"ids": ids.ravel().tolist(), "corners": corners.reshape(-1, 2).tolist(),
            "board_to_camera": transform.tolist(), "pnp_rms_px": rms,
            "board_markers": int(np.isin(marker_ids, board.getIds()).sum())}


def aggregate_observations(frames, board, camera_matrix, distortion,
                           max_jitter_px=0.35, max_rms_px=1.0):
    """Use the median location of corners shared by at least three stable frames."""
    if len(frames) < 3:
        raise ValueError("At least three valid frames are required per pose")
    common = sorted(set.intersection(*(set(frame["ids"]) for frame in frames)))
    if len(common) < 6:
        raise ValueError("Fewer than six shared ChArUco corners in the burst")
    stack = np.array([[frame["corners"][frame["ids"].index(i)] for i in common]
                      for frame in frames])
    median = np.median(stack, axis=0)
    jitter = np.linalg.norm(stack - median, axis=2)
    p95 = float(np.percentile(jitter, 95))
    if not np.isfinite(p95) or p95 > max_jitter_px:
        raise ValueError(f"Corner jitter {p95:.3f} px exceeds {max_jitter_px:.3f} px")
    transform, rms = board_pose(common, median, board, camera_matrix, distortion)
    if rms > max_rms_px:
        raise ValueError(f"Median-corner reprojection RMS is too high: {rms:.3f} px")
    return {"ids": common, "corners": median.tolist(),
            "board_to_camera": transform.tolist(), "pnp_rms_px": rms,
            "corner_jitter_p95_px": p95, "frame_count": len(frames)}


def project_observation(parameters, observation, object_points, camera_matrix, distortion):
    base_T_camera = parameter_matrix(parameters[:6])
    tcp_T_board = parameter_matrix(parameters[6:])
    camera_T_board = (np.linalg.inv(base_T_camera)
                      @ np.asarray(observation["tcp_to_base"]) @ tcp_T_board)
    points = object_points[np.asarray(observation["ids"], dtype=int)]
    rvec = Rotation.from_matrix(camera_T_board[:3, :3]).as_rotvec()
    pixels, _ = cv2.projectPoints(points, rvec, camera_T_board[:3, 3],
                                 camera_matrix, distortion)
    depths = (points @ camera_T_board[:3, :3].T + camera_T_board[:3, 3])[:, 2]
    return pixels.reshape(-1, 2), depths


def pixel_residuals(parameters, observations, object_points, camera_matrix, distortion):
    errors = []
    for observation in observations:
        predicted, _ = project_observation(parameters, observation, object_points,
                                            camera_matrix, distortion)
        errors.append((predicted - np.asarray(observation["corners"])).ravel())
    return np.concatenate(errors)


def error_metrics(parameters, observations, object_points, camera_matrix, distortion):
    pixel_errors, translation_errors, rotation_errors = [], [], []
    base_T_camera = parameter_matrix(parameters[:6])
    tcp_T_board = parameter_matrix(parameters[6:])
    for observation in observations:
        pixels, depths = project_observation(parameters, observation, object_points,
                                             camera_matrix, distortion)
        if np.any(depths <= 0) or not np.isfinite(pixels).all():
            raise ValueError("Solution projects a board behind the camera or has nonfinite pixels")
        pixel_errors.extend(np.linalg.norm(pixels - observation["corners"], axis=1))
        predicted = np.asarray(observation["tcp_to_base"]) @ tcp_T_board
        measured = base_T_camera @ np.asarray(observation["board_to_camera"])
        translation_errors.append(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]) * 1000)
        rotation_errors.append(np.rad2deg(Rotation.from_matrix(
            predicted[:3, :3].T @ measured[:3, :3]).magnitude()))
    pixels = np.asarray(pixel_errors)
    return {"poses": len(observations), "corners": len(pixels),
            "pixel_rms": float(np.sqrt(np.mean(pixels**2))),
            "pixel_median": float(np.median(pixels)),
            "pixel_p95": float(np.percentile(pixels, 95)), "pixel_max": float(pixels.max()),
            "board_origin_mean_mm": float(np.mean(translation_errors)),
            "board_origin_max_mm": float(np.max(translation_errors)),
            "board_orientation_mean_deg": float(np.mean(rotation_errors)),
            "board_orientation_max_deg": float(np.max(rotation_errors))}


def seed_parameters(observations, camera_to_base=None):
    ee = np.asarray([obs["tcp_to_base"] for obs in observations])
    boards = np.asarray([obs["board_to_camera"] for obs in observations])
    inverse_ee = np.linalg.inv(ee)
    seeds = []
    methods = {"PARK": cv2.CALIB_HAND_EYE_PARK, "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
               "TSAI": cv2.CALIB_HAND_EYE_TSAI, "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
               "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
    cameras = []
    if camera_to_base is not None:
        cameras.append(("supplied", np.asarray(camera_to_base)))
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                inverse_ee[:, :3, :3], inverse_ee[:, :3, 3],
                boards[:, :3, :3], boards[:, :3, 3], method=method)
            camera = np.eye(4)
            camera[:3, :3], camera[:3, 3] = rotation, translation.ravel()
            cameras.append((name, camera))
        except cv2.error:
            continue
    for name, camera in cameras:
        if (camera.shape != (4, 4) or not np.isfinite(camera).all()
                or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(camera[:3, :3]), 1, atol=1e-5)):
            continue
        # Estimate a separate marker offset for EACH initializer.
        offset = mean_transform(inverse_ee @ camera @ boards)
        seeds.append((name, np.r_[matrix_parameters(camera), matrix_parameters(offset)]))
    return seeds


def solve_dataset(dataset, validation_fraction=0.25, split_seed=7,
                  loss_scale_px=1.0, initial_camera=None):
    """Hold out entire poses before initializing or fitting; no all-data refit."""
    observations = dataset["observations"]
    if len(observations) < 16:
        raise ValueError("Need at least 16 accepted distinct poses (12 fit + 4 validation)")
    if not 0.15 <= validation_fraction <= 0.4:
        raise ValueError("Validation fraction must be between 0.15 and 0.4")
    camera_matrix = np.asarray(dataset["camera_matrix"], dtype=float)
    distortion = np.asarray(dataset["distortion"], dtype=float)
    points = np.asarray(dataset["object_points"], dtype=float)
    for observation in observations:
        ids = np.asarray(observation["ids"])
        corners = np.asarray(observation["corners"])
        if (ids.ndim != 1 or len(ids) < 6 or len(np.unique(ids)) != len(ids)
                or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= len(points))
                or corners.shape != (len(ids), 2) or not np.isfinite(corners).all()):
            raise ValueError("Invalid corner observations")
        for key in ("tcp_to_base", "board_to_camera"):
            transform = np.asarray(observation[key])
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
                raise ValueError(f"Invalid rigid transform: {key}")
    permutation = np.random.default_rng(split_seed).permutation(len(observations))
    n_validation = max(4, int(round(len(observations) * validation_fraction)))
    validation_indices = sorted(permutation[:n_validation].tolist())
    training_indices = sorted(permutation[n_validation:].tolist())
    training = [observations[i] for i in training_indices]
    validation = [observations[i] for i in validation_indices]
    if len(training) < 12:
        raise ValueError("Need at least twelve training poses after the validation split")
    rotations = Rotation.from_matrix(np.asarray([obs["tcp_to_base"] for obs in training])[:, :3, :3])
    rotation_vectors = (rotations[0].inv() * rotations).as_rotvec()
    excitation = np.linalg.svd(rotation_vectors, compute_uv=False)
    if excitation[1] < np.deg2rad(2):
        raise ValueError("Insufficient rotation diversity about different axes; no calibration saved")
    args = (training, points, camera_matrix, distortion)
    candidates = []
    for name, seed in seed_parameters(training, initial_camera):
        try:
            metrics = error_metrics(seed, *args)
            candidates.append((metrics["pixel_rms"], name, seed, metrics))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("No finite hand-eye initializer with positive board depths")
    candidates.sort(key=lambda item: item[0])
    # Try the two best training-only initializers. Validation never selects a seed.
    fits = []
    for _, name, seed, _ in candidates[:2]:
        fit = least_squares(pixel_residuals, seed, args=args, method="trf",
                            loss="soft_l1", f_scale=loss_scale_px,
                            x_scale="jac", max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        if fit.success and np.isfinite(fit.x).all():
            try:
                error_metrics(fit.x, *args)
                fits.append((fit.cost, name, fit))
            except ValueError:
                pass
    if not fits:
        raise ValueError("Optimization did not converge to a valid camera pose")
    _, seed_name, fit = min(fits, key=lambda item: item[0])
    norms = np.linalg.norm(fit.jac, axis=0)
    normalized_jac = fit.jac / np.maximum(norms, np.finfo(float).eps)
    singular_values = np.linalg.svd(normalized_jac, compute_uv=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-6))
    if rank != 12:
        raise ValueError(f"Calibration is underconstrained (Jacobian rank {rank}/12)")
    train_metrics = error_metrics(fit.x, *args)
    validation_metrics = error_metrics(fit.x, validation, points, camera_matrix, distortion)
    warnings = []
    if validation_metrics["pixel_rms"] > 1.0:
        warnings.append("Validation pixel RMS exceeds 1 px; inspect observations before using this candidate.")
    if singular_values[0] / singular_values[-1] > 1e4:
        warnings.append("Weak calibration geometry: normalized Jacobian condition exceeds 10000.")
    return {"camera_to_base": parameter_matrix(fit.x[:6]).tolist(),
            "board_to_tcp": parameter_matrix(fit.x[6:]).tolist(),
            "training": train_metrics, "validation": validation_metrics,
            "training_indices": training_indices, "validation_indices": validation_indices,
            "split_seed": split_seed, "optimizer": "trf/soft_l1 pixel residuals",
            "loss_scale_px": loss_scale_px, "initializer": seed_name,
            "initializer_training_metrics": {name: metrics for _, name, _, metrics in candidates},
            "optimizer_evaluations": fit.nfev, "optimizer_message": fit.message,
            "jacobian_rank": rank,
            "normalized_jacobian_condition": float(singular_values[0] / singular_values[-1]),
            "warnings": warnings,
            "validation_note": "Held-out pose consistency; not independent absolute robot positioning accuracy."}
