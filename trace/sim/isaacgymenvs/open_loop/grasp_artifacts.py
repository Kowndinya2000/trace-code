"""Persist camera inputs and GN outputs for exact offline visualization."""
import json
from pathlib import Path

import cv2
import numpy as np

from isaacgymenvs.utils.prediction_vis import DEFAULT_TILE_SIZE


ARRAY_KEYS = ("rgb", "height", "segm", "preds", "best", "camera_bgr",
              "camera_depth", "camera_to_base")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def camera_debug(color_bgr, depth_m, intrinsics, camera_to_base):
    """Own the source buffers so later captures cannot change this result."""
    return dict(camera_bgr=np.array(color_bgr, copy=True),
                camera_depth=np.array(depth_m, copy=True),
                camera_intrinsics=dict(intrinsics),
                camera_to_base=np.array(camera_to_base, copy=True))


def save_grasp_record(dbg, grasp, panel_path, threshold=0.7, tile_size=DEFAULT_TILE_SIZE):
    """Always save available source data, including rejected/no-target captures.

    The NPZ contains only numeric arrays and JSON text (no pickled objects).
    A separate native-resolution PNG makes the original D455 frame viewable.
    """
    panel_path = Path(panel_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(dbg[key]) for key in ARRAY_KEYS if key in dbg}
    metadata = dict(version=1, threshold=threshold, tile_size=tile_size,
                    grasp=None if grasp is None else {k: v for k, v in grasp.items() if k != "_debug"},
                    camera_intrinsics=dbg.get("camera_intrinsics"))
    data_path = panel_path.with_name(panel_path.stem + "_predictions.npz")
    np.savez_compressed(data_path, **arrays,
                        metadata=np.array(json.dumps(metadata, default=_json_value)))
    if "camera_bgr" in dbg:
        frame_path = panel_path.with_name(panel_path.stem + "_d455.png")
        if not cv2.imwrite(str(frame_path), dbg["camera_bgr"]):
            raise OSError(f"Could not write D455 frame: {frame_path}")
    return data_path


def load_grasp_record(path):
    """Read a saved record without camera, GPU, or network/model access."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("version") != 1:
            raise ValueError("Unsupported grasp record version")
        dbg = {key: data[key].copy() for key in ARRAY_KEYS if key in data}
    if metadata.get("camera_intrinsics") is not None:
        dbg["camera_intrinsics"] = metadata["camera_intrinsics"]
    return dbg, metadata
