"""Shared grasp-network visualization.

Rendering is display-only and must not modify inference inputs or scores.
"""
import cv2
import numpy as np

from .constants import (
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL, GRIPPER_GRASP_WIDTH_PIXEL,
    PUSH_DISTANCE_PIXEL,
)


# Defaults shared by all grasp-network panel entry points.
DEFAULT_TILE_SIZE = 640
DEFAULT_MAGNIFICATION = 3.5
OVERVIEW_LABEL = "Scene overview"

# OpenCV BGR. Distinct outside/mat tones preserve rotation without a blue wash.
BACKGROUND_BGR = (57, 51, 44)
MAT_BGR = (114, 104, 90)
INK_BGR = (236, 233, 224)
ACCENT_BGR = (178, 228, 106)
HIGH_BGR = (108, 245, 255)
GRIPPER_BGR = (55, 55, 255)


def prediction_colors(scores):
    """Fixed Q=0..1 scale: coral -> orange -> bright yellow; Q>=1 saturates."""
    q = np.clip(np.nan_to_num(np.asarray(scores), nan=0.0), 0, 1)
    stops = np.array([0.0, 0.4, 0.7, 1.0])
    colors = np.array([[116, 115, 225], [65, 166, 255],
                       [55, 222, 255], HIGH_BGR])
    return np.stack([np.interp(q, stops, colors[:, c]) for c in range(3)],
                    axis=-1).astype(np.uint8)


def prediction_overlay(scores, color_heightmap, mat_mask=None):
    """Composite aligned Q/RGB (0..255) without altering either input.

    The mat mask distinguishes the rotated canvas from its outside corners.
    Zero Q leaves object colors visible. High Q stands out against the blue
    target without coloring unrelated objects or the mat.
    """
    q = np.clip(np.nan_to_num(np.asarray(scores), nan=0.0), 0, 1)
    rgb = np.asarray(color_heightmap)
    background = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    background = 0.85 * np.clip(background, 0, 255) + 0.15 * 255
    background[~np.any(rgb > 0, axis=-1)] = MAT_BGR
    if mat_mask is not None:
        background[np.asarray(mat_mask) == 0] = BACKGROUND_BGR
    alpha = np.where(q > 0, 0.7 + 0.3 * q, 0)[..., None]
    return np.rint((1 - alpha) * background + alpha * prediction_colors(q)).astype(np.uint8)


def _project(points, transform):
    return cv2.transform(np.asarray(points, np.float32).reshape(1, -1, 2), transform)[0]


def _view(scores, rgb, transform, size):
    """Warp original data once, keeping tiny Q cells at their true values."""
    color = cv2.warpAffine(rgb, transform, (size, size), flags=cv2.INTER_LINEAR)
    q = cv2.warpAffine(scores, transform, (size, size), flags=cv2.INTER_NEAREST)
    mat = cv2.warpAffine(np.ones(scores.shape, np.uint8), transform,
                         (size, size), flags=cv2.INTER_NEAREST)
    result = prediction_overlay(q, color, mat)
    h, w = scores.shape
    corners = _project([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)], transform)
    corners = np.rint(corners).astype(np.int32)
    cv2.polylines(result, [corners], True, (157, 149, 134), 1, cv2.LINE_AA)
    # Highlight one source edge to make the full-mat orientation explicit.
    cv2.line(result, tuple(corners[0]), tuple(corners[1]), (193, 210, 204),
             max(1, size//180), cv2.LINE_AA)
    return result, q


def _crosshair(image, point, scale):
    """Four separated arms leave the high-score pixel itself uncovered."""
    x, y = np.rint(point).astype(int)
    gap, reach = max(7, round(2.5*scale)), max(13, round(5*scale))
    arms = [((x-reach, y), (x-gap, y)), ((x+gap, y), (x+reach, y)),
            ((x, y-reach), (x, y-gap)), ((x, y+gap), (x, y+reach))]
    for start, end in arms:
        cv2.line(image, start, end, (25, 25, 25), 4, cv2.LINE_AA)
        cv2.line(image, start, end, INK_BGR, 2, cv2.LINE_AA)


def _gripper_outline(image, point, scale, is_push):
    x, y = np.rint(point).astype(int)
    if is_push:
        end = (x+round(PUSH_DISTANCE_PIXEL*scale), y)
        cv2.arrowedLine(image, (x+round(5*scale), y), end, ACCENT_BGR,
                        2, cv2.LINE_AA, tipLength=0.15)
        return
    reach = round(GRIPPER_GRASP_OUTER_DISTANCE_PIXEL*scale/2)
    half_width = round(GRIPPER_GRASP_WIDTH_PIXEL*scale/2)
    # Outline the gripper footprint while leaving the peak visible inside.
    a, b = (x-reach, y-half_width), (x+reach, y+half_width)
    cv2.rectangle(image, a, b, (28, 30, 25), 4, cv2.LINE_AA)
    cv2.rectangle(image, a, b, GRIPPER_BGR, 3, cv2.LINE_AA)


def render_prediction_grid(predictions, rgb, best, is_push=False, tile_size=DEFAULT_TILE_SIZE,
                           target_mask=None, threshold=0.7):
    """4-column grid with a common target crop and full rotated mat insets.

    A 640/3.5-source-pixel crop is shown at 3.5x with default 640px tiles. This is
    display magnification, not new network resolution or synthesized scores.
    Nearest-neighbor Q sampling preserves narrow peaks and their values.
    """
    if tile_size < 320:
        raise ValueError("tile_size must be at least 320 for readable annotations")
    scores = np.nan_to_num(np.asarray(predictions, np.float32), nan=0.0)
    rgb = np.asarray(rgb, np.uint8)
    n, h, w = scores.shape
    if target_mask is not None and np.any(target_mask):
        rows, cols = np.nonzero(target_mask)
        center = (float(cols.mean()), float(rows.mean()))
    elif np.any(scores > 0):
        center = (float(best[2]), float(best[1]))
    else:
        center = (w/2, h/2)
    crop_size = min(DEFAULT_TILE_SIZE/DEFAULT_MAGNIFICATION, h, w)
    scale = tile_size/crop_size
    canvas = np.full((((n+3)//4)*tile_size, 4*tile_size, 3), BACKGROUND_BGR, np.uint8)
    font, ui = cv2.FONT_HERSHEY_SIMPLEX, tile_size/DEFAULT_TILE_SIZE
    inset_size, pad = round(156*ui), round(12*ui)
    for k in range(n):
        angle = k*360.0/n
        transform = cv2.getRotationMatrix2D(center, angle, scale)
        transform[:, 2] += [tile_size/2-center[0], tile_size/2-center[1]]
        tile, q_view = _view(scores[k], rgb, transform, tile_size)
        high = (q_view >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Border actual high-Q cells without growing the predicted region.
        cv2.drawContours(tile, contours, -1, (24, 36, 42), 3, cv2.LINE_AA)
        cv2.drawContours(tile, contours, -1, HIGH_BGR, 1, cv2.LINE_AA)
        if scores[k].max() > 0:
            row, col = np.unravel_index(np.argmax(scores[k]), (h, w))
            peak = _project([(col, row)], transform)[0]
            _crosshair(tile, peak, scale)
            # The jaw footprint marks an executable grasp only; a sub-threshold
            # network best keeps just its crosshair.
            if k == best[0] and (is_push or scores[k].max() >= threshold):
                _gripper_outline(tile, peak, scale, is_push)

        # Fit the complete rotating square, including all four corners.
        full_scale = (inset_size-12*ui)/np.hypot(h, w)
        overview_transform = cv2.getRotationMatrix2D((w//2, h//2), angle, full_scale)
        overview_transform[:, 2] += [inset_size/2-w//2, inset_size/2-h//2]
        overview, _ = _view(scores[k], rgb, overview_transform, inset_size)
        inverse = cv2.invertAffineTransform(transform)
        crop_corners = _project([(0, 0), (tile_size-1, 0),
                                 (tile_size-1, tile_size-1), (0, tile_size-1)], inverse)
        crop_corners = np.rint(_project(crop_corners, overview_transform)).astype(np.int32)
        cv2.polylines(overview, [crop_corners], True, ACCENT_BGR, 1, cv2.LINE_AA)
        ix, iy = tile_size-pad-inset_size, tile_size-pad-inset_size
        tile[iy:iy+inset_size, ix:ix+inset_size] = overview
        cv2.rectangle(tile, (ix-1, iy-1), (ix+inset_size, iy+inset_size), (150, 144, 129), 1)
        cv2.rectangle(tile, (ix-1, iy-round(25*ui)), (ix+inset_size, iy-1), BACKGROUND_BGR, -1)
        cv2.putText(tile, OVERVIEW_LABEL,
                    (ix+round(5*ui), iy-round(8*ui)), font, .36*ui,
                    INK_BGR, max(1, round(ui)), cv2.LINE_AA)
        r, c = divmod(k, 4)
        canvas[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tile
    return canvas
