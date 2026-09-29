import torch 
import torch.nn.functional as F
import math
import cv2
import numpy as np
import base64
from isaacgymenvs.video_streaming_server_multi_cam import calculate_center
H, W = 224, 224
def offset_concave_orientation(matched_concave_angle: float) -> float:
    return (-matched_concave_angle) % 360

def offset_rect_orientation(matched_rect_angle: float) -> float: 
    return (-matched_rect_angle) % 180

def offset_half_cube_orientation(matched_half_cube_angle: float) -> float: 
    return (-matched_half_cube_angle) % 180

def offset_cube_orientation(matched_cube_angle: float) -> float: 
    return (-matched_cube_angle) % 90

def offset_triangle_orientation(matched_triangle_angle: float) -> float:
    return (-matched_triangle_angle + 180)%360


def get_ref_mask_info(mask_path, info_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    cx, cy = calculate_center(mask)
    return mask, cx, cy
def crop_center_pad(mask: np.ndarray, cx: int, cy: int, out=360) -> np.ndarray:
    H, W = mask.shape[:2]
    half = out // 2

    # desired crop box in source coords
    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    # clip to source
    sx0 = max(0, x0); sx1 = min(W, x1)
    sy0 = max(0, y0); sy1 = min(H, y1)

    # if nothing overlaps, return blank
    if sx0 >= sx1 or sy0 >= sy1:
        return np.zeros((out, out), dtype=mask.dtype)

    crop = mask[sy0:sy1, sx0:sx1]

    # paste into centered canvas
    canvas = np.zeros((out, out), dtype=mask.dtype)
    dx0 = sx0 - x0   # where crop lands in canvas
    dy0 = sy0 - y0
    canvas[dy0:dy0 + crop.shape[0], dx0:dx0 + crop.shape[1]] = crop
    return canvas
def base64_to_ndarray(payload: dict) -> np.ndarray:
    # 1) Decode base64 string back to bytes
    data_bytes = base64.b64decode(payload["data"])
    # 2) Recover dtype and shape
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    # 3) Create array from buffer and reshape
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return arr


def shift_zeropad(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    mask: (H,W) bool or uint8
    dx: +right, -left
    dy: +down,  -up
    Returns: (H,W) same dtype, translated with zero padding (NO wrap).
    """
    H, W = mask.shape
    out = torch.zeros_like(mask)

    x0_src = max(0, -dx)
    x0_dst = max(0,  dx)
    y0_src = max(0, -dy)
    y0_dst = max(0,  dy)

    w = min(W - x0_src, W - x0_dst)
    h = min(H - y0_src, H - y0_dst)

    if w > 0 and h > 0:
        out[y0_dst:y0_dst+h, x0_dst:x0_dst+w] = mask[y0_src:y0_src+h, x0_src:x0_src+w]
    return out

@torch.no_grad()
def precompute_rotations(ref_mask_u8: torch.Tensor,
                         angles_deg,
                         cx: float, cy: float,
                         device="cuda",
                         align_corners=True):
    """
    ref_mask_u8: (H,W) uint8 (0/255) or (0/1)
    angles_deg: iterable of angles
    cx, cy: rotation pivot in PIXEL coordinates (x=col, y=row)
            (pass your ref centroid, for example)
    returns: templates_bool (A,H,W) bool on device
    """
    assert ref_mask_u8.ndim == 2
    H, W = ref_mask_u8.shape

    ref = (ref_mask_u8 > 0).to(device=device, dtype=torch.float32)[None, None]  # (1,1,H,W)

    angles = torch.as_tensor(angles_deg, device=device, dtype=torch.float32) * (math.pi / 180.0)
    cos, sin = torch.cos(angles), torch.sin(angles)
    A = angles.numel()

    theta = torch.zeros((A, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] =  cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] =  sin
    theta[:, 1, 1] =  cos

    # --- pixel pivot -> normalized pivot ---
    if align_corners:
        # x_norm = 2*x/(W-1) - 1, y_norm = 2*y/(H-1) - 1
        cx_n = (2.0 * cx / (W - 1.0)) - 1.0
        cy_n = (2.0 * cy / (H - 1.0)) - 1.0
    else:
        # x_norm = 2*(x+0.5)/W - 1, y_norm = 2*(y+0.5)/H - 1
        cx_n = (2.0 * (cx + 0.5) / W) - 1.0
        cy_n = (2.0 * (cy + 0.5) / H) - 1.0

    c = torch.tensor([cx_n, cy_n], device=device, dtype=torch.float32)  # (2,)
    # t = c - R c (broadcast over A)
    Rc_x = theta[:, 0, 0] * c[0] + theta[:, 0, 1] * c[1]
    Rc_y = theta[:, 1, 0] * c[0] + theta[:, 1, 1] * c[1]
    theta[:, 0, 2] = c[0] - Rc_x
    theta[:, 1, 2] = c[1] - Rc_y

    grid = F.affine_grid(theta, size=(A, 1, H, W), align_corners=align_corners)  # (A,H,W,2)
    rot  = F.grid_sample(ref.expand(A, -1, -1, -1), grid,
                         mode="nearest", padding_mode="zeros",
                         align_corners=align_corners)  # (A,1,H,W)

    templates_bool = (rot[:, 0] > 0.5)  # (A,H,W) bool
    return templates_bool

@torch.no_grad()
def match_batched_iou(obj_mask_u8: torch.Tensor,
                      templates_bool: torch.Tensor,
                      base_dx: int, base_dy: int,
                      shifts=(-2, -1, 0, 1, 2),
                      obj_id=0,
                      device="cuda",
                      save_debug=False):
    """
    obj_mask_u8: (H,W) uint8 (0/255 or 0/1)
    templates_bool: (A,H,W) bool (precomputed rotations of REF about ref pivot)
    base_dx, base_dy: translation to align centers (ref_center - obj_center)
                      (since you're shifting OBJ into REF frame)
    shifts: local refinement shifts around base
    Returns: best_angle_index, best_dx, best_dy, best_iou
    """
    # print("base_dx, base_dy:", base_dx, base_dy)
    H, W = obj_mask_u8.shape
    obj = (obj_mask_u8 > 0).to(device=device, dtype=torch.bool)  # (H,W)
    # if save_debug:
    #     cv2.imwrite(f"obj_{obj_id}_original.png",
    #                 (obj.detach().cpu().numpy().astype(np.uint8) * 255))
    # base center alignment (ZERO PAD, no wrap)
    # obj0 = shift_zeropad(obj, base_dx, base_dy)  # (H,W)
    obj0 = obj

    if save_debug:
        cv2.imwrite(f"obj_{obj_id}_center_shifted.png",
                    (obj0.detach().cpu().numpy().astype(np.uint8) * 255))

    # stack refined shifts (S,H,W)
    shifted_list = []
    shift_pairs = []
    for dx in shifts:
        for dy in shifts:
            shifted_list.append(shift_zeropad(obj0, dx, dy))
            shift_pairs.append((dx, dy))

    shifted_objs = torch.stack(shifted_list, dim=0)  # (S,H,W)
    S = shifted_objs.shape[0]
    A = templates_bool.shape[0]

    shifted_objs = shifted_objs[:, None, :, :]        # (S,1,H,W)
    temps = templates_bool[None, :, :, :]             # (1,A,H,W)

    inter = (shifted_objs & temps).sum(dim=(2, 3), dtype=torch.int32)  # (S,A)
    union = (shifted_objs | temps).sum(dim=(2, 3), dtype=torch.int32)  # (S,A)
    iou = inter.float() / union.clamp_min(1).float()

    best = torch.argmax(iou).item()
    s_idx = best // A
    a_idx = best %  A

    best_iou = float(iou[s_idx, a_idx].item())
    dx_ref, dy_ref = shift_pairs[s_idx]
    best_dx = base_dx + dx_ref
    best_dy = base_dy + dy_ref
    return a_idx, best_dx, best_dy, best_iou
def warp_sprite(img, mask, cx, cy, rot_deg, out_h=H, out_w=W, flags=cv2.INTER_NEAREST):
    """
    img:   (h,w,C) uint8 or float32
    mask:  (h,w)   uint8 {0,255} or {0,1}
    cx,cy: center position in output image coordinates
    rot_deg: rotation in degrees (positive = CCW)
    Returns: warped_img, warped_mask (both in output size)
    """
    h, w = mask.shape[:2]
    # ys, xs = np.nonzero(mask > 0)
    # pts = np.column_stack([xs, ys]).astype(np.float32)
    # (px, py), (rw, rh), ang = cv2.minAreaRect(pts)
    # print("sprite center from minAreaRect:", px, py)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    obj_cnt, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("number of contours found in the mask:", len(obj_cnt))
    # rect = cv2.minAreaRect(obj_cnt[0])  # ((cx, cy), (w, h), angle)

    obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))[-1]  # the mask r cnn could give bad masks
    Mo = cv2.moments(obj_cnt)  # get center

    # # calculate x,y coordinate of center
    cX = int(Mo["m10"] / Mo["m00"])
    cY = int(Mo["m01"] / Mo["m00"])
    # Fix cY to be 111 
    cY = 111

    # print("mask center from moments:", cX, cY)
    # cX, cY = rect[0]
    # rotation about sprite center
    M = cv2.getRotationMatrix2D((cX, cY), rot_deg, 1.0)
    # translate so sprite center lands at (cx, cy)
    M[0, 2] += (cx - cX)
    M[1, 2] += (cy - cY)

    warped_img = cv2.warpAffine(
        img, M, (out_w, out_h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    warped_mask = cv2.warpAffine(
        mask, M, (out_w, out_h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped_img, warped_mask

