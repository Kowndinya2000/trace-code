
# Date: 
import math
import numpy as np
import os
import time

import torch
root_dir = os.getcwd()
import torch.nn.functional as F
import math
from isaacgymenvs.video_streaming_server_multi_cam import calculate_center
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np  
import time
from robotiq_gripper import RobotiqGripper
import requests 
from scipy.spatial.transform import Rotation as R
import cv2 
from utils.constants import NUM_ROTATION
# import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 224, 224
from utils.constants import NUM_ROTATION, REAL_WORKSPACE_LIMITS, REAL_PIXEL_SIZE
from utils.mtcs_utils import MCTSHelper 
mcts_helper = MCTSHelper(f"logs_grasp/snapshot-post-020000.reinforcement.pth", f"logs_grasp/grasp_model-89.pth", device="cuda" if torch.cuda.is_available() else "cpu")
        
# mcts_helper = MCTSHelper(f"logs_grasp/snapshot-post-020000.reinforcement.pth", device=device)
import base64
import numpy as np
def get_ref_mask_info(mask_path, info_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    cx, cy = calculate_center(mask)
    # with open(info_path, "r") as f:
    #     info = json.load(f)
    # angle = info["angle"]
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

# cube
ref_cube_mask, ref_cube_cx, ref_cube_cy = get_ref_mask_info("ref_cube_l515_cropped_mask_perfect_orientation.png", "ref_cube_l515_info.txt")
ref_cylinder_mask, ref_cylinder_cx, ref_cylinder_cy = get_ref_mask_info("ref_cylinder_l515_cropped_mask_perfect_orientation.png", "ref_cylinder_l515_info.txt")
ref_concave_mask, ref_concave_cx, ref_concave_cy = get_ref_mask_info("ref_concave_l515_cropped_mask_perfect_orientation.png", "ref_concave_l515_info.txt")
ref_rect_mask, ref_rect_cx, ref_rect_cy = get_ref_mask_info("ref_rect_l515_cropped_mask_perfect_orientation.png", "ref_rect_l515_info.txt")
ref_half_cube_mask, ref_half_cube_cx, ref_half_cube_cy = get_ref_mask_info("ref_half_cube_l515_cropped_mask_perfect_orientation.png", "ref_half_cube_l515_info.txt")
ref_triangle_mask, ref_triangle_cx, ref_triangle_cy = get_ref_mask_info("ref_triangle_l515_cropped_mask_perfect_orientation.png", "ref_triangle_l515_info.txt")
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

angles = list(range(0, 360)) 
ref_cube_rot_template = precompute_rotations(torch.from_numpy(ref_cube_mask.astype(np.uint8)).to(device=device),
                                              angles_deg=angles,
                                              cx=ref_cube_cx, cy=ref_cube_cy,
                                              device=device, align_corners=True)

ref_cylinder_rot_template = precompute_rotations(torch.from_numpy(ref_cylinder_mask.astype(np.uint8)).to(device=device),
                                              angles_deg=angles,
                                              cx=ref_cylinder_cx, cy=ref_cylinder_cy,
                                              device=device, align_corners=True)

ref_concave_rot_template = precompute_rotations(torch.from_numpy(ref_concave_mask.astype(np.uint8)).to(device=device),
                                              angles_deg=angles,
                                              cx=ref_concave_cx, cy=ref_concave_cy,
                                              device=device, align_corners=True)

ref_rect_rot_template = precompute_rotations(torch.from_numpy(ref_rect_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=angles,
                                                cx=ref_rect_cx, cy=ref_rect_cy,
                                                device=device, align_corners=True)

ref_half_cube_rot_template = precompute_rotations(torch.from_numpy(ref_half_cube_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=angles,
                                                cx=ref_half_cube_cx, cy=ref_half_cube_cy,
                                                device=device, align_corners=True)

ref_triangle_rot_template = precompute_rotations(torch.from_numpy(ref_triangle_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=angles,
                                                cx=ref_triangle_cx, cy=ref_triangle_cy,
                                                device=device, align_corners=True)


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
def warp_sprite(img, mask, cx, cy, rot_deg, out_h=H, out_w=W):
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
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    warped_mask = cv2.warpAffine(
        mask, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped_img, warped_mask

def compose_scene(objects, ref_db, bg_rgb=(0,0,0), bg_depth=0.0):
    """
    objects: list of dicts:
        {"class": "circle", "cx": 120, "cy": 80, "rot": 30, "obj_id": 0}  # obj_id is zero for the target obj 
        or use class_id int if you want.
    ref_db: dict[class] -> {"rgb":..., "depth":..., "seg":...}
        rgb:   (h,w,3) uint8
        depth: (h,w) float32 (or uint16) for reference sprite depth
        seg:   (h,w) uint8/uint16, nonzero where object exists (or just a mask image)
    Returns:
        scene_rgb: (224,224,3) uint8
        scene_depth:(224,224) float32
        scene_seg:  (224,224) uint16  (instance ids)
    """
    scene_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    scene_rgb[:] = bg_rgb

    scene_depth = np.full((H, W), bg_depth, dtype=np.float32)

    # instance id segmentation (0=background)
    scene_seg = np.zeros((H, W), dtype=np.uint16)

    for inst_id, obj in enumerate(objects, start=1):
        cls = obj["class"]
        cx, cy, rot = obj["cx"], obj["cy"], obj["angle"]
        
        cx, cy = world2pix(cx, cy)
        rgb_ref   = ref_db[cls]["rgb"]
        depth_ref = ref_db[cls]["depth"]
        seg_ref   = ref_db[cls]["seg"]
        # if cls == 6:
        #     print("triangle obj[cy]:", obj["cy"])
        #     cy, cx = 224-cx, 224-cy 
        # Build a binary mask from the reference seg (nonzero = object)
        mask_ref = (seg_ref > 0).astype(np.uint8) * 255

        # Warp RGB + mask
        rgb_w, mask_w = warp_sprite(rgb_ref, mask_ref, cx, cy, rot)

        # Warp depth too (use linear is fine since it's smooth; nearest also ok if constant)
        depth_w, _ = warp_sprite(depth_ref.astype(np.float32), mask_ref, cx, cy, rot)
        depth_w = depth_w.astype(np.float32)

        # Alpha blend RGB using mask
        m = (mask_w > 0)
        scene_rgb[m] = rgb_w[m]

        # Depth: since all objects "same depth", just write depth where object lands.
        # If you want constant depth for all objects, replace depth_w[m] with a scalar.
        scene_depth[m] = depth_w[m]
        # Segmentation: write instance id (later objects overwrite earlier ones if overlap)
        if obj["obj_id"] == 0:
            scene_seg[m] = 255  # target object gets instance id 255

        else:
            scene_seg[m] = 50 + (10 * obj["obj_id"])  # use the obj_id for instance segmentation

    return scene_rgb, scene_depth, scene_seg

# ------------------ Example usage ------------------
# Suppose you loaded your 6 reference sprites like:
ref_db = {
    1:   {
                "rgb": cv2.imread("obj-ref-images/color_cube.png"), 
                "depth": cv2.imread("obj-ref-images/depth_cube.png", cv2.IMREAD_UNCHANGED), 
                "seg": cv2.imread("obj-ref-images/segm_cube.png", cv2.IMREAD_UNCHANGED)
            },
    2: {
                "rgb": cv2.imread("obj-ref-images/color_cylinder.png"),   
                "depth": cv2.imread("obj-ref-images/depth_cylinder.png", cv2.IMREAD_UNCHANGED), 
                "seg": cv2.imread("obj-ref-images/segm_cylinder.png", cv2.IMREAD_UNCHANGED)
            },
    3: {
                "rgb": cv2.imread("obj-ref-images/color_concave.png"),   
                "depth": cv2.imread("obj-ref-images/depth_concave.png", cv2.IMREAD_UNCHANGED), 
                "seg": cv2.imread("obj-ref-images/segm_concave.png", cv2.IMREAD_UNCHANGED)
            },
    4: {  
                "rgb": cv2.imread("obj-ref-images/color_rect.png"),
                "depth": cv2.imread("obj-ref-images/depth_rect.png", cv2.IMREAD_UNCHANGED),
                "seg": cv2.imread("obj-ref-images/segm_rect.png", cv2.IMREAD_UNCHANGED)
            },
    5: {
                "rgb": cv2.imread("obj-ref-images/color_half-cube.png"),
                "depth": cv2.imread("obj-ref-images/depth_half-cube.png", cv2.IMREAD_UNCHANGED),
                "seg": cv2.imread("obj-ref-images/segm_half-cube.png", cv2.IMREAD_UNCHANGED)
            },
    6: {
                "rgb": cv2.imread("obj-ref-images/color_triangle.png"),
                "depth": cv2.imread("obj-ref-images/depth_triangle.png", cv2.IMREAD_UNCHANGED),
                # "seg": cv2.imread("obj-ref-images/segm_triangle.png", cv2.IMREAD_UNCHANGED)
                "seg": cv2.imread("obj-ref-images/segm_triangle_centered.png", cv2.IMREAD_UNCHANGED)
            },
}

WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])
PIXEL_SIZE = 0.002
pix2world = lambda cx, cy: (WORKSPACE_LIMITS[0,0] + cx*PIXEL_SIZE, WORKSPACE_LIMITS[1,0] + cy*PIXEL_SIZE)
pix2world_real = lambda cx, cy: (REAL_WORKSPACE_LIMITS[0][0] + cx*REAL_PIXEL_SIZE, REAL_WORKSPACE_LIMITS[1][0] + cy*REAL_PIXEL_SIZE)
world2pix = lambda x, y: (int((x - WORKSPACE_LIMITS[0,0]) / PIXEL_SIZE), int((y - WORKSPACE_LIMITS[1,0]) / PIXEL_SIZE))
# objects = [
#     {"class":"circle",   "cx": 60,  "cy": 120, "rot":  0},
#     {"class":"square",   "cx": 120, "cy": 120, "rot": 45},
#     {"class":"circle",   "cx": 170, "cy": 70,  "rot": 20},
# ]

class_to_obj_name = {
                        1: "cube",
                        2: "cylinder",
                        3: "concave",
                        4: "rect",
                        5: "half-cube",
                        6: "triangle"
                    }

def get_target_pose(tcp_pose, req_count):
    server_url = "http://localhost:7777/stream"
    payload = {
        "id" : f"{req_count:04d}",
        "tcp_pose": tcp_pose
    }
    response = requests.post(server_url, json=payload)
    objects = []
    #  [ 
    #       {"class":"2",   "cx": 0.528,  "cy": 0.005, "rot":  90},
    #  ]
    if response.status_code == 200:
        data = response.json()
        ## All poses are in sim base frame
        data["left"] = {
            "object_poses": {
                        1: [],  # cube
                        2: [],  # cylinder
                        3: [],  # concave
                        4: [],  # rect
                        5: [],  # half-cube
                        6: []   # triangle
                    },
            "target_obj_class": None
        }
        left_target_object_class = data.get("left", {}).get("target_obj_class", None)
        left_cam_poses = data.get("left", {}).get("object_poses", {})
        
        right_target_object_class = data.get("right", {}).get("target_obj_class", None)
        right_cam_poses = data.get("right", {}).get("object_poses", {})
        # Add the target 
        if right_target_object_class is not None:
            objects.append({
                "obj_id": 0, # 0 means target object
                "class": right_target_object_class,
                "cx": float(right_cam_poses[str(right_target_object_class)][0]['cx']),
                "cy": float(right_cam_poses[str(right_target_object_class)][0]['cy']),
                "angle": float(right_cam_poses[str(right_target_object_class)][0]['angle']),
                "mask": base64_to_ndarray(right_cam_poses[str(right_target_object_class)][0]['mask'])
            })

        if left_target_object_class is not None:
            if right_target_object_class is not None:
                pass  # already added from right cam data, do not add again to avoid duplicates
            else:
                objects.append({
                    "obj_id": 0, 
                    "class": left_target_object_class,
                    "cx": float(left_cam_poses[str(left_target_object_class)][0]['cx']),
                    "cy": float(left_cam_poses[str(left_target_object_class)][0]['cy']),
                    "angle": float(left_cam_poses[str(left_target_object_class)][0]['angle']),
                    "mask": base64_to_ndarray(left_cam_poses[str(left_target_object_class)][0]['mask'])
                })
        if left_target_object_class is None and right_target_object_class is None:
            print("Warning: No target object detected by either camera.")

        counter = 1
        
        # Add objects of the same class as the target from the right camera first (if any)
        if right_target_object_class is not None:
            for pose_info in right_cam_poses[str(right_target_object_class)][1:]:
                duplicate = False
                for existing_obj in objects:
                    # if existing_obj["class"] == int(right_target_object_class):
                    dist = np.sqrt((existing_obj["cx"] - float(pose_info['cx']))**2 + (existing_obj["cy"] - float(pose_info['cy']))**2)
                    if dist < 0.02:  # 2 cm threshold for duplicate
                        duplicate = True
                        break
                if duplicate:
                    continue  # skip adding this object since it's likely a duplicate of the target object already added
                objects.append({
                    "obj_id": counter, 
                    "class": right_target_object_class,
                    "cx": float(pose_info['cx']),
                    "cy": float(pose_info['cy']),
                    "angle": float(pose_info['angle']),
                    "mask": base64_to_ndarray(pose_info['mask'])
                })
                counter += 1

        # Add the rest of the objects from the right camera (excluding the target class which is already added)        
        for class_id in right_cam_poses:
            if class_id == str(right_target_object_class):
                continue  # already added all poses for this class
            for pose in right_cam_poses[class_id]:
                duplicate = False
                for existing_obj in objects:
                    # if existing_obj["class"] == int(class_id):
                    dist = np.sqrt((existing_obj["cx"] - float(pose['cx']))**2 + (existing_obj["cy"] - float(pose['cy']))**2)
                    if dist < 0.02:  # 2 cm threshold for duplicate
                        duplicate = True
                        break
                if duplicate:
                    continue  # skip adding this object since it's likely a duplicate of one already added from the left cam
                objects.append({
                    "obj_id": counter, 
                    "class": int(class_id),
                    "cx": float(pose['cx']),
                    "cy": float(pose['cy']),
                    "angle": float(pose['angle']),
                    "mask": base64_to_ndarray(pose['mask'])
                })
                counter += 1

        # Add objects of the same class as the target from the left camera (if any and not already added from right cam)
        if left_target_object_class is not None:
            for pose_info in left_cam_poses[str(left_target_object_class)][1:]:
                # check if this pose is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
                duplicate = False
                for existing_obj in objects:
                    # if existing_obj["class"] == int(left_target_object_class):
                    dist = np.sqrt((existing_obj["cx"] - float(pose_info['cx']))**2 + (existing_obj["cy"] - float(pose_info['cy']))**2)
                    if dist < 0.02:  # 2 cm threshold for duplicate
                        duplicate = True
                        break
                if not duplicate:
                    objects.append({
                        "obj_id": counter, 
                        "class": int(left_target_object_class),
                        "cx": float(pose_info['cx']),
                        "cy": float(pose_info['cy']),
                        "angle": float(pose_info['angle']),
                        "mask": base64_to_ndarray(pose_info['mask'])
                    })
                    counter += 1


        # Add the rest of the objects from the left camera belonging to non-target classes 
        for obj_class, poses in left_cam_poses.items():
            if left_target_object_class is not None and obj_class == str(left_target_object_class):
                continue  # already added as target object
            for pose in poses:
                # check if this object (class+pose) is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
                duplicate = False
                for existing_obj in objects:
                    # if existing_obj["class"] == int(obj_class):
                    dist = np.sqrt((existing_obj["cx"] - float(pose['cx']))**2 + (existing_obj["cy"] - float(pose['cy']))**2)
                    if dist < 0.02:  # 2 cm threshold for duplicate
                        duplicate = True
                        break
                if not duplicate:
                    objects.append({
                        "obj_id": counter, # assign a new obj_id
                        "class": int(obj_class),
                        "cx": float(pose['cx']),
                        "cy": float(pose['cy']),
                        "angle": float(pose['angle']),
                        "mask": base64_to_ndarray(pose['mask'])
                    })
                    counter += 1
        # print("Target object classes:")
        # for obj in objects:
        #     if obj["class"] == 5:
        #         print(f"Target Object id {obj['obj_id']} - Class: {obj['class']}, cx: {obj['cx']}, cy: {obj['cy']}, angle: {obj['angle']}")
        print("Number of objects received from perception server:", len(objects))
        # for object in objects:
        #     print(f"Object ID: {object['obj_id']}, Class: {object['class']}, cx: {object['cx']}, cy: {object['cy']}, angle: {object['angle']}")

        objects_w_masks = objects.copy()  # make a copy of the objects list to modify with masks
        t_angle_matching_start = time.perf_counter()
        for obj_info in objects_w_masks:
            obj_mask = obj_info["mask"]
            cv2.imwrite(f"obj_{obj_info['obj_id']}_original_mask.png", obj_mask)    
            obj_mask_cx, obj_mask_cy = calculate_center(obj_mask)
            # print("obj_mask_cx, obj_mask_cy:", obj_mask_cx, obj_mask_cy)
            # show the cx, cy on the original mask for debugging
            debug_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_mask, (obj_mask_cx, obj_mask_cy), 5, (0,0,255), -1)    
            cv2.imwrite(f"obj_{obj_info['obj_id']}_debug_mask.png", debug_mask)

            cropped_obj_mask = crop_center_pad(obj_mask, obj_mask_cx, obj_mask_cy, out=360)
            
            cv2.imwrite(f"obj_{obj_info['obj_id']}_cropped_mask.png", cropped_obj_mask)
            obj_mask_torch_u8 = torch.from_numpy(cropped_obj_mask.astype(np.uint8)).to(device)
            cropped_obj_mask_cx, cropped_obj_mask_cy = calculate_center(cropped_obj_mask)
            if obj_info["class"] == 1:  # cube
                a_idx_cube, dx, dy, best_iou_cube = match_batched_iou(obj_mask_torch_u8, ref_cube_rot_template,
                                            base_dx=int(ref_cube_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_cube_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-cube',
                                            save_debug=False)
                matched_cube_angle = angles[a_idx_cube]

                # a_idx_cylinder, dx, dy, best_iou_cylinder = match_batched_iou(obj_mask_torch_u8, ref_cylinder_rot_template,
                #                             base_dx=int(ref_cylinder_cx - cropped_obj_mask_cx),
                #                             base_dy=int(ref_cylinder_cy - cropped_obj_mask_cy),
                #                             shifts=(-1,0,1),
                #                             device="cuda",
                #                             obj_id=f'{obj_info["obj_id"]}-cylinder',
                #                             save_debug=False)
                # matched_cylinder_angle = angles[a_idx_cylinder]
                best_iou_cylinder = 0.0  # skip cylinder matching for now since it can be confused with cube and hurts overall performance
                if best_iou_cube > best_iou_cylinder:
                    obj_info["class"] = 1  # cube
                    obj_info["angle"] = matched_cube_angle
                    # obj_info["angle"] = -obj_info["angle"]
                    # print(f"[CLASS 1] CUBE_IOU: {best_iou_cube:.3f}, CYLINDER_IOU: {best_iou_cylinder:.3f}, matched angle: {matched_cube_angle:.1f} DEG. CHOSEN AS [CUBE W.A. {obj_info['angle']:.1f}] DEG.")
                # else:
                #     obj_info["class"] = 2  # cylinder
                #     obj_info["angle"] = matched_cylinder_angle + (90 - ref_cylinder_angle)
                    # print(f"[CLASS 2] CUBE_IOU: {best_iou_cube:.3f}, CYLINDER_IOU: {best_iou_cylinder:.3f}, matched angle: {matched_cylinder_angle:.1f} DEG. CHOSEN AS [CYLINDER W.A. {obj_info['angle']:.1f}] DEG.")


            elif obj_info["class"] == 4:  # rect
                a_idx_rect, dx, dy, best_iou_rect = match_batched_iou(obj_mask_torch_u8, ref_rect_rot_template,
                                            base_dx=int(ref_rect_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_rect_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-rect',
                                            save_debug=True)
                matched_rect_angle = angles[a_idx_rect]

                a_idx_concave, dx, dy, best_iou_concave = match_batched_iou(obj_mask_torch_u8, ref_concave_rot_template,
                                            base_dx=int(ref_concave_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_concave_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-concave',
                                            save_debug=False)
                matched_concave_angle = angles[a_idx_concave]

                if best_iou_rect > best_iou_concave:
                    obj_info["class"] = 4  # rect
                    obj_info["angle"] = matched_rect_angle
                    # obj_info["angle"] = -obj_info["angle"] 
                    # print(f"[CLASS 4] RECT_IOU: {best_iou_rect:.3f}, CONCAVE_IOU: {best_iou_concave:.3f}, matched angle: {matched_rect_angle:.1f} DEG. CHOSEN AS [RECT W.A. {obj_info['angle']:.1f}] DEG.")
                # print(f"[CLASS 4] RECT_IOU: {best_iou_rect:.3f}, matched angle: {matched_rect_angle:.1f} DEG. CHOSEN AS [RECT W.A. {obj_info['angle']:.1f}] DEG.")
                else:
                    obj_info["class"] = 3  # concave
                    obj_info["angle"] = matched_concave_angle 
                    # obj_info["angle"] = -obj_info["angle"]  
                    # print(f"[CLASS 3] RECT_IOU: {best_iou_rect:.3f}, CONCAVE_IOU: {best_iou_concave:.3f}, matched angle: {matched_concave_angle:.1f} DEG. CHOSEN AS [CONCAVE W.A. {obj_info['angle']:.1f}] DEG.")

            elif obj_info["class"] == 3:  # concave
                a_idx_concave, dx, dy, best_iou_concave = match_batched_iou(obj_mask_torch_u8, ref_concave_rot_template,
                                            base_dx=int(ref_concave_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_concave_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-concave',
                                            save_debug=False)
                matched_concave_angle = angles[a_idx_concave]
                
                if matched_concave_angle > 337.5 and matched_concave_angle <= 360:
                    # print("matched_concave_angle in range (337.5, 22.5], adding + 180 deg")
                    obj_info["angle"] = matched_concave_angle # checked 

                elif matched_concave_angle >= 0 and matched_concave_angle <= 22.5:
                    # print("matched_concave_angle in range (337.5, 22.5], adding + 180 deg")
                    obj_info["angle"] = matched_concave_angle # checked

                elif matched_concave_angle > 22.5 and matched_concave_angle <= 67.5:
                    # print("matched_concave_angle in range (22.5, 67.5], adding + 90 deg")
                    obj_info["angle"] = matched_concave_angle - 90


                elif matched_concave_angle > 67.5 and matched_concave_angle <= 112.5:
                    # print("matched_concave_angle in range (67.5, 112.5], keeping angle as is")
                    obj_info["angle"] = matched_concave_angle + 180 # checked
                
                elif matched_concave_angle > 112.5 and matched_concave_angle <= 157.5:
                    # print("matched_concave_angle in range (112.5, 157.5], subtracting 90 deg")
                    obj_info["angle"] = matched_concave_angle - 90 

                elif matched_concave_angle > 157.5 and matched_concave_angle <= 202.5:
                    obj_info["angle"] = matched_concave_angle - 180

                elif matched_concave_angle > 202.5 and matched_concave_angle <= 247.5:
                    obj_info["angle"] = matched_concave_angle - 360
                
                elif matched_concave_angle > 247.5 and matched_concave_angle <= 292.5:
                    obj_info["angle"] = matched_concave_angle - 360
                
                elif matched_concave_angle > 292.5 and matched_concave_angle <= 337.5:
                    obj_info["angle"] = matched_concave_angle - 450

                # obj_info["angle"] = -obj_info["angle"]
                # print(f"[CLASS 3] CONCAVE_IOU: {best_iou_concave:.3f}, matched angle: {matched_concave_angle:.1f} DEG. CHOSEN AS [CONCAVE W.A. {obj_info['angle']:.1f}] DEG.")
                
            elif obj_info["class"] == 5:  # half-cube
                # a_idx_half_cube, dx, dy, best_iou_half_cube = match_batched_iou(obj_mask_torch_u8, ref_half_cube_rot_template,
                #                             base_dx=int(ref_half_cube_cx - cropped_obj_mask_cx),
                #                             base_dy=int(ref_half_cube_cy - cropped_obj_mask_cy),
                #                             shifts=(-1,0,1),
                #                             device="cuda",
                #                             obj_id=f'{obj_info["obj_id"]}-half-cube',
                #                             save_debug=False)
                # matched_half_cube_angle = angles[a_idx_half_cube]
                # obj_info["angle"] = matched_half_cube_angle 
                # obj_info["angle"] = -obj_info["angle"]
                # print(f"[CLASS 5] HALF_CUBE_IOU: {best_iou_half_cube:.3f}, matched angle: {matched_half_cube_angle:.1f} DEG. CHOSEN AS [HALF_CUBE W.A. {obj_info['angle']:.1f}] DEG.")
                pass

            elif obj_info["class"] == 6:  # triangle
                a_idx_triangle, dx, dy, best_iou_triangle = match_batched_iou(obj_mask_torch_u8, ref_triangle_rot_template,
                                            base_dx=int(ref_triangle_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_triangle_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-triangle',
                                            save_debug=False)
                matched_triangle_angle = angles[a_idx_triangle]
                if matched_triangle_angle > 337.5 and matched_triangle_angle <= 360:
                    # print("matched_triangle_angle in range (337.5, 22.5], adding + 180 deg")
                    obj_info["angle"] = matched_triangle_angle + 180 # checked 

                elif matched_triangle_angle >= 0 and matched_triangle_angle <= 22.5:
                    # print("matched_triangle_angle in range (337.5, 22.5], adding + 180 deg")
                    obj_info["angle"] = matched_triangle_angle + 180 # checked 

                elif matched_triangle_angle > 22.5 and matched_triangle_angle <= 67.5:
                    # print("matched_triangle_angle in range (22.5, 67.5], adding + 90 deg")
                    obj_info["angle"] = matched_triangle_angle + 90 # checked


                elif matched_triangle_angle > 67.5 and matched_triangle_angle <= 112.5:
                    # print("matched_triangle_angle in range (67.5, 112.5], keeping angle as is")
                    obj_info["angle"] = matched_triangle_angle # checked
                
                elif matched_triangle_angle > 112.5 and matched_triangle_angle <= 157.5:
                    # print("matched_triangle_angle in range (112.5, 157.5], subtracting 90 deg")
                    obj_info["angle"] = matched_triangle_angle - 90 # checked

                elif matched_triangle_angle > 157.5 and matched_triangle_angle <= 202.5:
                    obj_info["angle"] = matched_triangle_angle - 180 # checked
                
                elif matched_triangle_angle > 202.5 and matched_triangle_angle <= 247.5:
                    obj_info["angle"] = matched_triangle_angle - 360 # checked
                
                elif matched_triangle_angle > 247.5 and matched_triangle_angle <= 292.5:
                    obj_info["angle"] = matched_triangle_angle - 360 # checked
                
                elif matched_triangle_angle > 292.5 and matched_triangle_angle <= 337.5:
                    obj_info["angle"] = matched_triangle_angle - 450

                #  = (matched_triangle_angle)
                # obj_info["angle"] = -obj_info["angle"]
                # print(f"[CLASS 6] TRIANGLE_IOU: {best_iou_triangle:.3f}, matched angle: {matched_triangle_angle:.1f} DEG. CHOSEN AS [TRIANGLE W.A. {obj_info['angle']:.1f}] DEG.")
            del obj_info["mask"]  # remove mask from the info dict to save memory, since we have already extracted the necessary pose info from it
        
        print(f"Time taken for angle matching: {1000*(time.perf_counter() - t_angle_matching_start):.3f} milliseconds")
        return objects_w_masks
        # Target obj class from (left, right) cams: (None, 1)
        # print(f"Target obj class from (left, right) cams: ({  left_target_object_class}, {right_target_object_class})")
        # print(f"Left Camera Tgt Object Poses: {left_cam_poses['2']}")
        # Right Camera Tgt Object Poses: [{'angle': 90.0, 'cx': '0.528', 'cy': '0.005', 'cz': '0.040'}]
        # print(f"Right Camera Tgt Object Poses: {right_cam_poses['2']}")         
    else:
        print(f"Error: Received status code {response.status_code} from server.")
        return None



def main():
    tool_vel = 0.2
    tool_acc = 0.2 
    joint_vel = 0.5
    joint_acc = 0.5
    _ip = os.environ.get("TRACE_ROBOT_IP", "192.168.1.102") 
    min_z_clearance = 0.018

    # Connect to the UR5e robot
    rtde_c = RTDEControl(_ip)
    rtde_r = RTDEReceive(_ip, use_upper_range_registers=False) 
    gripper = RobotiqGripper(os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"), 63352)
    print("Trying to connect to the gripper...")
    # gripper.connect()
    # gripper.activate(auto_calibrate=True)   # <-- important

    # print("Connected to the gripper.")
    # gripper.open(80, 120)
    # print("Gripper opened.")
    # gripper.close_and_wait_for_grasp_pos(80, 120)
    # print("Gripper closed to grasp position.")
    # gripper.close_and_wait_for_pos(80, 120)
    # print("Gripper closed to full position.")
    home_joints = [65.88, -118.58, 121.07, -91.79, -89.84, 155.98]
    home_joints_rad = [x*(np.pi/180) for x in home_joints]
    rtde_c.moveJ(home_joints_rad, joint_vel, joint_acc)

    gripper.connect()
    print("Connected to the gripper.")

    # gripper.activate(auto_calibrate=True)
    # print("Activated. Calibrated range:", gripper.get_min_position(), gripper.get_max_position())

    speed = 80
    force = 120

    # Open fully (calibrated open)
    pos, status = gripper.open_and_wait_for_pos(speed, force)
    print("Opened:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))

    # Close to "grasp" (pick a value between min and max)
    grasp_pos = int(0.7 * gripper.get_max_position())  # example
    pos, status = gripper.move_and_wait_for_pos(grasp_pos, speed, force)
    print("Closed to grasp:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))

    # Close fully (calibrated close)
    pos, status = gripper.close_and_wait_for_pos(speed, force)
    print("Closed:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))
    # exit(0)
    current_tcp_pose = rtde_r.getActualTCPPose()
    # print("Current TCP Pose:", current_tcp_pose)
    # exit(0)
    radius_shrink_rate = 0.0025
    theta = (93 * np.pi) / 180  # Start at 92 degrees in radians
    min_radius = 0.06 # 0.07
    theta_increment = 0.01 # radians
    orbit_loops_after_reach = 3
    in_orbit_mode = False
    radius = 0
    target_cx, target_cy = None, None
    objects = None
    reset_once = 0
    for req_count in range(200):
        if req_count == 0 or req_count % 20 == 0:
            objects = get_target_pose(current_tcp_pose, req_count=req_count)
            scene_rgb, scene_depth, scene_seg = compose_scene(objects, ref_db)
            # cv2.imwrite(f"scene_rgb_{req_count}.png", scene_rgb.astype(np.uint8))
            # cv2.imwrite(f"scene_depth_{req_count}.png", scene_depth.astype(np.uint16))  # save raw depth for debugging
            cv2.imwrite(f"scene_seg_{req_count}.png", scene_seg.astype(np.uint8))  # save raw seg for debugging
            q_start = time.time()
            # Single Image Analysis
            # q_value, best_pix_ind, grasp_predictions, raw_grasp_predictions = mcts_helper.get_grasp_q(
            #     scene_rgb, scene_depth / (1000 * 100), scene_seg, post_checking=True
            # )

            q_value, best_pix_ind, grasp_predictions = mcts_helper.get_grasp_q_parallel_x16(
                    scene_rgb, scene_depth/ (1000 * 100), scene_seg, post_checking=True
            )
            print(f"Max grasp Q value: {q_value}, time: {1000*(time.time() - q_start)} ms. Best rotation angle index (x of 16):",(best_pix_ind[0]))
            # if q_value > 0.75:
                # if reset_once < 4:
                #     reset_once += 1
                    # q_value = 0
    
        else:
            q_value = 0

        if q_value > 0.75:  
            print(f"[SEQ] Max grasp Q value: {q_value}, time: {1000*(time.time() - q_start)} ms. Best rotation angle index (x of 16):",(best_pix_ind[0]))
        
            # if the best predicted grasp is good enough, execute it directly without RL policy
            # print the cx, cy in world coordinates
            print("Best grasp pixel indices (x of 16, y of 16):", best_pix_ind[1:3])
            if objects[0]["obj_id"] == 0:
                print("target exists in the scene!!")
                # For cube, half-cube and cylinder, discard the grasp point by the GPN and consider the center of the target object as the grasp point
                small_obj_classes = [1, 2, 5]  # cube, cylinder, half-cube
                if int(objects[0]["class"]) in small_obj_classes:
                    print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a small object. Using its center as the grasp point instead of GPN prediction.")
                    grasp_cx, grasp_cy = objects[0]["cx"] - 0.01, objects[0]["cy"]
                else:
                    grasp_cx, grasp_cy = world2pix(best_pix_ind[1], best_pix_ind[2])
                    print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a large object. Using GPN predicted grasp point.")
                
                grasp_cx, grasp_cy = grasp_cy, -grasp_cx  # swap and negate to convert from sim to real robot coordinates
                best_rotation_angle = np.deg2rad(best_pix_ind[0].item() * (360.0 / NUM_ROTATION))
                grasp_orientation = [1.0, 0.0]
                heightmap_rotation_angle = best_rotation_angle
                if heightmap_rotation_angle > np.pi / 2 and heightmap_rotation_angle < np.pi * 3 / 2:
                    heightmap_rotation_angle = heightmap_rotation_angle - np.pi
                elif heightmap_rotation_angle >= np.pi * 3 / 2 and heightmap_rotation_angle <= np.pi * 2:
                    heightmap_rotation_angle = heightmap_rotation_angle - np.pi * 2

                heightmap_rotation_angle = -heightmap_rotation_angle + np.pi / 2
                
                
                # ref_yaw = 0.0   # or keep a running last_yaw and use that instead

                # def wrap_pi(a): # map any angle to [-pi, pi]
                #     return (a + np.pi) % (2*np.pi) - np.pi

                # cand1 = heightmap_rotation_angle
                # cand2 = heightmap_rotation_angle + np.pi

                # # pick whichever is closer to ref_yaw
                # if abs(wrap_pi(cand2 - ref_yaw)) < abs(wrap_pi(cand1 - ref_yaw)):
                #     heightmap_rotation_angle = cand2

                
                tool_rotation_angle = heightmap_rotation_angle / 2
                tool_orientation = np.asarray(
                [
                    grasp_orientation[0] * np.cos(tool_rotation_angle) -
                    grasp_orientation[1] * np.sin(tool_rotation_angle),
                    grasp_orientation[0] * np.sin(tool_rotation_angle) +
                    grasp_orientation[1] * np.cos(tool_rotation_angle),
                    0.0]) * np.pi

                print("tool orientation (x,y,z):", tool_orientation)



                grasp_tcp_pose = [grasp_cx, grasp_cy, 0.0, tool_orientation[0], tool_orientation[1], 0.0]

                min_z_clearance = 0.018

                # Lift the arm up first to avoid collision with the objects in the scene
                current_pose = rtde_r.getActualTCPPose()
                print("Current Pose: ", current_pose)
                current_pose_lift = current_pose.copy()
                current_pose_lift[2] = min_z_clearance + 0.07
                rtde_c.moveL(current_pose_lift, tool_vel, tool_acc)

                # Move to the grasp pose above the target object and orient the gripper according to the predicted grasp orientation
                grasp_orient_top_pose = [grasp_cx, grasp_cy, min_z_clearance + 0.07, tool_orientation[0], tool_orientation[1], 0.0]
                rtde_c.moveL(grasp_orient_top_pose, tool_vel, tool_acc)
                gripper.open_and_wait_for_pos(80, 120) 
                
                # Move down to the grasp pose
                go_down = rtde_r.getActualTCPPose()
                go_down[2] = min_z_clearance
                rtde_c.moveL(go_down, tool_vel, tool_acc)

                # Close the fingers to grasp the target object
                # gripper.close_and_wait_for_pos(80, 120)
                # grasp_pos = int(0.7 * gripper.get_max_position())  # example
                grasp_pos = int(0.9 * gripper.get_max_position())  # example
                pos, status = gripper.move_and_wait_for_pos(grasp_pos, speed, force)
                print("Closed to grasp:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))


                # Carefully lift the arm up after grasping the object to avoid dropping it
                lift_pose = rtde_r.getActualTCPPose()
                lift_pose[2] = 0.085
                rtde_c.moveL(lift_pose, tool_vel, tool_acc)
                exit(0)
            else:
                print("Do not attempt to grasp since the target object is not detected!!")

        else:     
            if req_count == 0 or req_count % 20 == 0:
                current_tcp_pose = rtde_r.getActualTCPPose()
                for obj in objects:
                    if obj["obj_id"] == 0: # target object
                        tgt_cx_sim, tgt_cy_sim, tgt_rot = obj["cx"], obj["cy"], obj["angle"]
                        target_cx, target_cy = tgt_cy_sim, -tgt_cx_sim  # swap and negate to convert from sim to real robot coordinates
                        print("Initial target position (real robot coordinates):", target_cx, target_cy)
                        break            
                if target_cx is not None and target_cy is not None: 
                    radius = np.sqrt((target_cx - current_tcp_pose[0])**2 + (target_cy - current_tcp_pose[1])**2)
                    print("Initial radius set to:", round(radius*100, 2), "cms")
            else:
                current_tcp_pose = rtde_r.getActualTCPPose()
                if target_cx is not None and target_cy is not None: 
                    dist_to_target = np.sqrt((target_cx - current_tcp_pose[0])**2 + (target_cy - current_tcp_pose[1])**2)
                else:
                    print("Target object not detected in the scene!!")
                    exit(0)            

                if dist_to_target <= min_radius:
                    in_orbit_mode = True 

                if not in_orbit_mode:
                    radius = max(radius - radius_shrink_rate, min_radius)
                else:
                    radius = min_radius
                    if theta >= 2 * np.pi * orbit_loops_after_reach:
                        print("Reached the target point, exiting orbit mode.")
                        break
                print("updated radius:", round(radius*100, 2), "cms, distance to target:", round(dist_to_target*100, 2), "cms")
                theta += theta_increment / radius

                new_gripper_pos = current_tcp_pose.copy()
                new_gripper_pos[0] = target_cx + (radius * np.cos(theta))
                new_gripper_pos[1] = target_cy + (radius * np.sin(theta)) 

                print("radius:", round(radius*100, 2), "cms")
                print(f"2D Gripper Pos(X,Y) | Current: [{round(current_tcp_pose[0], 2), round(current_tcp_pose[1], 2)}], \
                    Next: [{round(new_gripper_pos[0], 2), round(new_gripper_pos[1], 2)}]")
                rtde_c.moveL(new_gripper_pos, tool_vel, tool_acc)
            

if __name__ == "__main__":
    main()

    # tool_vel = 0.2
    # tool_acc = 0.2 
    # joint_vel = 0.5
    # joint_acc = 0.5
    # _ip = os.environ.get("TRACE_ROBOT_IP", "192.168.1.102") 
    # min_z_clearance = 0.018

    # # Connect to the UR5e robot
    # rtde_c = RTDEControl(_ip)
    # rtde_r = RTDEReceive(_ip, use_upper_range_registers=False) 
    # gripper = RobotiqGripper(os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"), 63352)
    # print("Trying to connect to the gripper...")
    # # gripper.connect()
    # # gripper.activate(auto_calibrate=True)   # <-- important

    # # print("Connected to the gripper.")
    # # gripper.open(80, 120)
    # # print("Gripper opened.")
    # # gripper.close_and_wait_for_grasp_pos(80, 120)
    # # print("Gripper closed to grasp position.")
    # # gripper.close_and_wait_for_pos(80, 120)
    # # print("Gripper closed to full position.")

    # gripper.connect()
    # print("Connected to the gripper.")

    # # gripper.activate(auto_calibrate=True)
    # # print("Activated. Calibrated range:", gripper.get_min_position(), gripper.get_max_position())

    # speed = 80
    # force = 120

    # # Open fully (calibrated open)
    # pos, status = gripper.open_and_wait_for_pos(speed, force)
    # print("Opened:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))

    # # Close to "grasp" (pick a value between min and max)
    # grasp_pos = int(0.7 * gripper.get_max_position())  # example
    # pos, status = gripper.move_and_wait_for_pos(grasp_pos, speed, force)
    # print("Closed to grasp:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))

    # # Close fully (calibrated close)
    # pos, status = gripper.close_and_wait_for_pos(speed, force)
    # print("Closed:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))
    # # exit(0)
    # home_joints = [65.88, -118.58, 121.07, -91.79, -89.84, 155.98]
    # home_joints_rad = [x*(np.pi/180) for x in home_joints]
    # rtde_c.moveJ(home_joints_rad, joint_vel, joint_acc)
    # current_tcp_pose = rtde_r.getActualTCPPose()
    # print("Current TCP Pose:", current_tcp_pose)
    # # exit(0)

    # for req_count in range(1):
    #     objects = get_target_pose(current_tcp_pose, req_count=req_count)
    #     # print("Objects received from perception server:", objects)
    #     if objects is not None:
    #         scene_rgb, scene_depth, scene_seg = compose_scene(objects, ref_db)
    #         segm_values = [255, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    #         bboxes = np.zeros((11, 8), dtype=np.float32)
    #         filled = 0

    #         x0 = WORKSPACE_LIMITS[0, 0]
    #         y0 = WORKSPACE_LIMITS[1, 0]
    #         s  = PIXEL_SIZE

    #         for segm_value in segm_values:
    #             mask = (scene_seg == segm_value)
    #             ys, xs = np.nonzero(mask)          # pixel indices
    #             if xs.size == 0:
    #                 continue

    #             pts = np.column_stack([xs, ys]).astype(np.float32)  # (N,2) as (x,y)
    #             rect = cv2.minAreaRect(pts)
    #             box = cv2.boxPoints(rect).astype(np.float32)        # (4,2)

    #             # vectorized pix2world
    #             box[:, 0] = x0 + box[:, 0] * s
    #             box[:, 1] = y0 + box[:, 1] * s

    #             bboxes[filled] = box.reshape(-1)
    #             filled += 1
    #             if filled == 11:
    #                 break
    #         padded_obj_centers = [
    #             [0.850, -0.0],
    #             [0.850, -0.108],
    #             [0.850, -0.208],
    #             [0.850, -0.30],
    #             [0.850, 0.108],
    #             [0.850, 0.208],
    #             [0.850, 0.30],
    #             [0.750, 0.30]
    #         ]
    #         # pad missing boxes (vectorized)
    #         needed = 11 - filled
    #         if needed > 0:
    #             centers = np.asarray(padded_obj_centers[:needed], dtype=np.float32)  # (needed,2)
    #             w = 0.045 / 2
    #             x = centers[:, 0]; y = centers[:, 1]
    #             padded = np.stack([x-w, y-w, x+w, y-w, x+w, y+w, x-w, y+w], axis=1)
    #             bboxes[filled:filled+needed] = padded
    #         # bboxes is (11, 8) np.float32
            
    #         bboxes_t = torch.from_numpy(bboxes).to(device=device)  # zero-copy on CPU to GPU copy
    #         current_tcp_pose = rtde_r.getActualTCPPose()
    #         gripper_pos = current_tcp_pose[:2]  # (x, y) in real world coordinates
    #         # real to sim coordinates: swap and negate y
    #         gripper_pos[0], gripper_pos[1] = -gripper_pos[1], gripper_pos[0]
            
    #         # gripper_pos is small -> make torch tensor directly
    #         gripper_t = torch.tensor(gripper_pos, device=device, dtype=torch.float32)

    #         # flatten + concat
    #         obs_buf_tensor = torch.cat(
    #             [bboxes_t.view(-1), gripper_t],
    #             dim=0
    #         ).unsqueeze(0)   # (1, 90)

            # now bboxes is (11,8) float32

            # print("segm unique values:", np.unique(scene_seg))

            # segm_values = [255, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
            # bboxes = []
            # for segm_value in segm_values:
            # # tgt_segm_value = 255  # target object has segm value 255 in our setup
            #     mask_u8 = (scene_seg == segm_value).astype(np.uint8) * 255                        
            #     # find contours
            #     cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cnt = max(cnts, key=cv2.contourArea)   # pick largest component (if multiple)

            #     # rotated min-area rect
            #     rect = cv2.minAreaRect(cnt)           # ((cx, cy), (w, h), angle)
            #     box = cv2.boxPoints(rect)             # (4,2) float
            #     box = box.astype(np.int32)
            #     # pix2world to get the box corners in world coordinates
            #     for i in range(4):
            #         box[i, 0], box[i, 1] = pix2world(box[i, 0], box[i, 1])
            #     box = box.reshape(-1)  # flatten to (8,)
            #     bboxes.append(box)

            # padded_obj_centers = [
            #     [0.850, -0.0],
            #     [0.850, -0.108],
            #     [0.850, -0.208],
            #     [0.850, -0.30],
            #     [0.850, 0.108],
            #     [0.850, 0.208],
            #     [0.850, 0.30],
            #     [0.750, 0.30]
            # ]
            # ctr = 0
            # for i in range(len(bboxes), 11):
            #     width = 0.045/2
            #     padded_bbox = np.asarray([
            #         padded_obj_centers[ctr][0] - width, padded_obj_centers[ctr][1] - width,
            #         padded_obj_centers[ctr][0] + width, padded_obj_centers[ctr][1] - width,
            #         padded_obj_centers[ctr][0] + width, padded_obj_centers[ctr][1] + width,
            #         padded_obj_centers[ctr][0] - width, padded_obj_centers[ctr][1] + width,
            #     ])
            #     bboxes.append(padded_bbox.reshape(-1))  # flatten to (8,)
            #     ctr += 1
            # assert len(bboxes) == 11, f"Expected 11 bounding boxes (1 target + 10 distractors), but got {len(bboxes)}"
            # bboxes = np.array(bboxes, dtype=np.float32)  # (11, 8)
            # assert bboxes.shape == (11, 8), f"Expected bboxes shape (11, 8), but got {bboxes.shape}"
            # current_tcp_pose = rtde_r.getActualTCPPose()
            # gripper_pos = current_tcp_pose[:2]  # (x, y) in real world coordinates
            # # real to sim coordinates: swap and negate y
            # gripper_pos[0], gripper_pos[1] = -gripper_pos[1], gripper_pos[0]
            # obs_buf = np.concatenate([bboxes.flatten(), gripper_pos])  # shape: (11*8 + 2,) = (90,)
            # obs_buf_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(device)  # (1, 90)

            # save the scene for visualization
    #         cv2.imwrite(f"scene_rgb_{req_count}.png", scene_rgb.astype(np.uint8))
    #         cv2.imwrite(f"scene_depth_{req_count}.png", scene_depth.astype(np.uint16))  # save raw depth for debugging
    #         cv2.imwrite(f"scene_seg_{req_count}.png", scene_seg.astype(np.uint8))  # save raw seg for debugging
    #         # cv2.imshow("scene_rgb", scene_rgb)
    #         # cv2.imshow("scene_depth", scene_depth)  # normalize for visualization
    #         # cv2.imshow("scene_seg", scene_seg * 255)  # scale for visualization
    #         # cv2.waitKey(0)
    #     # print("scene depth unique values:", np.unique(scene_depth))
    #         exit(0)
    #         q_start = time.time()
    #         # Single Image Analysis
    #         q_value_seq, best_pix_ind_seq, grasp_predictions_seq, raw_grasp_predictions_seq = mcts_helper.get_grasp_q(
    #             scene_rgb, scene_depth / (1000 * 100), scene_seg, post_checking=True
    #         )
    #         print(f"[SEQ] Max grasp Q value: {q_value_seq}, time: {1000*(time.time() - q_start)} ms. Best rotation angle index (x of 16):",(best_pix_ind_seq[0]))
    #         # print the cx, cy in world coordinates
    #         print("Best grasp pixel indices (x of 16, y of 16):", best_pix_ind_seq[1:3])
    #         if objects[0]["obj_id"] == 0:
    #             print("target exists in the scene!!")
    #             # For cube, half-cube and cylinder, discard the grasp point by the GPN and consider the center of the target object as the grasp point
    #             small_obj_classes = [1, 2, 5]  # cube, cylinder, half-cube
    #             if int(objects[0]["class"]) in small_obj_classes:
    #                 print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a small object. Using its center as the grasp point instead of GPN prediction.")
    #                 grasp_cx, grasp_cy = objects[0]["cx"], objects[0]["cy"]
    #             else:
    #                 grasp_cx, grasp_cy = world2pix(best_pix_ind_seq[1], best_pix_ind_seq[2])
    #                 print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a large object. Using GPN predicted grasp point.")
                
    #             grasp_cx, grasp_cy = grasp_cy, -grasp_cx  # swap and negate to convert from sim to real robot coordinates
    #             best_rotation_angle = np.deg2rad(best_pix_ind_seq[0].item() * (360.0 / NUM_ROTATION))
    #             grasp_orientation = [1.0, 0.0]
    #             heightmap_rotation_angle = best_rotation_angle
    #             if heightmap_rotation_angle > np.pi / 2 and heightmap_rotation_angle < np.pi * 3 / 2:
    #                 heightmap_rotation_angle = heightmap_rotation_angle - np.pi
    #             elif heightmap_rotation_angle >= np.pi * 3 / 2 and heightmap_rotation_angle <= np.pi * 2:
    #                 heightmap_rotation_angle = heightmap_rotation_angle - np.pi * 2

    #             heightmap_rotation_angle = -heightmap_rotation_angle + np.pi / 2
    #             tool_rotation_angle = heightmap_rotation_angle / 2
    #             tool_orientation = np.asarray(
    #             [
    #                 grasp_orientation[0] * np.cos(tool_rotation_angle) -
    #                 grasp_orientation[1] * np.sin(tool_rotation_angle),
    #                 grasp_orientation[0] * np.sin(tool_rotation_angle) +
    #                 grasp_orientation[1] * np.cos(tool_rotation_angle),
    #                 0.0]) * np.pi

    #             print("tool orientation (x,y,z):", tool_orientation)
    #             grasp_tcp_pose = [grasp_cx, grasp_cy, 0.0, tool_orientation[0], tool_orientation[1], 0.0]

    #             min_z_clearance = 0.018

    #             # Lift the arm up first to avoid collision with the objects in the scene
    #             current_pose = rtde_r.getActualTCPPose()
    #             print("Current Pose: ", current_pose)
    #             current_pose_lift = current_pose.copy()
    #             current_pose_lift[2] = min_z_clearance + 0.07
    #             rtde_c.moveL(current_pose_lift, tool_vel, tool_acc)

    #             # Move to the grasp pose above the target object and orient the gripper according to the predicted grasp orientation
    #             grasp_orient_top_pose = [grasp_cx, grasp_cy, min_z_clearance + 0.07, tool_orientation[0], tool_orientation[1], 0.0]
    #             rtde_c.moveL(grasp_orient_top_pose, tool_vel, tool_acc)
    #             gripper.open_and_wait_for_pos(80, 120) 
                
    #             # Move down to the grasp pose
    #             go_down = rtde_r.getActualTCPPose()
    #             go_down[2] = min_z_clearance
    #             rtde_c.moveL(go_down, tool_vel, tool_acc)

    #             # Close the fingers to grasp the target object
    #             # gripper.close_and_wait_for_pos(80, 120)
    #             grasp_pos = int(0.7 * gripper.get_max_position())  # example
    #             pos, status = gripper.move_and_wait_for_pos(grasp_pos, speed, force)
    #             print("Closed to grasp:", pos, status, "OBJ:", gripper._get_var(gripper.OBJ), "FLT:", gripper._get_var(gripper.FLT))


    #             # Carefully lift the arm up after grasping the object to avoid dropping it
    #             lift_pose = rtde_r.getActualTCPPose()
    #             lift_pose[2] = 0.085
    #             rtde_c.moveL(lift_pose, tool_vel, tool_acc)

    #         else:
    #             print("Do not attempt to grasp since the target object is not detected!!")
    
    # for req_count in range(10):
    #     get_target_pose(current_tcp_pose, req_count=req_count)
    #     next_tcp_pose = current_tcp_pose.copy()
    #     next_tcp_pose[1] -= 0.015  # Move forward direction by 1 cm
    #     # rtde_c.moveL(next_tcp_pose, tool_vel, tool_acc)
    #     current_tcp_pose = rtde_r.getActualTCPPose()