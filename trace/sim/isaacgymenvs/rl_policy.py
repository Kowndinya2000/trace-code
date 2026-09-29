
"""CLOSED-LOOP real-robot executor for the trained RL push policy (IROS26).

Loads the rl_games PPO checkpoint, queries the fort perception server for the
live scene (get_target_pose), builds the 90-D observation, and executes the
selected primitive on the UR5e via ur_rtde each step, logging TCP poses to
NNNNNN-*-tcp-poses.txt (old logs now in archive/experiment_logs/). Shares its
perception/template-matching helpers with orthographic_construction.py and
spiral_policy.py (copy-pasted — consolidate when touching).

STATUS (Aug 2026): this is the closed-loop pipeline being replaced by the
open-loop module (solve in the digital twin, replay trajectory). Kept for
baseline comparisons. Its policy-loading and primitive->moveL execution code
are the reference for the open-loop solver and executor.
"""
import json
import math
import isaacgym

import numpy as np
import os
import time
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.spiral_policy import get_ref_mask_info
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from gym import spaces

import isaacgymenvs.utils.torch_jit_utils as torch_utils
import torch
from tqdm import tqdm 

import pickle
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

from isaacgymenvs.orientation_utils import offset_concave_orientation, offset_cube_orientation, offset_half_cube_orientation, offset_rect_orientation, offset_triangle_orientation, \
                                            precompute_rotations, match_batched_iou, \
                                            get_ref_mask_info, crop_center_pad, base64_to_ndarray, warp_sprite
H, W = 224, 224
from utils.constants import NUM_ROTATION, REAL_WORKSPACE_LIMITS, REAL_PIXEL_SIZE
from utils.mtcs_utils import MCTSHelper 
mcts_helper = MCTSHelper(f"logs_grasp/snapshot-post-020000.reinforcement.pth", f"logs_grasp/grasp_model-89.pth", device="cuda" if torch.cuda.is_available() else "cpu")
        
# mcts_helper = MCTSHelper(f"logs_grasp/snapshot-post-020000.reinforcement.pth", device=device)

import numpy as np
# cube
ref_cube_mask, ref_cube_cx, ref_cube_cy = get_ref_mask_info("ref_cube_l515_cropped_mask_perfect_orientation.png", "ref_cube_l515_info.txt")
# ref_cylinder_mask, ref_cylinder_cx, ref_cylinder_cy = get_ref_mask_info("ref_cylinder_l515_cropped_mask_perfect_orientation.png", "ref_cylinder_l515_info.txt")
ref_concave_mask, ref_concave_cx, ref_concave_cy = get_ref_mask_info("ref_concave_l515_cropped_mask_perfect_orientation.png", "ref_concave_l515_info.txt")
ref_rect_mask, ref_rect_cx, ref_rect_cy = get_ref_mask_info("ref_rect_l515_cropped_mask_perfect_orientation.png", "ref_rect_l515_info.txt")
ref_half_cube_mask, ref_half_cube_cx, ref_half_cube_cy = get_ref_mask_info("ref_half_cube_l515_cropped_mask_perfect_orientation.png", "ref_half_cube_l515_info.txt")
ref_triangle_mask, ref_triangle_cx, ref_triangle_cy = get_ref_mask_info("ref_triangle_l515_cropped_mask_perfect_orientation.png", "ref_triangle_l515_info.txt")

angles = list(range(0, 360)) 
ref_cube_rot_template = precompute_rotations(torch.from_numpy(ref_cube_mask.astype(np.uint8)).to(device=device),
                                              angles_deg=list(range(0, 90)),
                                              cx=ref_cube_cx, cy=ref_cube_cy,
                                              device=device, align_corners=True)


ref_concave_rot_template = precompute_rotations(torch.from_numpy(ref_concave_mask.astype(np.uint8)).to(device=device),
                                              angles_deg=angles,
                                              cx=ref_concave_cx, cy=ref_concave_cy,
                                              device=device, align_corners=True)

ref_rect_rot_template = precompute_rotations(torch.from_numpy(ref_rect_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=list(range(0, 180)),
                                                cx=ref_rect_cx, cy=ref_rect_cy,
                                                device=device, align_corners=True)

ref_half_cube_rot_template = precompute_rotations(torch.from_numpy(ref_half_cube_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=list(range(0, 180)),
                                                cx=ref_half_cube_cx, cy=ref_half_cube_cy,
                                                device=device, align_corners=True)

ref_triangle_rot_template = precompute_rotations(torch.from_numpy(ref_triangle_mask.astype(np.uint8)).to(device=device),
                                                angles_deg=angles,
                                                cx=ref_triangle_cx, cy=ref_triangle_cy,
                                                device=device, align_corners=True)
COLOR_CHOICE = np.asarray(
        [
            # [78, 121, 167],  # blue
            [89, 161, 79],  # green
            # [156, 117, 95],  # brown
            [242, 142, 43],  # orange
            [237, 201, 72],  # yellow
            [186, 176, 172],  # gray
            [255, 87, 89],  # red
            # [176, 122, 161],  # purple
            # [118, 183, 178],  # cyan
            [255, 157, 167],  # pink
        ]
    )
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
        existing = (scene_seg > 0)

        intersection = np.logical_and(m, existing).sum()
        new_area = m.sum()

        overlap_ratio = intersection / new_area if new_area > 0 else 0.0
        # print(f"Object {inst_id} - class {cls}: overlap with existing scene = {overlap_ratio:.3f}")
        if overlap_ratio > 0.45:   # 45% occluded
            # print(f"Skipping object {inst_id}, overlap={overlap_ratio:.3f}")
            continue
        # ---------------------
        # scene_rgb[m] = rgb_w[m]
        if obj["obj_id"] == 0:
            color = (176, 122, 161)  # purple for target object
        else:
            color = COLOR_CHOICE[obj["class"] % len(COLOR_CHOICE)]  # color by class for distractors
        scene_rgb[m] = color  # Assign a random color from COLOR_CHOICE to the masked pixels
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
    print(f"Requesting target pose with payload: {payload}")
    response = requests.post(server_url, json=payload)
    objects = []
    #  [ 
    #       {"class":"2",   "cx": 0.528,  "cy": 0.005, "rot":  90},
    #  ]
    if response.status_code == 200:
        data = response.json()
        ## All poses are in sim base frame
        # data["left"] = {
        #     "object_poses": {
        #                 1: [],  # cube
        #                 2: [],  # cylinder
        #                 3: [],  # concave
        #                 4: [],  # rect
        #                 5: [],  # half-cube
        #                 6: []   # triangle
        #             },
        #     "target_obj_class": None
        # }
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

        # # Add objects of the same class as the target from the left camera (if any and not already added from right cam)
        # if left_target_object_class is not None:
        #     for pose_info in left_cam_poses[str(left_target_object_class)][1:]:
        #         # check if this pose is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
        #         duplicate = False
        #         for existing_obj in objects:
        #             # if existing_obj["class"] == int(left_target_object_class):
        #             dist = np.sqrt((existing_obj["cx"] - float(pose_info['cx']))**2 + (existing_obj["cy"] - float(pose_info['cy']))**2)
        #             if dist < 0.02:  # 2 cm threshold for duplicate
        #                 duplicate = True
        #                 break
        #         if not duplicate:
        #             objects.append({
        #                 "obj_id": counter, 
        #                 "class": int(left_target_object_class),
        #                 "cx": float(pose_info['cx']),
        #                 "cy": float(pose_info['cy']),
        #                 "angle": float(pose_info['angle']),
        #                 "mask": base64_to_ndarray(pose_info['mask'])
        #             })
        #             counter += 1


        # # Add the rest of the objects from the left camera belonging to non-target classes 
        # for obj_class, poses in left_cam_poses.items():
        #     if left_target_object_class is not None and obj_class == str(left_target_object_class):
        #         continue  # already added as target object
        #     for pose in poses:
        #         # check if this object (class+pose) is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
        #         duplicate = False
        #         for existing_obj in objects:
        #             # if existing_obj["class"] == int(obj_class):
        #             dist = np.sqrt((existing_obj["cx"] - float(pose['cx']))**2 + (existing_obj["cy"] - float(pose['cy']))**2)
        #             if dist < 0.02:  # 2 cm threshold for duplicate
        #                 duplicate = True
        #                 break
        #         if not duplicate:
        #             objects.append({
        #                 "obj_id": counter, # assign a new obj_id
        #                 "class": int(obj_class),
        #                 "cx": float(pose['cx']),
        #                 "cy": float(pose['cy']),
        #                 "angle": float(pose['angle']),
        #                 "mask": base64_to_ndarray(pose['mask'])
        #             })
        #             counter += 1
        
        
        # print("Target object classes:")
        # for obj in objects:
        #     if obj["class"] == 5:
        #         print(f"Target Object id {obj['obj_id']} - Class: {obj['class']}, cx: {obj['cx']}, cy: {obj['cy']}, angle: {obj['angle']}")
        print("Number of objects received from perception server:", len(objects))
        # 1478: 4 rect, 2 concave, 2 cube, 2 cylinder, 1 triangle
        class_counts = {
            1: 2,
            2: 2,
            3: 2,
            4: 4,
            5: 0,
            6: 1
        }
        # revised_object_list = []
        # observed_class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        # for object in objects:
        #     observed_class_counts[object["class"]] += 1
        #     if observed_class_counts[object["class"]] <= class_counts[object["class"]]:
        #         revised_object_list.append(object)
        #         if object["obj_id"] != 0:  # if not the target object, reassign obj_id based on the order in the revised list
        #             revised_object_list[-1]["obj_id"] = len(revised_object_list)  # reassign obj_id based on the revised list
        #     else:
        #         print(f"Warning: More objects of class {object['class']} than expected. Received {observed_class_counts[object['class']]} but expected at most {class_counts[object['class']]}.")
        # objects_w_masks = revised_object_list.copy()  # make a copy of the revised object list to modify with masks

        # for object in objects:
        #     print(f"Object ID: {object['obj_id']}, Class: {object['class']}, cx: {object['cx']}, cy: {object['cy']}, angle: {object['angle']}")
        objects_w_masks = objects.copy()  # make a copy of the original object list to modify with masks
        t_angle_matching_start = time.perf_counter()
        for obj_info in objects_w_masks:
            obj_mask = obj_info["mask"]
            # cv2.imwrite(f"obj_{obj_info['obj_id']}_original_mask.png", obj_mask)    
            obj_mask_cx, obj_mask_cy = calculate_center(obj_mask)
            # print("obj_mask_cx, obj_mask_cy:", obj_mask_cx, obj_mask_cy)
            # show the cx, cy on the original mask for debugging
            debug_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
            # cv2.circle(debug_mask, (obj_mask_cx, obj_mask_cy), 5, (0,0,255), -1)    
            # cv2.imwrite(f"obj_{obj_info['obj_id']}_debug_mask.png", debug_mask)

            cropped_obj_mask = crop_center_pad(obj_mask, obj_mask_cx, obj_mask_cy, out=360)
            
            # cv2.imwrite(f"obj_{obj_info['obj_id']}_cropped_mask.png", cropped_obj_mask)
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

                best_iou_cylinder = 0.0  # skip cylinder matching for now since it can be confused with cube and hurts overall performance
                if best_iou_cube > best_iou_cylinder:
                    obj_info["class"] = 1  # cube
                    obj_info["angle"] = offset_cube_orientation(matched_cube_angle)
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
                                            save_debug=False)
                matched_rect_angle = angles[a_idx_rect]
                obj_info["angle"] = offset_rect_orientation(matched_rect_angle)
                # print("================================================================================")
                # print("[CLASS 4] matched rect angle:", matched_rect_angle, "adjusted_angle:", obj_info["angle"])
                # print("================================================================================")
            elif obj_info["class"] == 3:  # concave
                a_idx_concave, dx, dy, best_iou_concave = match_batched_iou(obj_mask_torch_u8, ref_concave_rot_template,
                                            base_dx=int(ref_concave_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_concave_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-concave',
                                            save_debug=False)
                matched_concave_angle = angles[a_idx_concave]
                obj_info["angle"] = offset_concave_orientation(matched_concave_angle)
                # print("================================================================================")
                # print(f"[CLASS 3] CONCAVE_IOU: {best_iou_concave:.3f}, matched angle: {matched_concave_angle:.1f} DEG. CHOSEN AS [CONCAVE W.A. {obj_info['angle']:.1f}] DEG.")
                # print("================================================================================")
            elif obj_info["class"] == 5:  # half-cube
                a_idx_half_cube, dx, dy, best_iou_half_cube = match_batched_iou(obj_mask_torch_u8, ref_half_cube_rot_template,
                                            base_dx=int(ref_half_cube_cx - cropped_obj_mask_cx),
                                            base_dy=int(ref_half_cube_cy - cropped_obj_mask_cy),
                                            shifts=(-1,0,1),
                                            device="cuda",
                                            obj_id=f'{obj_info["obj_id"]}-half-cube',
                                            save_debug=False)
                matched_half_cube_angle = angles[a_idx_half_cube]
                obj_info["angle"] = offset_half_cube_orientation(matched_half_cube_angle)  # half-cube has same symmetry as rect for orientation purposes
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
                obj_info["angle"] = offset_triangle_orientation(matched_triangle_angle)
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

def build_two_step_plan(action_idx, total=0.01):
    # total = 1 cm per RL step
    s = total / 2.0  # 0.5 cm per phase
    ds = s / math.sqrt(2)

    B = action_idx.shape[0]
    device = action_idx.device
    d1 = torch.zeros((B,2), device=device)
    d2 = torch.zeros((B,2), device=device)

    # 0..3 cardinals: N,E,S,W
    dirs = torch.tensor([[0, s], [s, 0], [0, -s], [-s, 0]], device=device)
    is_card = action_idx < 4
    d1[is_card] = dirs[action_idx[is_card]]
    d2[is_card] = dirs[action_idx[is_card]]

    # diagonals: 4..15 (12 actions)
    is_diag = ~is_card
    a = action_idx[is_diag] - 4
    diag_dir = a // 3  # 0 NE,1 NW,2 SE,3 SW
    mode = a % 3       # 0 diag-diag, 1 grid1, 2 grid2

    # diag-diag: each phase is diagonal length 0.5 cm
    diag = torch.tensor([[ ds,  ds], [-ds,  ds], [ ds, -ds], [-ds, -ds]], device=device)

    # grid orders
    E = torch.tensor([ s, 0.0], device=device)
    W = torch.tensor([-s, 0.0], device=device)
    N = torch.tensor([0.0,  s], device=device)
    S = torch.tensor([0.0, -s], device=device)

    first_grid  = torch.stack([E, W, E, W], dim=0)  # NE,NW,SE,SW
    second_grid = torch.stack([N, N, S, S], dim=0)
    first_grid_sw  = second_grid
    second_grid_sw = first_grid

    idx = torch.nonzero(is_diag, as_tuple=False).squeeze(1)

    m0 = (mode == 0)
    d1[idx[m0]] = diag[diag_dir[m0]]
    d2[idx[m0]] = diag[diag_dir[m0]]

    m1 = (mode == 1)
    d1[idx[m1]] = first_grid[diag_dir[m1]]
    d2[idx[m1]] = second_grid[diag_dir[m1]]

    m2 = (mode == 2)
    d1[idx[m2]] = first_grid_sw[diag_dir[m2]]
    d2[idx[m2]] = second_grid_sw[diag_dir[m2]]

    return d1, d2

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print(cfg_dict["task"]["sim"]["dt"], "Simulation dt")

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)
    print("Num envs:", cfg.task.env.numEnvs)
    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.test,
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        return envs

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    args = {
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma' : None
    }
    checkpoint = args['checkpoint']
    if "_" not in checkpoint.split("/")[-1]:
        epoch = "best"
    else:
        epoch = checkpoint.split("/")[-1].split("_")[-3]
    print('epoch:', epoch)

    checkpoint_name_parts = args['checkpoint'].split('/')
    checkpoint_result = checkpoint_name_parts[-3]
    save_ckpt_result_name = f'results/{checkpoint_result.split("_")[0]}/eval_results_{checkpoint_result}_ep_{epoch}.pkl'
    print('save checkpoint_name:', save_ckpt_result_name)
    # exit(0)
    print('Started to play')
    player = runner.create_player()
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        player.restore(args['checkpoint'])

    is_deterministic = player.is_deterministic
    obses = player.env_reset(player.env)
    batch_size = 1
    batch_size = player.get_batch_size(obses, batch_size)
    # all_env_ids = torch.arange(cfg.task.env.numEnvs, device=cfg.sim_device)
    # player.env.reset_idx(all_env_ids)
    
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
    fp = open("000510-n-tcp-poses.txt", "w")
    # fp = open("tcp_poses_test.txt", "w")
    for req_count in range(35):
        fp.write(f"{current_tcp_pose}\n")
        objects = get_target_pose(current_tcp_pose, req_count=req_count)
        if objects is not None:
            scene_rgb, scene_depth, scene_seg = compose_scene(objects, ref_db)
            cv2.imwrite(f"scene_rgb_{req_count}.png", scene_rgb.astype(np.uint8))
            cv2.imwrite(f"scene_depth_{req_count}.png", scene_depth.astype(np.uint16))  # save raw depth for debugging
            cv2.imwrite(f"scene_seg_{req_count}.png", scene_seg.astype(np.uint8))  # save raw seg for debugging
            
            # exit(0)
            if req_count > 0 and req_count < 15:
                q_value = 0                
            else:
                q_start = time.time()
                # Single Image Analysis
                # q_value, best_pix_ind, grasp_predictions, raw_grasp_predictions = mcts_helper.get_grasp_q(
                #     scene_rgb, scene_depth / (1000 * 100), scene_seg, post_checking=True
                # )
                q_value, best_pix_ind, grasp_predictions = mcts_helper.get_grasp_q_parallel_x16(
                        scene_rgb, scene_depth/ (1000 * 100), scene_seg, post_checking=True
                )
                print(f"Max grasp Q value: {q_value}, time: {1000*(time.time() - q_start)} ms. Best rotation angle index (x of 16):",(best_pix_ind[0]))
            if q_value > 0.75:  
                # if the best predicted grasp is good enough, execute it directly without RL policy
                # print the cx, cy in world coordinates
                print("Best grasp pixel indices (x of 16, y of 16):", best_pix_ind[1:3])
                if objects[0]["obj_id"] == 0:
                    print("target exists in the scene!!")
                    # For cube, half-cube and cylinder, discard the grasp point by the GPN and consider the center of the target object as the grasp point
                    small_obj_classes = [1, 2, 5]  # cube, cylinder, half-cube
                    if int(objects[0]["class"]) in small_obj_classes:
                        print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a small object. Using its center as the grasp point instead of GPN prediction.")
                        grasp_cx, grasp_cy = objects[0]["cx"], objects[0]["cy"]
                        # grasp_cx, grasp_cy = pix2world(best_pix_ind[1], best_pix_ind[2])
                    else:
                        grasp_cx, grasp_cy = pix2world(best_pix_ind[1], best_pix_ind[2])
                        print(f"Target object class {class_to_obj_name[int(objects[0]['class'])]} is a large object. Using GPN predicted grasp point at ({grasp_cx}, {grasp_cy}).")
                    
                    grasp_cx, grasp_cy = grasp_cy, -grasp_cx  # swap and negate to convert from sim to real robot coordinates
                    confirm_orient = input(f"\n============================================\nPredicted best rotation angle index (x of 16) is {best_pix_ind[0]}, corresponding to {best_pix_ind[0] * (360.0 / NUM_ROTATION):.1f} degrees. Do you want to use this angle, x,y offset? (y/n)")
                    confirm_rotation_angle = confirm_orient.split()[0]
                    if confirm_rotation_angle.lower() == 'y':
                        best_rotation_angle = np.deg2rad(best_pix_ind[0].item() * (360.0 / NUM_ROTATION))
                    else:
                        best_rotation_angle = np.deg2rad(int(confirm_rotation_angle) * (360.0 / NUM_ROTATION))
                    
                    grasp_cx_offset = int(confirm_orient.split()[1])/100.0 if len(confirm_orient.split()) > 1 else 0
                    grasp_cy_offset = int(confirm_orient.split()[2])/100.0 if len(confirm_orient.split()) > 2 else 0
                    if abs(grasp_cx_offset) > 0.03:
                        grasp_cx_offset = 0.0
                    if abs(grasp_cy_offset) > 0.03:
                        grasp_cy_offset = 0.0
                    print(f"\n============================================================\nApplying user-provided offsets - grasp_cx_offset: {grasp_cx_offset}, grasp_cy_offset: {grasp_cy_offset}")
                    input2 = input("Do you want to proceed with the grasp execution using the above parameters? (y/n)")
                    if input2.lower() != 'y':
                        print("Aborting grasp execution as per user request.")
                        exit(0)  
                    grasp_cx += grasp_cy_offset
                    grasp_cy -= grasp_cx_offset

                    # best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / NUM_ROTATION))
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
                segm_values = [255, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                bboxes = np.zeros((11, 8), dtype=np.float32)
                filled = 0

                x0 = WORKSPACE_LIMITS[0, 0]
                y0 = WORKSPACE_LIMITS[1, 0]
                s  = PIXEL_SIZE

                for segm_value in segm_values:
                    mask = (scene_seg == segm_value)
                    ys, xs = np.nonzero(mask)          # pixel indices
                    if xs.size == 0:
                        continue

                    pts = np.column_stack([xs, ys]).astype(np.float32)  # (N,2) as (x,y)
                    rect = cv2.minAreaRect(pts)
                    box = cv2.boxPoints(rect).astype(np.float32)        # (4,2)

                    # vectorized pix2world
                    box[:, 0] = x0 + box[:, 0] * s
                    box[:, 1] = y0 + box[:, 1] * s

                    bboxes[filled] = box.reshape(-1)
                    filled += 1
                    if filled == 11:
                        break
                padded_obj_centers = [
                    [0.850, -0.0],
                    [0.850, -0.108],
                    [0.850, -0.208],
                    [0.850, -0.30],
                    [0.850, 0.108],
                    [0.850, 0.208],
                    [0.850, 0.30],
                    [0.750, 0.30]
                ]
                # pad missing boxes (vectorized)
                print("num of identified objects:", filled)
                needed = 11 - filled
                if needed > 0:
                    centers = np.asarray(padded_obj_centers[:needed], dtype=np.float32)  # (needed,2)
                    w = 0.045 / 2
                    x = centers[:, 0]; y = centers[:, 1]
                    padded = np.stack([x-w, y-w, x+w, y-w, x+w, y+w, x-w, y+w], axis=1)
                    # print("padded boxes shape:", padded.shape)

                    bboxes[filled:filled+needed] = padded
                # bboxes is (11, 8) np.float32
                
                bboxes_t = torch.from_numpy(bboxes).to(device=device)  # zero-copy on CPU to GPU copy
                current_tcp_pose = rtde_r.getActualTCPPose()
                gripper_pos = current_tcp_pose[:2]  # (x, y) in real world coordinates
                # real to sim coordinates: swap and negate y
                gripper_pos[0], gripper_pos[1] = -gripper_pos[1], gripper_pos[0]
                
                # gripper_pos is small -> make torch tensor directly
                gripper_t = torch.tensor(gripper_pos, device=device, dtype=torch.float32)

                # flatten + concat
                obs_buf_tensor = torch.cat(
                    [bboxes_t.view(-1), gripper_t],
                    dim=0
                ).unsqueeze(0)   # (1, 90)

                actions = player.get_action(obs_buf_tensor, is_deterministic)
                print("Action tensor from policy:", actions, "shape:", actions.shape, "values:", actions.cpu().numpy())
                d1, d2 = build_two_step_plan(actions.unsqueeze(0), total=0.02)
                # print("Planned d1:", d1, "d2:", d2)
                # print("Shapes d1:", d1.shape, "d2:", d2.shape) # [1,2]
                d1 = d1.squeeze(0).cpu().numpy()  # (2,)
                d2 = d2.squeeze(0).cpu().numpy()  # (2,)
                # convert to real world coordinates (swap and negate back)
                d1[0], d1[1] = d1[1], -d1[0]
                d2[0], d2[1] = d2[1], -d2[0]
                # print("Planned d1 in real world coords:", d1, "d2 in real world coords:", d2)
                current_tcp_pose = rtde_r.getActualTCPPose()
                # print("Current TCP Pose before move:", current_tcp_pose)
                target_pose_d1 = current_tcp_pose.copy()
                target_pose_d1[0] += d1[0]
                target_pose_d1[1] += d1[1]
                rtde_c.moveL(target_pose_d1, tool_vel, tool_acc)
                current_tcp_pose = rtde_r.getActualTCPPose()
                # print("Current TCP Pose after d1 move:", current_tcp_pose)
                target_pose_d2 = current_tcp_pose.copy()
                target_pose_d2[0] += d2[0]
                target_pose_d2[1] += d2[1]
                rtde_c.moveL(target_pose_d2, tool_vel, tool_acc)


            # exit(0)

    fp.close()   

if __name__ == "__main__":
    launch_rlg_hydra()

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