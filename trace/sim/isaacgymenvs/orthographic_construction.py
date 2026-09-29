"""Orthographic scene reconstruction from the DUAL eye-on-hand perception.

get_target_pose(data): ingests left/right perception-server output
(object_poses per class + target class + base64 masks), de-duplicates
cross-view detections, recovers each object's orientation by GPU IoU
template-matching against the ref_*_l515 reference masks (match_batched_iou),
and composes the orthographic RGB/depth/seg scene (compose_scene).

Part of the closed-loop pipeline; the open-loop twin builder replaces this
with single-external-camera perception (PMBS-style), but the template-match
orientation recovery here is directly reusable.
"""
import numpy as np
import os
import time

from isaacgymenvs.spiral_policy import get_ref_mask_info

from tqdm import tqdm 

root_dir = os.getcwd()
import math
from isaacgymenvs.video_streaming_server_multi_cam import calculate_center
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np  
import time
from scipy.spatial.transform import Rotation as R
import cv2 
from utils.constants import NUM_ROTATION
# import torch

from isaacgymenvs.orientation_utils import offset_concave_orientation, offset_cube_orientation, offset_half_cube_orientation, offset_rect_orientation, offset_triangle_orientation, \
                                            precompute_rotations, match_batched_iou, \
                                            get_ref_mask_info, crop_center_pad, base64_to_ndarray, warp_sprite
import torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        rgb_w, mask_w = warp_sprite(rgb_ref, mask_ref, cx, cy, rot, flags=cv2.INTER_LINEAR)

        # Warp depth too (use linear is fine since it's smooth; nearest also ok if constant)
        depth_w, _ = warp_sprite(depth_ref.astype(np.float32), mask_ref, cx, cy, rot, flags=cv2.INTER_LINEAR)
        depth_w = depth_w.astype(np.float32)

        # Alpha blend RGB using mask
        m = (mask_w > 0)
        existing = (scene_seg > 0)

        intersection = np.logical_and(m, existing).sum()
        new_area = m.sum()

        overlap_ratio = intersection / new_area if new_area > 0 else 0.0
        print(f"Object {inst_id} - class {cls}: overlap with existing scene = {overlap_ratio:.3f}")
        if overlap_ratio > 0.45:   # 45% occluded
            print(f"Skipping object {inst_id}, overlap={overlap_ratio:.3f}")
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
def dist_check(obj1, obj2, threshold=0.02):
    dist = np.sqrt((float(obj1["cx"]) - float(obj2["cx"]))**2 + (obj1["cy"] - float(obj2["cy"]))**2)
    return dist < threshold

def get_target_pose(data):
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
    objects = []
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
            print("Adding the target object from left camera data since it's not detected in right camera.")
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
    
    # Add the rest of the objects from the right camera (excluding the target class which is already added)        
    for class_id in right_cam_poses:
        if class_id == str(right_target_object_class):
            continue  # already added all poses for this class
        for pose in right_cam_poses[class_id]:
            duplicate = False
            for existing_obj in objects:
                if existing_obj["class"] != 5 and int(class_id) != 5:
                    duplicate = dist_check(existing_obj, pose, threshold=0.02)  # 3 cm threshold for duplicate for non-half-cube targets
                    if duplicate:
                        break
                else:
                    duplicate = dist_check(existing_obj, pose, threshold=0.02)  # 2.5 cm threshold for duplicate for half-cube targets
                    if duplicate:
                        break

                # # if existing_obj["class"] == int(class_id):
                # dist = np.sqrt((existing_obj["cx"] - float(pose['cx']))**2 + (existing_obj["cy"] - float(pose['cy']))**2)
                # if dist < 0.02:  # 2 cm threshold for duplicate
                #     duplicate = True
                #     break
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
    # Add objects of the same class as the target from the right camera first (if any)
    if right_target_object_class is not None:
        for pose_info in right_cam_poses[str(right_target_object_class)][1:]:
            duplicate = False
            for existing_obj in objects:
                if existing_obj["class"] != 5 and right_target_object_class != 5:
                    duplicate = dist_check(existing_obj, pose_info, threshold=0.02)  # 3 cm threshold for duplicate for non-half-cube targets
                    if duplicate:
                        break
                else:
                    duplicate = dist_check(existing_obj, pose_info, threshold=0.02)  # 2.5 cm threshold for duplicate for half-cube targets
                    if duplicate:
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


    # # Add objects of the same class as the target from the left camera (if any and not already added from right cam)
    if left_target_object_class is not None:
        for pose_info in left_cam_poses[str(left_target_object_class)][1:]:
            # check if this pose is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
            duplicate = False
            for existing_obj in objects:
                if existing_obj["class"] != 5 and int(left_target_object_class) != 5:
                    duplicate = dist_check(existing_obj, pose_info, threshold=0.02)  # 3 cm threshold for duplicate for non-half-cube targets
                    if duplicate:
                        break
                else:
                    duplicate = dist_check(existing_obj, pose_info, threshold=0.02)  # 2.5 cm threshold for duplicate for half-cube targets
                    if duplicate:
                        break
                # if existing_obj["class"] == int(left_target_object_class):
                # dist = np.sqrt((existing_obj["cx"] - float(pose_info['cx']))**2 + (existing_obj["cy"] - float(pose_info['cy']))**2)
                # if dist < 0.02:  # 2 cm threshold for duplicate
                #     duplicate = True
                #     break
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


    # # Add the rest of the objects from the left camera belonging to non-target classes 
    for obj_class, poses in left_cam_poses.items():
        if left_target_object_class is not None and obj_class == str(left_target_object_class):
            continue  # already added as target object
        for pose in poses:
            # check if this object (class+pose) is already in the list from the right camera, if so skip to avoid duplicates. Use a simple distance threshold for pose similarity.
            duplicate = False
            for existing_obj in objects:
                if existing_obj["class"] != 5 and obj_class != 5:
                    duplicate = dist_check(existing_obj, pose, threshold=0.02)  # 3 cm threshold for duplicate for non-half-cube targets
                    if duplicate:
                        break
                else:
                    duplicate = dist_check(existing_obj, pose, threshold=0.02)  # 2.5 cm threshold for duplicate for half-cube targets
                    if duplicate:
                        break
                # if existing_obj["class"] == int(obj_class):
                # dist = np.sqrt((existing_obj["cx"] - float(pose['cx']))**2 + (existing_obj["cy"] - float(pose['cy']))**2)
                # if dist < 0.02:  # 2 cm threshold for duplicate
                #     duplicate = True
                #     break
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
        obj_mask = (obj_info["mask"] > 0).astype(np.uint8) * 255
        # cv2.imwrite(f"obj_{obj_info['obj_id']}_original_mask.png", obj_mask)    
        # cv2.imshow(f"obj_{obj_info['obj_id']}_class_{obj_info['class']}_original_mask.png", obj_mask)
        # cv2.waitKey(0)
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
