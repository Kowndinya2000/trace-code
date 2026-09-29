

import os
import json
import shutil
from argparse import ArgumentParser
import copy 
import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import base64
from PIL import Image
import io
import json
import random 
from flask import Flask, request, jsonify, send_file
import threading
from segmentation.segmentation.MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from segmentation.segmentation.MobileSAM.auto_labeling import PALETTE, clean_mask_by_color_and_morph, get_bounding_box_coords
import matplotlib.pyplot as plt
import time 
import torch
from scipy.spatial.transform import Rotation as R
# HSV MAP with lower and upper bounds for different colors
tgt_hsv_map = {
    'purple': ([100, 50, 120], [130, 200, 255]), # purple for target object
    'purple2': ([115, 80, 60], [145, 255, 255])
}
non_tgt_hsv_map = {
    'orange': ([0, 120, 70], [10, 255, 255]),
    'red': ([170, 120, 70], [179, 255, 255]),
    'pink': ([156, 13, 217], [166, 53, 255]),
    'green': ([40, 80, 80], [85, 255, 255]),
    'blue': ([85, 80, 80], [105, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]),
    'pink1': ([168, 49, 166], [187, 149, 255]),
    'pink2': ([145, 40, 160], [170, 140, 255]),
    'pink3': ([150, 60, 180], [165, 120, 255]),
    'yellow1': ([33, 90, 120], [45, 255, 255]),
    'yellow2': ([20, 127, 152], [26, 227, 252]),   
}
CLASS_STATS = {
1: {'area': [18947.57, 9359.15],
'bbox_area': [15249.18, 3650.79],
'aspect_ratio': [0.94, 0.085],
'extent': [1.25, 0.59],
'solidity': [1.32, 0.67],
'compactness': [14.62, 4.77],
'min_edge': [117.96, 8.73],
'max_edge': [129.03, 29.33],
'num_vertices': [4.28, 0.9]},
2: {'area': [11767.54, 840.65],
'bbox_area': [14686.32, 1080.37],
'aspect_ratio': [0.97, 0.04],
'extent': [0.8, 0.03],
'solidity': [1.0, 0.02],
'compactness': [14.02, 0.87],
'min_edge': [119.29, 4.8],
'max_edge': [123.09, 6.98],
'num_vertices': [8.0, 0.28]},
3: {'area': [27997.27, 9587.97],
'bbox_area': [30024.19, 6967.56],
'aspect_ratio': [0.53, 0.10],
'extent': [0.94, 0.31],
'solidity': [0.97, 0.32],
'compactness': [25.65, 8.33],
'min_edge': [125.74, 25.79],
'max_edge': [238.04, 14.77],
'num_vertices': [7.77, 1.2]},
4: {'area': [47141.36, 22086.61],
'bbox_area': [30403.17, 8873.71],
'aspect_ratio': [0.51, 0.08],
'extent': [1.58, 0.73],
'solidity': [1.66, 0.79],
'compactness': [13.75, 7.2],
'min_edge': [124.52, 25.08],
'max_edge': [242.18, 22.17],
'num_vertices': [4.23, 0.93]},
5: {'area': [8567.25, 2815.02],
'bbox_area': [8512.42, 2775.2],
'aspect_ratio': [0.55, 0.1],
'extent': [1.02, 0.13],
'solidity': [1.08, 0.26],
'compactness': [17.23, 3.79],
'min_edge': [68.54, 16.71],
'max_edge': [122.73, 7.83],
'num_vertices': [4.89, 0.9]},
6: {'area': [19738.42, 11465.22],
'bbox_area': [25330.16, 5100.89],
'aspect_ratio': [0.66, 0.23],
'extent': [0.77, 0.4],
'solidity': [1.35, 0.74],
'compactness': [19.04, 8.8],
'min_edge': [126.2, 22.26],
'max_edge': [204.46, 39],
'num_vertices': [3.29, 1.65]}
}

def calculate_center(img, shown=0):
    # print("Calculating center for image of shape:", img.shape)
    # convert the grayscale image to binary image
    _, thresh = cv2.threshold(img, 127, 255, 0)

    obj_cnt = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("Number of contours found:", len(obj_cnt[0]))
    obj_cnt = obj_cnt[0]

    obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))[-1]  # the mask r cnn could give bad masks
    M = cv2.moments(obj_cnt)  # get center

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if shown == 1:
        cv2.circle(img, (cX, cY), 5, (128), -1)
        cv2.imshow("Image_center", img)
        cv2.waitKey(1)
    return cX, cY


def compare_images(img1, img2):
    return np.count_nonzero(np.logical_and(img1, img2))

def translate(img, dx, dy, shown=0):
    rows, cols = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    if shown == 1:
        cv2.imshow("img_translate", dst)

    return dst


def rotate(img, angle, center=None, shown=0):
    rows, cols = img.shape
    if center is None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_NEAREST)
    if shown == 1:
        cv2.imshow("img_rotate", dst)
        cv2.waitKey(0)
    return dst


def match(img1, img2, cx1, cy1, calc_iou=False):
    max_matched = 0
    max_iou = 0
    best_angle = 0
    best_x = 0
    best_y = 0
    best_rotated_img2 = None
    t1 = time.perf_counter()
    for x in range(-1, 2, 1):
        for y in range(-1, 2, 1):
            trans_img = translate(img2, x, y)
            for angle in np.arange(0, 360, 1):
                center = (cx1 + x, cy1 + y)
                img3 = rotate(trans_img, angle, center)
                matched = compare_images(img1, img3)
                if matched > max_matched:
                    best_angle = angle
                    best_x = x
                    best_y = y
                    max_matched = matched
                    best_rotated_img2 = img3
    t2 = time.perf_counter()
    print(f"Time taken for matching: {1000*(t2 - t1):.2f} milliseconds")
    t3 = time.perf_counter()
    if calc_iou:
        assert best_rotated_img2 is not None, "best_rotated_img2 should not be None when calc_iou is True"
        union = np.count_nonzero(np.logical_or(img1, best_rotated_img2))
        max_iou = max_matched / union if union > 0 else 0
        t4 = time.perf_counter()
        print(f"Time taken for IoU calculation: {1000*(t4 - t3):.2f} milliseconds")
    else:
        max_iou = 0
    return best_angle, best_x, best_y, max_iou

ref_cube_mask = cv2.imread("ref_cube_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_cube_cx, ref_cube_cy = calculate_center(ref_cube_mask)

ref_cylinder_mask = cv2.imread("ref_cylinder_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_cylinder_cx, ref_cylinder_cy = calculate_center(ref_cylinder_mask)

ref_concave_mask = cv2.imread("ref_concave_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_concave_cx, ref_concave_cy = calculate_center(ref_concave_mask)
with open("ref_concave_l515_info.txt", "r") as f:
    ref_concave_info = json.load(f)
ref_concave_angle = ref_concave_info["angle"]

ref_rect_mask = cv2.imread("ref_rect_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_rect_cx, ref_rect_cy = calculate_center(ref_rect_mask)
with open("ref_rect_l515_info.txt", "r") as f:
    ref_rect_info = json.load(f)
ref_rect_angle = ref_rect_info["angle"]
    
ref_half_cube_mask = cv2.imread("ref_half_cube_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_half_cube_cx, ref_half_cube_cy = calculate_center(ref_half_cube_mask, shown=0)
with open("ref_half_cube_l515_info.txt", "r") as f:
    ref_half_cube_info = json.load(f)
ref_half_cube_angle = ref_half_cube_info["angle"]

ref_triangle_mask = cv2.imread("ref_triangle_l515_mask.png", cv2.IMREAD_UNCHANGED)
ref_triangle_cx, ref_triangle_cy = calculate_center(ref_triangle_mask)
ref_triangle_mask = cv2.imread("ref_triangle_l515_mask.png", cv2.IMREAD_UNCHANGED)
with open("ref_triangle_l515_info.txt", "r") as f:
    ref_triangle_info = json.load(f)
ref_triangle_angle = ref_triangle_info["angle"]


cam_info1 = dict()
cam_info2 = dict()

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "CUDA not available, please switch device to CPU!!"

model_type = "vit_t"
sam_checkpoint = "segmentation/segmentation/MobileSAM/weights/mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
# Only sample 256 prompts (points) instead of 1024 (32x32) for faster prediction
# Set points_per_batch to 128, instead of 64. Increase further if GPU has larger memory 
mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=8, points_per_batch=128)

right_cam2ee_pose = np.loadtxt(f"segmentation/mar01_2026_right_cam2ee_pose.txt")
left_cam2ee_pose = np.loadtxt(f"segmentation/mar01_2026_left_cam2ee_pose.txt")


def ndarray_to_base64(arr: np.ndarray) -> str:
    # 1) Raw bytes
    data_bytes = arr.tobytes()
    # 2) Attach shape and dtype so the receiver can reconstruct the array
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(data_bytes).decode("utf-8")  # Encode bytes to base64 string
    }
    

def mask_iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    iou = intersection / union if union != 0 else 0
    return iou  

def containment(a, b):
    # fraction of b contained inside a
    inter = np.count_nonzero((a > 0) & (b > 0))
    b_area = np.count_nonzero(b > 0) + 1e-9
    return inter / b_area

def get_object_poses(rgb, depth, depth_scale, camera_intrinsics, cam_loc, tcp_pose, fixed_depth):
    smoothed = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    enhanced = cv2.addWeighted(smoothed, 1.2, rgb, -0.2, 0)
    enhanced_rgb = enhanced.copy()
    hsv_image = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    rx, ry, rz = tcp_pose[3:6]  # rotation in radians
    tx, ty, tz = tcp_pose[0:3]  # translation in meters
    # print("tcp pose", tcp_pose)
    # 1. Create rotation matrix from RPY (roll-pitch-yaw)
    rotation = R.from_euler('xyz', [rx, ry, rz])  # make sure it's in correct order!
    R_mat = rotation.as_matrix()  # 3x3

    # 2. Construct homogeneous transformation matrix T_base^ee
    T_ee2base = np.eye(4)
    T_ee2base[:3, :3] = R_mat
    T_ee2base[:3, 3] = [tx, ty, tz]
    mobile_sam_latency_ms = 0
    
    start_time = time.time()
    mobile_sam_masks = mask_generator.generate(enhanced_rgb)
    mobile_sam_latency_ms = int(1000*(time.time() - start_time))
    print("Time to generate masks from MobileSAM:", mobile_sam_latency_ms, "ms")
    print(f"Generated {len(mobile_sam_masks)} masks.")

    tgt_predicted_masks = []
    predicted_masks = []
    for color, (lower, upper) in tgt_hsv_map.items():
        LOWER_HSV = np.array(lower, dtype=np.uint8)
        UPPER_HSV = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, LOWER_HSV, UPPER_HSV)

        # # clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # # good, just a little noise left. do connected component analysis to find the largest blob and keep only that
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # if num_labels > 1:
        #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
        #     mask = (labels == largest_label).astype(np.uint8) * 255

        tgt_predicted_masks.append(mask)

    print(">>>>>>>> Number of predicted target masks:", len(tgt_predicted_masks))
    for color, (lower, upper) in non_tgt_hsv_map.items():
        LOWER_HSV = np.array(lower, dtype=np.uint8)
        UPPER_HSV = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, LOWER_HSV, UPPER_HSV)

        # clean up the mask with morphological operations
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # # good, just a little noise left. do connected component analysis to find the largest blob and keep only that
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # if num_labels > 1:
        #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
        #     mask = (labels == largest_label).astype(np.uint8) * 255

        predicted_masks.append(mask)

    for mask_info in mobile_sam_masks:
        mask = mask_info['segmentation'].astype(np.uint8) 
        filtered_mask = clean_mask_by_color_and_morph(enhanced_rgb, mask)
        mask_uint8 = (np.array(filtered_mask) > 0).astype(np.uint8)
        # check if this mask overlaps significantly with any of the existing predicted masks. if yes, skip this mask to avoid duplicates. if no, add this mask to the predicted masks list
        is_duplicate = False
        total_masks = predicted_masks + tgt_predicted_masks
        for existing_mask in total_masks:
            iou = mask_iou(existing_mask > 0, mask_uint8 > 0)
            if iou > 0.85:
                is_duplicate = True
                break
        if not is_duplicate:
            containment_score = containment(mask_uint8 > 0, existing_mask > 0)
            # print("Containment score with hsv masks:", containment_score)
            if containment_score > 0.85:
                is_duplicate = True 
        
        if not is_duplicate:
            predicted_masks.append(mask_uint8)
            total_masks.append(mask_uint8)

    unique_masks = []
    num_objs_found = 0
    sim_obj_poses_by_class = {
        1: [],  # cube
        2: [],  # cylinder
        3: [],  # concave
        4: [],  # rect
        5: [],  # half-cube
        6: []   # triangle
    }
    target_obj_class = None
    for mask in tgt_predicted_masks:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 5500:
                obj_mask = np.zeros_like(mask)
                cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
                is_duplicate = False
                for u_mask in unique_masks:
                    iou = mask_iou(u_mask > 0, obj_mask > 0)
                    if iou > 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_masks.append(obj_mask)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)  # ((cx, cy), (w, h), angle)
                    box = np.intp(box)
                    (w, h) = rect[1]
                    # print("Center from rect (cx, cy):", rect[0], "angle:", rect[2])
                    min_edge, max_edge = min(w, h), max(w, h)   
                    bbox_area = h * w
                    M = cv2.moments(cnt)
                    target_center = (0, 0)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        target_center = (cX, cY)
                        # print(f"Target center: ({cX}, {cY})")

                    depth_raw = depth[cY, cX]  # Note: image is indexed as [row, column] → [y, x]
                    # print(f"Raw depth value at target center: {depth_raw}")

                    aspect_ratio = min(w, h) / max(w, h) if h > 0 else 0
                    extent = area / bbox_area if bbox_area > 0 else 0
                    perimeter = cv2.arcLength(cnt, True)

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)

                    solidity = area / hull_area if hull_area > 0 else 0
                    compactness = (perimeter ** 2) / area if area > 0 else 0
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    num_vertices = len(approx)

                    mask_stats = dict()
                    mask_stats["area"] = area
                    mask_stats["bbox_area"] = bbox_area
                    mask_stats["aspect_ratio"] = aspect_ratio
                    mask_stats["extent"] = extent
                    mask_stats["solidity"] = solidity
                    mask_stats["compactness"] = compactness
                    mask_stats["min_edge"] = min_edge
                    mask_stats["max_edge"] = max_edge
                    mask_stats["num_vertices"] = num_vertices


                    class_votes = {}
                    feature_keys = mask_stats.keys()

                    for cls, stats in CLASS_STATS.items():
                        votes = 0
                        for feat in feature_keys:
                            mean, std = stats[feat]
                            val = mask_stats[feat]

                            if mean - (1.1)*std <= val <= mean + (1.1)*std:
                                votes += 1

                        class_votes[cls] = votes

                    # select class with maximum votes
                    
                    best_class = None
                    # Best Class Precendence for similar sized geometries
                    for class_id in [5, 2, 1, 3, 4, 6]:
                        if class_votes[class_id] == 9:
                            best_class = class_id
                            break
                    
                    if best_class is None:
                        for class_id in [5, 2, 1, 3, 4, 6]:
                            if class_votes[class_id] >= 8: # allow some features to be out of range due to noisy depth and imperfect segmentation
                                best_class = class_id
                                break
                    # require all features to match
                    if best_class is None:
                        # print("class_votes", class_votes)
                        # print("Mask features:", mask_stats)
                        # print("perimeter:", perimeter)
                        # print("It is Garbage Mask!! Discard it!!")
                        # cv2.imwrite(f"debug_masks/{cam_loc}_garbage_mask_major_vote_class_{max(class_votes.values())}.png", obj_mask)
                        pass
                    else:
                        num_objs_found += 1
                        class_to_obj_name = {
                            1: "cube",
                            2: "cylinder",
                            3: "concave",
                            4: "rect",
                            5: "half-cube",
                            6: "triangle"
                        }

                        # print("Target object was discovered as class:", class_to_obj_name[best_class])
                        # angle is for simulation purposes
                        # print(">>> max length edge:", max_edge)
                        obj_cx, obj_cy = calculate_center(obj_mask)
                        angle = rect[2]
                        # if best_class in [1]: # obj could be a cylinder instead of a cube
                        #     translated_cylinder = translate(ref_cylinder_mask, obj_cx - ref_cylinder_cx, obj_cy - ref_cylinder_cy)
                        #     best_angle, best_x, best_y, max_iou_cylinder = match(obj_mask, translated_cylinder, obj_cx, obj_cy, calc_iou=True)

                        #     translated_cube = translate(ref_cube_mask, obj_cx - ref_cube_cx, obj_cy - ref_cube_cy)
                        #     best_angle, best_x, best_y, max_iou_cube = match(obj_mask, translated_cube, obj_cx, obj_cy, calc_iou=True)

                        #     if max_iou_cylinder <= max_iou_cube:
                        #         # Change the class to a cube
                        #         best_class = 1
                        #         # angle might need adjustment from reference cube mask

                        # if best_class in [4]: # obj could be a concave instead of a rect
                        #     translated_concave = translate(ref_concave_mask, obj_cx - ref_concave_cx, obj_cy - ref_concave_cy)
                        #     best_concave_angle, best_x, best_y, max_iou_concave = match(obj_mask, translated_concave, obj_cx, obj_cy, calc_iou=True)
                        #     concave_angle = best_concave_angle - ref_concave_angle + 90 # + 180
                        #     # print(f"Concave matching results - best angle: {best_concave_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", concave_angle)
                        #     angle = concave_angle
                                                                                    
                        #     translated_rect = translate(ref_rect_mask, obj_cx - ref_rect_cx, obj_cy - ref_rect_cy)
                        #     best_angle, best_x, best_y, max_iou_rect = match(obj_mask, translated_rect, obj_cx, obj_cy, calc_iou=True)
                        #     rect_angle = best_angle + ref_rect_angle - 90
                        #     # print(f"Rect matching results - best angle: {best_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", rect_angle)

                        #     if max_iou_concave <= max_iou_rect:
                        #         best_class = 4
                        #         angle = rect_angle # changed the angle
                        
                        # if best_class in [6]:
                        #     translated_triangle = translate(ref_triangle_mask, obj_cx - ref_triangle_cx, obj_cy - ref_triangle_cy)
                        #     best_triangle_angle, best_x, best_y, _ = match(obj_mask, translated_triangle, obj_cx, obj_cy, calc_iou=False)
                        #     triangle_angle = best_triangle_angle - ref_triangle_angle + 90
                        #     # print(f"Triangle matching results - best angle: {best_triangle_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", triangle_angle)
                        #     angle = triangle_angle

                        # if best_class in [5]:
                        #     translated_half_cube = translate(ref_half_cube_mask, obj_cx - ref_half_cube_cx, obj_cy - ref_half_cube_cy)
                        #     best_half_cube_angle, best_x, best_y, _ = match(obj_mask, translated_half_cube, obj_cx, obj_cy, calc_iou=False)
                        #     half_cube_angle = best_half_cube_angle - ref_half_cube_angle + 90
                        #     # print(f"Half cube matching results - best angle: {best_half_cube_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", half_cube_angle)
                        #     angle = half_cube_angle

                        target_obj_class = best_class


                        target_pose_camera = ((fixed_depth) * (np.linalg.inv(camera_intrinsics) @ np.array([rect[0][0], rect[0][1], 1]))).reshape(3,1)
                        # target_pose_camera = ((fixed_depth) * (np.linalg.inv(camera_intrinsics) @ np.array([obj_cx, obj_cy, 1]))).reshape(3,1)
                        
                        homogeneous_pose_camera = np.vstack((target_pose_camera, [1]))
                        if cam_loc == "left":
                            cam2ee_pose = left_cam2ee_pose
                        else:
                            cam2ee_pose = right_cam2ee_pose

                        target_pose_ee = cam2ee_pose @ homogeneous_pose_camera
                        target_pose_base = T_ee2base @ target_pose_ee
                        # print("target_pose_base", target_pose_base)
                        # print(f"Pose of {class_to_obj_name[best_class]} in real base frame: x={target_pose_base[0,0]:.3f}, y={target_pose_base[1,0]:.3f}, z={target_pose_base[2,0]:.3f}")
                        # Apply Z(-90 deg) rotation to convert to sim base frame
                        R_z_90 = np.array([[0, -1, 0, 0],
                                            [1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                        target_pose_sim = R_z_90 @ target_pose_base
                        # print(f"Pose of {class_to_obj_name[best_class]} in sim base frame: x={target_pose_sim[0,0]:.3f}, y={target_pose_sim[1,0]:.3f}, z={target_pose_sim[2,0]:.3f}")
                        print("--------------------------------------------------")
                        # sim_obj_poses_by_class[best_class].append(
                        #     {"cx": f"{target_pose_sim[0,0]:.3f}", "cy": f"{target_pose_sim[1,0]:.3f}", "cz": f"{target_pose_sim[2,0]:.3f}", "angle": angle}
                        # )
                        sim_obj_poses_by_class[best_class].append(
                            {"cx": f"{target_pose_sim[0,0]:.3f}", "cy": f"{target_pose_sim[1,0]:.3f}", "mask": ndarray_to_base64(obj_mask), "angle": angle}
                        )

    for mask in predicted_masks:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # sort contours 
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # print("no of contours:", len(contours))
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # print("Contour area:", area)
            if area >= 5500:
                obj_mask = np.zeros_like(mask)
                cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
                is_duplicate = False
                for u_mask in unique_masks:
                    iou = mask_iou(u_mask > 0, obj_mask > 0)
                    if iou > 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_masks.append(obj_mask)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)  # ((cx, cy), (w, h), angle)
                    box = np.intp(box)
                    (w, h) = rect[1]
                    # print("Center from rect (cx, cy):", rect[0], "angle:", rect[2])
                    min_edge, max_edge = min(w, h), max(w, h)   
                    bbox_area = h * w
                    M = cv2.moments(cnt)
                    target_center = (0, 0)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        target_center = (cX, cY)
                        # print(f"Target center: ({cX}, {cY})")

                    depth_raw = depth[cY, cX]  # Note: image is indexed as [row, column] → [y, x]
                    # print(f"Raw depth value at target center: {depth_raw}")

                    aspect_ratio = min(w, h) / max(w, h) if h > 0 else 0
                    extent = area / bbox_area if bbox_area > 0 else 0
                    perimeter = cv2.arcLength(cnt, True)

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)

                    solidity = area / hull_area if hull_area > 0 else 0
                    compactness = (perimeter ** 2) / area if area > 0 else 0
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    num_vertices = len(approx)

                    mask_stats = dict()
                    mask_stats["area"] = area
                    mask_stats["bbox_area"] = bbox_area
                    mask_stats["aspect_ratio"] = aspect_ratio
                    mask_stats["extent"] = extent
                    mask_stats["solidity"] = solidity
                    mask_stats["compactness"] = compactness
                    mask_stats["min_edge"] = min_edge
                    mask_stats["max_edge"] = max_edge
                    mask_stats["num_vertices"] = num_vertices

                    class_votes = {}
                    feature_keys = mask_stats.keys()

                    for cls, stats in CLASS_STATS.items():
                        votes = 0
                        for feat in feature_keys:
                            mean, std = stats[feat]
                            val = mask_stats[feat]

                            if mean - (1.1)*std <= val <= mean + (1.1)*std:
                                votes += 1

                        class_votes[cls] = votes

                    # select class with maximum votes
                    
                    best_class = None
                    # Best Class Precendence for similar sized geometries
                    for class_id in [5, 2, 1, 3, 4, 6]:
                        if class_votes[class_id] == 9:
                            best_class = class_id
                            break
                    
                    if best_class is None:
                        for class_id in [5, 2, 1, 3, 4, 6]:
                            if class_votes[class_id] >= 8: # allow some features to be out of range due to noisy depth and imperfect segmentation
                                best_class = class_id
                                break
                    # require all features to match
                    if best_class is None:
                        # print("class_votes", class_votes)
                        # print("Mask features:", mask_stats)
                        # print("perimeter:", perimeter)
                        # print("It is Garbage Mask!! Discard it!!")
                        # cv2.imwrite(f"debug_masks/{cam_loc}_garbage_mask_major_vote_class_{max(class_votes.values())}.png", obj_mask)
                        # cv2.imshow("Individual Masks", obj_mask)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        pass
                    else:
                        num_objs_found += 1
                        class_to_obj_name = {
                            1: "cube",
                            2: "cylinder",
                            3: "concave",
                            4: "rect",
                            5: "half-cube",
                            6: "triangle"
                        }
                        # print(f"It is a {class_to_obj_name[best_class]}!!")
                        # cv2.imshow("Individual Masks", obj_mask)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # angle is for simulation purposes
                        angle = rect[2]
                        obj_cx, obj_cy = calculate_center(obj_mask)
                        angle = rect[2]
                        # if best_class in [1]: # obj could be a cylinder instead of a cube
                        #     translated_cylinder = translate(ref_cylinder_mask, obj_cx - ref_cylinder_cx, obj_cy - ref_cylinder_cy)
                        #     best_angle, best_x, best_y, max_iou_cylinder = match(obj_mask, translated_cylinder, obj_cx, obj_cy, calc_iou=True)

                        #     translated_cube = translate(ref_cube_mask, obj_cx - ref_cube_cx, obj_cy - ref_cube_cy)
                        #     best_angle, best_x, best_y, max_iou_cube = match(obj_mask, translated_cube, obj_cx, obj_cy, calc_iou=True)

                        #     if max_iou_cylinder <= max_iou_cube:
                        #         # Change the class to a cube
                        #         best_class = 1
                        #         # angle might need adjustment from reference cube mask

                        # if best_class in [4]: # obj could be a concave instead of a rect
                        #     translated_concave = translate(ref_concave_mask, obj_cx - ref_concave_cx, obj_cy - ref_concave_cy)
                        #     best_concave_angle, best_x, best_y, max_iou_concave = match(obj_mask, translated_concave, obj_cx, obj_cy, calc_iou=True)
                        #     concave_angle = best_concave_angle - ref_concave_angle + 90 # + 180
                        #     # print(f"Concave matching results - best angle: {best_concave_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", concave_angle)
                        #     angle = concave_angle
                                                                                    
                        #     translated_rect = translate(ref_rect_mask, obj_cx - ref_rect_cx, obj_cy - ref_rect_cy)
                        #     best_angle, best_x, best_y, max_iou_rect = match(obj_mask, translated_rect, obj_cx, obj_cy, calc_iou=True)
                        #     rect_angle = best_angle + ref_rect_angle - 90
                        #     # print(f"Rect matching results - best angle: {best_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", rect_angle)

                        #     if max_iou_concave <= max_iou_rect:
                        #         best_class = 4
                        #         angle = rect_angle # changed the angle
                        
                        # if best_class in [6]:
                        #     translated_triangle = translate(ref_triangle_mask, obj_cx - ref_triangle_cx, obj_cy - ref_triangle_cy)
                        #     best_triangle_angle, best_x, best_y, _ = match(obj_mask, translated_triangle, obj_cx, obj_cy, calc_iou=False)
                        #     triangle_angle = best_triangle_angle - ref_triangle_angle + 90
                        #     # print(f"Triangle matching results - best angle: {best_triangle_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", triangle_angle)
                        #     angle = triangle_angle

                        # if best_class in [5]:
                        #     translated_half_cube = translate(ref_half_cube_mask, obj_cx - ref_half_cube_cx, obj_cy - ref_half_cube_cy)
                        #     best_half_cube_angle, best_x, best_y, _ = match(obj_mask, translated_half_cube, obj_cx, obj_cy, calc_iou=False)
                        #     half_cube_angle = best_half_cube_angle - ref_half_cube_angle + 90
                        #     # print(f"Half cube matching results - best angle: {best_half_cube_angle}, best x translation: {best_x}, best y translation: {best_y}")
                        #     # print("adjusted angle:", half_cube_angle)
                        #     angle = half_cube_angle


                        target_pose_camera = ((fixed_depth) * (np.linalg.inv(camera_intrinsics) @ np.array([rect[0][0], rect[0][1], 1]))).reshape(3,1)
                        # target_pose_camera = ((fixed_depth) * (np.linalg.inv(camera_intrinsics) @ np.array([obj_cx, obj_cy, 1]))).reshape(3,1)
                        
                        homogeneous_pose_camera = np.vstack((target_pose_camera, [1]))
                        if cam_loc == "left":
                            cam2ee_pose = left_cam2ee_pose
                        else:
                            cam2ee_pose = right_cam2ee_pose

                        target_pose_ee = cam2ee_pose @ homogeneous_pose_camera
                        target_pose_base = T_ee2base @ target_pose_ee
                        # print("target_pose_base", target_pose_base)
                        # print(f"Pose of {class_to_obj_name[best_class]} in real base frame: x={target_pose_base[0,0]:.3f}, y={target_pose_base[1,0]:.3f}, z={target_pose_base[2,0]:.3f}")
                        # Apply Z(-90 deg) rotation to convert to sim base frame
                        R_z_90 = np.array([[0, -1, 0, 0],
                                            [1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                        target_pose_sim = R_z_90 @ target_pose_base
                        # print(f"Pose of {class_to_obj_name[best_class]} in sim base frame: x={target_pose_sim[0,0]:.3f}, y={target_pose_sim[1,0]:.3f}, z={target_pose_sim[2,0]:.3f}")
                        # print("--------------------------------------------------")
                        sim_obj_poses_by_class[best_class].append(
                            {"cx": f"{target_pose_sim[0,0]:.3f}", "cy": f"{target_pose_sim[1,0]:.3f}", "mask": ndarray_to_base64(obj_mask), "angle": angle}
                        )
    # print(f"Total {num_objs_found} objects found in {cam_loc} camera view.")
    return sim_obj_poses_by_class, target_obj_class, num_objs_found, mobile_sam_latency_ms

@app.route('/stream', methods=['POST'])
def stream():    
    data = request.json

    id = data.get("id")
    tcp_pose = data.get("tcp_pose")  # [x, y, z, rx, ry, rz] in base frame
    global rgb_data1, depth_data1
    global rgb_data2, depth_data2
    global cam_info1, cam_info2

    left_cam_object_poses, left_target_obj_class, left_num_objs_found, left_mobile_sam_latency_ms = get_object_poses(rgb=cv2.cvtColor(rgb_data1, cv2.COLOR_BGR2RGB), depth=copy.deepcopy(depth_data1), depth_scale=cam_info1["depth_scale"], camera_intrinsics=cam_info1["cam_intr"], cam_loc="left", tcp_pose=tcp_pose, fixed_depth=0.34)
    right_cam_object_poses, right_target_obj_class, right_num_objs_found, right_mobile_sam_latency_ms = get_object_poses(rgb=cv2.cvtColor(rgb_data2, cv2.COLOR_BGR2RGB), depth=copy.deepcopy(depth_data2), depth_scale=cam_info2["depth_scale"], camera_intrinsics=cam_info2['cam_intr'], cam_loc="right", tcp_pose=tcp_pose, fixed_depth=0.335)
    
    # print("left_cam_object_poses", left_cam_object_poses, "left_target_obj_class", left_target_obj_class)
    # print("right_cam_object_poses", right_cam_object_poses, "right_target_obj_class", right_target_obj_class)
    print(f"Num objs by (left, right) cams: ({left_num_objs_found}, {right_num_objs_found}). MobileSAM latencies (left, right) cams: ({left_mobile_sam_latency_ms}, {right_mobile_sam_latency_ms}) ms.")
    obj_poses = {
        "left": {
            "object_poses": left_cam_object_poses,
            "target_obj_class": left_target_obj_class
        },
        "right": {
            "object_poses": right_cam_object_poses,
            "target_obj_class": right_target_obj_class
        }
    }
    return jsonify(obj_poses)
            
def start_flask_app():
    app.run(port=7777, debug=True, use_reloader=False)

def start_realsense_stream():
    global rgb_data1, depth_data1
    global rgb_data2, depth_data2

    video = 'test-000510-spiral-2' # 'test-001487-rl-v10'
    dataset = 'dataset/multi_cam'

    device_id = {
                "l515a": None,
                "l515b": None,
                "l515c": None
            }

    ctx = rs.context()
    devices = ctx.query_devices()
    print("Device Names:")
    for d in devices:
        print(f"  {d.get_info(rs.camera_info.name)} (SN: {d.get_info(rs.camera_info.serial_number)})")

    print(f"Found {len(devices)} RealSense devices:")
    use_third_cam = bool(os.environ.get("TRACE_D415_SERIAL"))

    config1 = rs.config()
    config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config1.enable_device(os.environ["TRACE_CAMERA1_SERIAL"])
    device1_name = devices[1].get_info(rs.camera_info.name)
    
    if device1_name == "Intel RealSense L515":
        device_id["l515a"] = 1
    config2 = rs.config()
    config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config2.enable_device(os.environ["TRACE_CAMERA2_SERIAL"])
    device2_name = devices[2].get_info(rs.camera_info.name)
    if device2_name == "Intel RealSense L515":
        device_id["l515b"] = 2
        

    if use_third_cam:
        config3 = rs.config()
        config3.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config3.enable_device(os.environ["TRACE_D415_SERIAL"])
        device3_name = devices[0].get_info(rs.camera_info.name)
        if device3_name == "Intel RealSense D415":
            device_id["d415c"] = 3

        
    
    record = True
    no_annotations = True 
    save_png = False


    pipeline1 = rs.pipeline()
    profile1 = pipeline1.start(config1)

    pipeline2 = rs.pipeline()
    profile2 = pipeline2.start(config2)

    if use_third_cam:
        pipeline3 = rs.pipeline()
        profile3 = pipeline3.start(config3)

    # depth align to color
    align = rs.align(rs.stream.color)
    color_profile1 = rs.video_stream_profile(profile1.get_stream(rs.stream.color))
    color_intrinsics1 = color_profile1.get_intrinsics()

    depth_sensor1 = profile1.get_device().first_depth_sensor()
    depth_scale1 = int(round(1 / depth_sensor1.get_depth_scale()))

    # align depth to color for camera 2
    color_profile2 = rs.video_stream_profile(profile2.get_stream(rs.stream.color))
    color_intrinsics2 = color_profile2.get_intrinsics()

    depth_sensor2 = profile2.get_device().first_depth_sensor()
    depth_scale2 = int(round(1 / depth_sensor2.get_depth_scale()))

    if use_third_cam:
        color_profile3 = rs.video_stream_profile(profile3.get_stream(rs.stream.color))
        color_intrinsics3 = color_profile3.get_intrinsics()

    if device1_name == "Intel RealSense L515":
        device_id["l515a"] = 0
        print("Setting visual preset to No Ambient Light")
        depth_sensor1.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.low_ambient_light))
        depth_sensor1.set_option(
            rs.option.min_distance, 200
        )  # 0.2 meters.
        depth_sensor1.set_option(
            rs.option.confidence_threshold, 3.0
        )  # default is 1.0
        color_sensor1 = profile1.get_device().first_color_sensor()
        color_sensor1.set_option(rs.option.enable_auto_exposure, True)

    if device2_name == "Intel RealSense L515":
        device_id["l515b"] = 1
        print("Setting visual preset to No Ambient Light")
        depth_sensor2.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.low_ambient_light))
        depth_sensor2.set_option(
            rs.option.min_distance, 200
        )  # 0.2 meters.
        depth_sensor2.set_option(
            rs.option.confidence_threshold, 3.0
        )  # default is 1.0
        color_sensor2 = profile2.get_device().first_color_sensor()
        color_sensor2.set_option(rs.option.enable_auto_exposure, True)
    
    if device3_name == "Intel RealSense D415":
        device_id["d415c"] = 2
        print("Setting visual preset to No Ambient Light")
        color_sensor3 = profile3.get_device().first_color_sensor()
        color_sensor3.set_option(rs.option.enable_auto_exposure, True)
    

    print('camera 1 - color_intrinsics:', color_intrinsics1)

    cam_info1['im_w'] = color_intrinsics1.width
    cam_info1['im_h'] = color_intrinsics1.height
    cam_info1['depth_scale'] = depth_scale1
    fx1, fy1 = color_intrinsics1.fx, color_intrinsics1.fy
    cx1, cy1 = color_intrinsics1.ppx, color_intrinsics1.ppy
    cam_info1['cam_intr'] = [[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]]

    print('camera 2 - color_intrinsics:', color_intrinsics2)
    cam_info2['im_w'] = color_intrinsics2.width
    cam_info2['im_h'] = color_intrinsics2.height
    cam_info2['depth_scale'] = depth_scale2
    fx2, fy2 = color_intrinsics2.fx, color_intrinsics2.fy
    cx2, cy2 = color_intrinsics2.ppx, color_intrinsics2.ppy
    cam_info2['cam_intr'] = [[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]]

    if use_third_cam:
        print('camera 3 - color_intrinsics:', color_intrinsics3)
        cam_info3 = dict()
        cam_info3['im_w'] = color_intrinsics3.width
        cam_info3['im_h'] = color_intrinsics3.height
        fx3, fy3 = color_intrinsics3.fx, color_intrinsics3.fy
        cx3, cy3 = color_intrinsics3.ppx, color_intrinsics3.ppy
        cam_info3['cam_intr'] = [[fx3, 0, cx3], [0, fy3, cy3], [0, 0, 1]]

    output_dir = os.path.join(dataset, video)
    print("output_dir:", output_dir)
    if os.path.exists(output_dir):
        key = input(f"{output_dir} has already existed, overwrite? [y/N]")
        if not key or key.upper() != "Y":
            exit(0)
        else:
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'color_l515a'))
    os.makedirs(os.path.join(output_dir, 'depth_l515a'))
    os.makedirs(os.path.join(output_dir, 'color_l515b'))
    os.makedirs(os.path.join(output_dir, 'depth_l515b'))

    if use_third_cam:
        os.makedirs(os.path.join(output_dir, 'color_d415c'))

    if output_dir is not None:
        cam_info1['id'] = os.path.basename(output_dir)
        cam_config_path1 = os.path.join(output_dir, 'config1.json')
        with open(cam_config_path1, 'w') as f:
            print(f"Camera info has been saved to: {cam_config_path1}.")
            json.dump(cam_info1, f, indent=4)

        cam_info2['id'] = os.path.basename(output_dir)
        cam_config_path2 = os.path.join(output_dir, 'config2.json')
        with open(cam_config_path2, 'w') as f:
            print(f"Camera info has been saved to: {cam_config_path2}.")
            json.dump(cam_info2, f, indent=4)

        if use_third_cam:
            cam_info3['id'] = os.path.basename(output_dir)
            cam_config_path3 = os.path.join(output_dir, 'config3.json')
            with open(cam_config_path3, 'w') as f:
                print(f"Camera info has been saved to: {cam_config_path3}.")
                json.dump(cam_info3, f, indent=4)

    depth_vis_scale = 100

    def mouse_callback(event, x, y, flags, params):
        im_h, im_w, _ = im_vis.shape
        H, W = im_h, im_w // 2
        if event == cv2.EVENT_LBUTTONDOWN:
            r, g, b = im_vis[y, x]
            if x < W:
                print(f"({x}, {y}), rgb: ({r}, {g}, {b})")
            else:
                if r == 255:
                    print(f"({x}, {y}), depth: >2.55 m")
                else:
                    d = r / depth_vis_scale
                    print(f"({x}, {y}), depth: {d}m")

    count = 0
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    called_once = False
    try:
        while True:
            frames1 = pipeline1.wait_for_frames()
            aligned_frames1 = align.process(frames1)
            depth_frame1 = aligned_frames1.get_depth_frame()
            color_frame1 = aligned_frames1.get_color_frame()

            if not depth_frame1 or not color_frame1:
                continue

            # Convert images to numpy arrays
            bgr_ims = []
            depth_ims = []
            depth_ims_vis = []
            bgr_ims.append(np.asanyarray(color_frame1.get_data()))
            depth_ims.append(np.asanyarray(depth_frame1.get_data()))
            im_h1, im_w1 = depth_ims[0].shape

            depth_scale1 = depth_sensor1.get_depth_scale()


            depth_ims_vis.append(np.clip(depth_ims[0].astype(np.float32) / depth_scale1 * depth_vis_scale, a_min=0, a_max=255))
            depth_ims_vis[0] = np.repeat(depth_ims_vis[0], 3).reshape((im_h1, im_w1, 3)).astype(np.uint8)

            # Capture frames from camera 2
            frames2 = pipeline2.wait_for_frames()
            aligned_frames2 = align.process(frames2)
            depth_frame2 = aligned_frames2.get_depth_frame()
            color_frame2 = aligned_frames2.get_color_frame()
            if not depth_frame2 or not color_frame2:
                continue
            # Convert images to numpy arrays
            bgr_ims.append(np.asanyarray(color_frame2.get_data()))
            depth_ims.append(np.asanyarray(depth_frame2.get_data()))
            im_h2, im_w2 = depth_ims[1].shape
            # depth_im_vis2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_im2, alpha=0.03), cv2.COLORMAP_JET)
            depth_ims_vis.append(np.clip(depth_ims[1].astype(np.float32) / depth_scale2 * depth_vis_scale, a_min=0, a_max=255))
            depth_ims_vis[1] = np.repeat(depth_ims_vis[1], 3).reshape((im_h2, im_w2, 3)).astype(np.uint8)
            

            if use_third_cam:
                # Capture frames from camera 3
                frames3 = pipeline3.wait_for_frames()
                aligned_frames3 = align.process(frames3)
                color_frame3 = aligned_frames3.get_color_frame()
                # Convert images to numpy arrays
                bgr_ims.append(np.asanyarray(color_frame3.get_data()))


            rev_device_id = {value:key for key, value in device_id.items()}
            # print(f"Device ID mapping: {rev_device_id}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0)
            thickness = 2
            if not no_annotations:
                cv2.putText(bgr_ims[0], f"cam_{rev_device_id[1]}_rgb", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(bgr_ims[1], f"cam_{rev_device_id[2]}_rgb", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                if use_third_cam:
                    cv2.putText(bgr_ims[2], f"cam_{rev_device_id[3]}_rgb", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                
                cv2.putText(depth_ims_vis[0], f"cam_{rev_device_id[1]}_depth", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(depth_ims_vis[1], f"cam_{rev_device_id[2]}_depth", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                # cv2.putText(depth_ims_vis[2], f"cam_{rev_device_id[3]}_depth", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                
            # stack the images into a grid
            im_vis = np.hstack((
                np.vstack((bgr_ims[0], depth_ims_vis[0])), 
                np.vstack((bgr_ims[1], depth_ims_vis[1])), 
                # np.vstack((bgr_ims[2], depth_ims_vis[2]))
                ))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', im_vis)
            cv2.setMouseCallback("RealSense", mouse_callback)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r') and output_dir is not None:
                print("Recording...")
                called_once = True
                record = True 
                print("Start recording...")
                # Convert an np array to Image.Open object
                rgb_data1 = bgr_ims[0]
                depth_data1 = depth_ims[0]

                rgb_data2 = bgr_ims[1]
                depth_data2 = depth_ims[1]
            if key == ord('s') and record:
                record = False
                print("Stop recording...")
            if called_once:
                count += 1
                rgb_data1 = bgr_ims[0]
                depth_data1 = depth_ims[0]

                rgb_data2 = bgr_ims[1]
                depth_data2 = depth_ims[1]

                if count % 1000 == 0:
                    print(f"{count} frames have been streamed.")

            if record:
                # print("bgr_ims length:", len(bgr_ims), "depth_ims_vis length:", len(depth_ims_vis))
                for i in range(2):
                    device_name = rev_device_id[i]
                    ext = 'png' if save_png else 'jpg'
                    cv2.imwrite(os.path.join(output_dir, f'color_{device_name}', f"{count:04d}-color.{ext}"), bgr_ims[i])
                    cv2.imwrite(os.path.join(output_dir, f'depth_{device_name}', f"{count:04d}-depth.png"), depth_ims_vis[i])

                # Save the external camera images as png files instead of jpg
                device_name = rev_device_id[2]
                # print(f"Saving color img from ext cam at {os.path.join(output_dir, f'color_{device_name}', f'{count:04d}-color.jpg')}...")
                cv2.imwrite(os.path.join(output_dir, f'color_{device_name}', f"{count:04d}-color.png"), bgr_ims[2])


    finally:
        pipeline1.stop()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    rgb_data1 = None
    depth_data1 = None
    
    rgb_data2 = None
    depth_data2 = None

    flask_thread = threading.Thread(target=start_flask_app)
    realsense_thread = threading.Thread(target=start_realsense_stream)

    flask_thread.start()
    realsense_thread.start()

    flask_thread.join()
    realsense_thread.join()



