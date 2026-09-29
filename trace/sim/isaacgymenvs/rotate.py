import json
import os
import cv2 
import numpy as np
from isaacgymenvs.rl_policy import calculate_center
from video_streaming_server_multi_cam import rotate
def get_ref_mask_info(mask_path, info_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    cx, cy = calculate_center(mask)
    with open(info_path, "r") as f:
        info = json.load(f)
    angle = info["angle"]
    return mask, cx, cy, angle

concave_mask = cv2.imread(f"{os.getcwd()}/obj-ref-images/segm_concave.png", cv2.IMREAD_UNCHANGED)
rotated_concaves = []

for angle in range(0, 360, 45):
    triangle_center = calculate_center(concave_mask)
    rotated_triangle = rotate(concave_mask, angle, center=triangle_center)
    rotated_concaves.append(rotated_triangle)
    print("angle:", angle, "center:", triangle_center)

row1 = cv2.hconcat(rotated_concaves[:4])
row2 = cv2.hconcat(rotated_concaves[4:])
triangle_rotations = cv2.vconcat([row1, row2])
cv2.imwrite("ref_db_concave_orientation_set.png", triangle_rotations)

# This triangle mask shape is 224 by 224, but the triangle is not at the center of the image. 
# triangle mask center: 111, 114 and I want it to be at 112, 112

# triangle_mask = cv2.imread(f"{os.getcwd()}/obj-ref-images/segm_triangle.png", cv2.IMREAD_UNCHANGED)

# triangle_mask_center = calculate_center(triangle_mask)
# print("triangle mask center:", triangle_mask_center)
# # I want to shift the triangle mask so that the center is at (112, 112)
# shift_x = 112 - triangle_mask_center[0]
# shift_y = 112 - triangle_mask_center[1]
# M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
# shifted_triangle_mask = cv2.warpAffine(triangle_mask, M, (triangle_mask.shape[1], triangle_mask.shape[0]))
# cv2.imwrite("ref_db_triangle_mask_centered.png", shifted_triangle_mask)
# shifted_triangle_mask_center = calculate_center(shifted_triangle_mask)
# print("shifted triangle mask center:", shifted_triangle_mask_center)


# concave_mask = cv2.imread(f"{os.getcwd()}/obj-ref-images/segm_concave.png", cv2.IMREAD_UNCHANGED)

# concave_mask_center = calculate_center(concave_mask)
# print("concave mask center:", concave_mask_center)
# # I want to shift the concave mask so that the center is at (112, 112)
# shift_x = 112 - concave_mask_center[0]
# shift_y = 112 - concave_mask_center[1]
# M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
# shifted_concave_mask = cv2.warpAffine(concave_mask, M, (concave_mask.shape[1], concave_mask.shape[0]))
# cv2.imwrite("ref_db_concave_mask_centered.png", shifted_concave_mask)
# shifted_concave_mask_center = calculate_center(shifted_concave_mask)
# print("shifted concave mask center:", shifted_concave_mask_center)

