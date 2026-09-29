from .dataset import LifelongEvalDataset
import math
import random
import time
import torch
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import numpy as np
import cv2
import math

import imutils
from .__init__ import rotate
from .prediction_vis import DEFAULT_TILE_SIZE, render_prediction_grid
from .models import reinforcement_net
from .models_seq import reinforcement_net_seq
# import ..utils. as utils
from vision.efficientnet import EfficientNet
from .constants import (
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL,
    PIXEL_SIZE,
    PUSH_DISTANCE_PIXEL,
    TARGET_LOWER,
    TARGET_UPPER,
    IMAGE_PAD_WIDTH,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    NUM_ROTATION,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    BG_THRESHOLD,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PUSH_DISTANCE,
)


class MCTSHelper:
    """
    Simulate the state after push actions.
    Evaluation the grasp rewards.
    """

    def __init__(self, grasp_model_path, grasp_eval_model_path, device):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        # Initialize Mask R-CNN
        # self.mask_model = get_model_instance_segmentation(2)
        # self.mask_model.load_state_dict(torch.load(mask_model_path))
        # self.mask_model = self.mask_model.to(self.device)
        # self.mask_model.eval()

        # Initialize Grasp Q Evaluation
        self.grasp_model = reinforcement_net(device=device)
        self.grasp_model.load_state_dict(
            torch.load(grasp_model_path, map_location=self.device)["model"], strict=False)
        self.grasp_model = self.grasp_model.to(self.device)
        self.grasp_model.eval()

        self.grasp_model_seq = reinforcement_net_seq(device=device)
        self.grasp_model_seq.load_state_dict(
            torch.load(grasp_model_path, map_location=self.device)["model"], strict=False)
        self.grasp_model_seq = self.grasp_model_seq.to(self.device)
        self.grasp_model_seq.eval()

        # EfficientNet-based grasp eval model
        self.grasp_eval_model = EfficientNet.from_name("efficientnet-b0", in_channels=1, num_classes=1)
        self.grasp_eval_model.load_state_dict(
            torch.load(grasp_eval_model_path, map_location=self.device)["model"], strict=False)
        self.grasp_eval_model = self.grasp_eval_model.to(self.device)
        self.grasp_eval_model.eval()

        self.move_recorder = {}
        self.simulation_recorder = {}

        # Build batched thetas: (R,2,3)
        R = NUM_ROTATION
        angles = torch.arange(R, device=device, dtype=torch.float32) * (2*math.pi / R)
        c = torch.cos(angles)
        s = torch.sin(angles)

        # theta for output->input rotation by -angle (i.e. image rotates by +angle)
        self.theta = torch.zeros((R, 2, 3), device=device, dtype=torch.float32)
        self.theta[:, 0, 0] =  c
        self.theta[:, 0, 1] =  -s
        self.theta[:, 1, 0] = s
        self.theta[:, 1, 1] =  c
        # theta[:, :, 2] already 0

        # inverse (rotate back)
        self.neg_theta = torch.zeros_like(self.theta)
        self.neg_theta[:, 0, 0] =  c
        self.neg_theta[:, 0, 1] = s
        self.neg_theta[:, 1, 0] =  -s
        self.neg_theta[:, 1, 1] =  c
        
    def reset(self):
        self.move_recorder = {}
        self.simulation_recorder = {}

    def rotate_image_point(self, x, y, W, H, theta):
        cx = W / 2.0
        cy = H / 2.0

        # shift to center
        x_c = x - cx
        y_c = y - cy

        # rotate
        c = math.cos(theta)
        s = math.sin(theta)

        x_r = x_c * c - y_c * s
        y_r = x_c * s + y_c * c

        # shift back
        return x_r + cx, y_r + cy
    def check_valid(self, point, point_on_contour, thresh):
        # out of boundary
        if not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
            < point[0]
            < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        ) or not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
            < point[1]
            < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        ):
            qualify = False
        else:
            # compute rotation angle
            down = (0, 1)
            current = (
                point_on_contour[0] - point[0],
                point_on_contour[1] - point[1],
            )
            dot = (
                down[0] * current[0] + down[1] * current[1]
            )  # dot product between [x1, y1] and [x2, y2]
            det = down[0] * current[1] - down[1] * current[0]  # determinant
            angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
            angle = math.degrees(angle)
            crop = thresh[
                point[1]
                - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[1]
                + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
                + 1,
                point[0]
                - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[0]
                + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
                + 1,
            ]
            # test the rotated crop part
            crop = rotate(crop, angle, is_mask=True)
            (h, w) = crop.shape
            crop_cy, crop_cx = (h // 2, w // 2)
            crop = crop[
                crop_cy
                - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
                - 1 : crop_cy
                + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
                + 2,
                crop_cx - GRIPPER_PUSH_RADIUS_PIXEL - 1 : crop_cx + GRIPPER_PUSH_RADIUS_PIXEL + 2,
            ]
            print("crop shape in px:", crop.shape)
            qualify = np.sum(crop > 0) == 0

        return qualify

    def global_adjust(self, point, point_on_contour, thresh):
        for dis in [0.01, 0.02]:
            dis = dis / PIXEL_SIZE
            diff_x = point_on_contour[0] - point[0]
            diff_y = point_on_contour[1] - point[1]
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            test_point = (round(point[0] - diff_x * dis), round(point[1] - diff_y * dis))
            qualify = self.check_valid(test_point, point_on_contour, thresh)
            if qualify:
                return qualify, test_point

        return False, None

    # rgb,
    @torch.no_grad()
    def grasp_prob_B(self, depth, seg, tile_length=112):
        """
        Leverage the grasp classifier to evaluate grasping probability for a batch of depth images.
        
        input:
            depth: (B, H, W) tensor
            tile: int, crop size around the target
        output:
            grasp_prob: (B,) tensor float
        """
        depth_bchw = depth.unsqueeze(1)  # (B,1,H,W)
        depth_bhwc = depth_bchw.permute(0,2,3,1)  # (B,H,W,1)
        seg = seg.int().float().unsqueeze(1)  # (B,1,224,224)
        tgt_centers, valid = self.get_tgt_centers(segm_map_bhw=seg.squeeze(1), target_label=255)
        # if not torch.all(valid):
            # save the failed segm image for debugging via cv2
            # print("env indices with failed target center:", torch.nonzero(~valid).squeeze().cpu().numpy())
            # cv2.imwrite("${TRACE_RUNS}/debug", seg[~valid][0,0,:,:].cpu().numpy().astype(np.uint8))
            # # save rgb image for debugging via cv2
            # cv2.imwrite("${TRACE_RUNS}/debug", (rgb[~valid][0].cpu().numpy()).astype(np.uint8))
            # print("Target center not found in some batch items!")
            # return torch.nonzero(~valid)
            # pass
        dep_crop_bctt = self.crop_fixed_tile_torch(depth_bhwc, tgt_centers=tgt_centers, tile=tile_length, pad_value=0)  # (B,1,t,t)
        
        # Pre-process depth image (normalize)
        dep_crop_bctt[:, 0, :, :] = dep_crop_bctt[:, 0, :, :] - DEPTH_MEAN[0]
        dep_crop_bctt[:, 0, :, :] = dep_crop_bctt[:, 0, :, :] / DEPTH_STD[0]
        output_prob = self.grasp_eval_model(dep_crop_bctt)  # (B,1)

        grasp_value = torch.sigmoid(output_prob)
        grasp_value[~valid, 0] = -2.0  # invalid ones set to -1
        # print("grasp_value shape:", grasp_value.shape)
        return grasp_value[:, 0]  # (B,)


    @torch.no_grad()
    def get_grasp_q_batch(self, color_heightmaps, depth_heightmaps, post_checking=False, is_real=False):
        """
            color_heightmaps: (B, H, W, 3) uint8 or float
            depth_heightmaps: (B, H, W) float
            returns:
            color_tensor_np: (B, 3, Hpad, Wpad) float32
            depth_tensor_np: (B, 1, Hpad, Wpad) float32
        """
        color = np.copy(np.asarray(color_heightmaps))
        depth = np.copy(np.asarray(depth_heightmaps))
        assert color.ndim == 4 and color.shape[-1] == 3, f"Expected color (B,H,W,3), got {color.shape}"
        assert depth.ndim == 3, f"Expected depth (B,H,W), got {depth.shape}"
        assert color.shape[0] == depth.shape[0], "Batch size mismatch"

        B, H, W, C = color.shape


        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color,
            ((0, 0), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(
            depth, 
            ((0, 0), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH)), 
            mode="constant", 
            constant_values=0
        )
    
        if color_heightmap_pad.dtype != np.float32 and color_heightmap_pad.dtype != np.float64:
            color_heightmap_pad = color_heightmap_pad.astype(np.float32)
        else:
            color_heightmap_pad = color_heightmap_pad.astype(np.float32)

        # If your color inputs are uint8 0..255, scale to 0..1.
        # If they are already 0..1 floats, skip scaling.
        if color_heightmap_pad.max() > 1.5:
            color_heightmap_pad = color_heightmap_pad / 255.0

        # Pre-process color image (scale and normalize)        
        color_mean = np.asarray(COLOR_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        color_std  = np.asarray(COLOR_STD,  dtype=np.float32).reshape(1, 1, 1, 3)
        color_norm = (color_heightmap_pad - color_mean) / color_std

        # Pre-process depth image (normalize)
        # Depth: normalize (assumes depth already in meters or consistent scale)
        depth_heightmap_pad = depth_heightmap_pad.astype(np.float32)
        depth_mean = np.asarray(DEPTH_MEAN, dtype=np.float32).reshape(1, 1, 1)
        depth_std  = np.asarray(DEPTH_STD,  dtype=np.float32).reshape(1, 1, 1)
        depth_norm = (depth_heightmap_pad - depth_mean) / depth_std

        # Add channel dim to depth: (B, Hpad, Wpad, 1)
        depth_norm = depth_norm[..., None]

        # Convert to Numpy BHWC to Torch (B,C,H,W)
        input_color_data = torch.from_numpy(color_norm.astype(np.float32)).permute(0, 3, 1, 2)
        input_depth_data = torch.from_numpy(depth_norm.astype(np.float32)).permute(0, 3, 1, 2)
        
        print("[Grasp Model Input] color img shape:", input_color_data.shape, "depth img shape:", input_depth_data.shape)
        # Pass input data through model
        output_prob = self.grasp_model(input_color_data, input_depth_data)
                                    #    , True, -1, False, device=device)
        # print('len(output_prob):', len(output_prob))
        # print("output_prob[0] shape:", output_prob[0].shape)
        
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,]
                )
            else:
                grasp_predictions = np.concatenate(
                    (
                        grasp_predictions,
                        output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,],
                    ),
                    axis=0,
                )
       
        # post process, only grasp one object, focus on blue object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        # Mask stats
        mask_uint8 = (np.array(mask) > 0).astype(np.uint8)
        area = mask_uint8.sum()

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w,h = None, None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (w, h) = rect[1]
                
            min_edge, max_edge = min(w, h), max(w, h)   
            bbox_area = h * w

            aspect_ratio = min(w, h) / max(w, h) if h > 0 else 0
            extent = area / bbox_area if bbox_area > 0 else 0

            # contour-based features
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

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
            print("Center (cx, cy):", rect[0])
            print("mask_stats:", mask_stats)
        else:
            print("Target recognition by color was not possible!!")
            
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask_pad != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones(
            (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
        )
        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
        for rotate_idx in range(len(grasp_predictions)):
            color_mask = rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            color_mask[color_mask == 0] = 1
            color_mask[color_mask == 255] = 0
            no_target_mask = color_mask
            bg_mask = rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            no_target_mask[bg_mask == 255] = 0
            # only grasp one object
            invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
            invalid_mask = rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, only work for one level
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            check_kernel = np.ones(
                (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
            )
            left_bound = math.floor(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            right_bound = (
                math.ceil(
                    (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
                )
                + 1
            )
            check_kernel[:, left_bound:right_bound] = 0
            for rotate_idx in range(len(grasp_predictions)):
                object_mask = rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
                invalid_mask[invalid_mask > 5] = 255
                invalid_mask = rotate(
                    invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
                )
                grasp_predictions[rotate_idx][invalid_mask > 128] = 0
        grasp_predictions = grasp_predictions[
            :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
        ]

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return grasp_q_value, best_pix_ind, grasp_predictions

    def crop_fixed_tile_np(self, img, center_xy, tile=56, pad_value=0):
        H, W = img.shape[:2]
        half = tile // 2
        cx, cy = center_xy
        cx, cy = int(round(cx)), int(round(cy))

        # NOTE: numpy indexing: [y, x]
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half

        src_y0, src_y1 = max(0, y0), min(H, y1)
        src_x0, src_x1 = max(0, x0), min(W, x1)

        tile_img = img[src_y0:src_y1, src_x0:src_x1].copy()

        pad_top = src_y0 - y0
        pad_left = src_x0 - x0
        pad_bottom = y1 - src_y1
        pad_right = x1 - src_x1

        if img.ndim == 3:
            tile_img = np.pad(tile_img, ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),
                            mode="constant", constant_values=pad_value)
            tile_img = tile_img[:tile, :tile, :]
        else:
            tile_img = np.pad(tile_img, ((pad_top,pad_bottom),(pad_left,pad_right)),
                            mode="constant", constant_values=pad_value)
            tile_img = tile_img[:tile, :tile]
        return tile_img

    @torch.no_grad()
    def get_grasp_q_tile_56(self, color_heightmap, depth_heightmap, segm_map, post_checking=False, is_real=False):
        device = "cuda"
        # non-tgt segm labels (10 objs): [60  70  80  90 100 110 120 130 140], tgt segm label (1 obj): 255
        # print("segm unique:", np.unique(segm_map))
        mask_uint8 = (segm_map == 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[ERROR] Could find the target mask!!")
            return None 

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        tgt_center = rect[0]  # (cx, cy) in x,y

        tile_length = 56
        grid_n = 4  # 4x4 grid for R=4 rotations

        # ---- fixed crops (pad-safe) ----
        rgb_tile_u8 = self.crop_fixed_tile_np(color_heightmap, tgt_center, tile=tile_length, pad_value=0)   # (112,112,3)
        dep_tile    = self.crop_fixed_tile_np(depth_heightmap, tgt_center, tile=tile_length, pad_value=0)   # (112,112)
        seg_tile    = self.crop_fixed_tile_np(segm_map, tgt_center, tile=tile_length, pad_value=0) # (112, 112)

        # ---- to torch ----
        # RGB: (1,3,56,56) in [0,1]
        rgb = torch.from_numpy(rgb_tile_u8).to(device)
        rgb = rgb.float().permute(2,0,1).unsqueeze(0) / 255.0

        # Depth: (1,1,56,56) float (keep in native scale; normalize later if needed)
        dep = torch.from_numpy(dep_tile).to(device)
        dep = dep.float().unsqueeze(0).unsqueeze(0)  # (1,1,56,56)

        seg = torch.from_numpy(seg_tile).to(device)
        seg = seg.int().float().unsqueeze(0).unsqueeze(0) # (1, 1, 56, 56)

        # ---- repeat to (R,C,H,W) ----
        R = 16 # NUM_ROTATION
        rgb_rep = rgb.repeat(R, 1, 1, 1)  # (R,3,56,56)
        dep_rep = dep.repeat(R, 1, 1, 1)  # (R,1,56,56)
        seg_rep = seg.repeat(R, 1, 1, 1)  # (R,1,56,56)
        # ---- build grid ONCE and reuse ----

        # self.theta must be (R,2,3) on correct device/dtype
        theta = self.theta.to(device=device, dtype=torch.float32)  # (R,2,3)
        # theta = theta[:,:,:]  # in case you have more precomputed rotations
        # print("Theta shape:", theta)
        grid = F.affine_grid(theta, rgb_rep.size(), align_corners=True)  # (R,56,56,2)

        # ---- sample both with same grid ----
        rgb_rot = F.grid_sample(rgb_rep, grid, mode="nearest", align_corners=True)  # (R,3,56,56)
        dep_rot = F.grid_sample(dep_rep, grid, mode="nearest",  align_corners=True)  # (R,1,56,56)
        seg_rot = F.grid_sample(seg_rep, grid, mode="nearest", align_corners=True) # (R, 1, 56, 56)
        # print("1. Rotated rgb_rot shape:", rgb_rot.shape, "dep_rot shape:", dep_rot.shape, "seg_rot shape:", seg_rot.shape)

        debug = False
        if debug:
            # ---- back to numpy ----
            rgb_rot_debug = rgb_rot.permute(0,2,3,1).clamp(0,1).cpu().numpy()
            rgb_u8  = np.clip(rgb_rot_debug * 255.0 + 0.5, 0, 255).astype(np.uint8)  # (R,56,56,3)

            dep_rot_debug = dep_rot[:,0,:,:].cpu().numpy()  # (R,56,56) float

            seg_rot_debug = seg_rot[:,0,:,:].cpu().numpy().astype(np.uint8) # (R, 56, 56) float

            # ---- assemble crops in a grid ----
            rgb_grid = np.zeros((grid_n*tile_length, grid_n*tile_length, 3), dtype=np.uint8)
            dep_grid = np.zeros((grid_n*tile_length, grid_n*tile_length), dtype=dep_rot_debug.dtype)
            seg_grid = np.zeros((grid_n*tile_length, grid_n*tile_length), dtype=np.uint8)
            for k in range(R):
                r, c = divmod(k, grid_n)
                y0, y1 = r*tile_length, (r+1)*tile_length
                x0, x1 = c*tile_length, (c+1)*tile_length
                rgb_grid[y0:y1, x0:x1] = rgb_u8[k]
                dep_grid[y0:y1, x0:x1] = dep_rot_debug[k]
                seg_grid[y0:y1, x0:x1] = seg_rot_debug[k]

            # Optional: save depth visualization (depth itself should not be cast to uint8 blindly)
            # Example visualization:
            dep_vis = dep_grid.copy()
            dep_vis = dep_vis - np.nanmin(dep_vis)
            if np.nanmax(dep_vis) > 1e-6:
                dep_vis = dep_vis / np.nanmax(dep_vis)
            dep_vis_u8 = (dep_vis * 255.0 + 0.5).astype(np.uint8)
            save_path = "<debug output dir>/spiral_grid/000000.txt/color"
            print("Saving tiled crops to:", save_path)
            # cv2.imwrite(f"{save_path}/depth_cropped_tiled_vis.png", dep_vis_u8)
            cv2.imwrite(f"{save_path}/depth_cropped_tiled_{tile_length}px.png", dep_vis)
            cv2.imwrite(f"{save_path}/color_cropped_tiled_{tile_length}px.png", cv2.cvtColor(rgb_grid, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{save_path}/segm_cropped_tiled_{tile_length}px.png", cv2.cvtColor(seg_grid, cv2.COLOR_RGB2BGR))
              
        # ---- normalizing the input (x - mu)/sigma for the GPN network ----
        rgb_rot[:, 0, :, :] = rgb_rot[:, 0, :, :] - COLOR_MEAN[0]
        rgb_rot[:, 1, :, :] = rgb_rot[:, 1, :, :] - COLOR_MEAN[1]
        rgb_rot[:, 2, :, :] = rgb_rot[:, 2, :, :] - COLOR_MEAN[2]

        rgb_rot[:, 0, :, :] = rgb_rot[:, 0, :, :] / COLOR_STD[0]
        rgb_rot[:, 1, :, :] = rgb_rot[:, 1, :, :] / COLOR_STD[1]
        rgb_rot[:, 2, :, :] = rgb_rot[:, 2, :, :] / COLOR_STD[2]

        dep_rot[:, 0, :, :] = dep_rot[:, 0, :, :] - DEPTH_MEAN[0]
        dep_rot[:, 0, :, :] = dep_rot[:, 0, :, :] / DEPTH_STD[0]

        # print("2. Normalized rgb_rot shape:", rgb_rot.shape, "dep_rot shape:", dep_rot.shape)
        # Convert rgb_rot tensor of shape (R, 3, 112, 112) to tensor of shape (3, R*112,  R*112)
        rgb_rot = rgb_rot.permute(1, 0, 2, 3).contiguous().view(3, grid_n*tile_length, grid_n*tile_length)
        dep_rot = dep_rot.permute(1, 0, 2, 3).contiguous().view(1, grid_n*tile_length, grid_n*tile_length)
        seg_rot = seg_rot.permute(1, 0, 2, 3).contiguous().view(1, grid_n*tile_length, grid_n*tile_length)
        # print("3. Tiled rgb_rot shape:", rgb_rot.shape, "dep_rot shape:", dep_rot.shape, "seg_rot shape:", seg_rot.shape)
        # pad rgb_rot (3, 4*56, 4*56) back to (3, 320, 320)
        pad_size = IMAGE_PAD_WIDTH
        rgb_rot_pad = F.pad(rgb_rot, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        dep_rot_pad = F.pad(dep_rot, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        seg_rot_pad = F.pad(seg_rot, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        # print("4. Padded rgb_rot_pad shape:", rgb_rot_pad.shape, "dep_rot_pad shape:", dep_rot_pad.shape, "seg_rot_pad shape:", seg_rot_pad.shape)
         # add batch dim: (1,3,320,320), (1,1,320,320)
        rgb_rot_pad = rgb_rot_pad.unsqueeze(0)  # (1,3,320,320)
        dep_rot_pad = dep_rot_pad.unsqueeze(0)  # (1,1,320,320)
        seg_rot_pad = seg_rot_pad.unsqueeze(0)  # (1,1,320,320) 
        # print("5. [Batched] rgb_rot_pad shape:", rgb_rot_pad.shape, "dep_rot_pad shape:", dep_rot_pad.shape, "seg_rot_pad shape:", seg_rot_pad.shape)
       
        # print("[Grasp Model Input] color img shape:", rgb_rot_pad.shape, "depth img shape:", dep_rot_pad.shape)
        # Pass input data through model
        output_prob = self.grasp_model(rgb_rot_pad, dep_rot_pad) # B, 1, pad_H, pad_W

        # output_prob = self.grasp_model(input_color_data, input_depth_data)
                                    #    , True, -1, False, device=device)
        # print("output_prob.shape:",output_prob.shape)
        grasp_predictions = output_prob[0, 0, :, :,] # pad_H, pad_W
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        
        # print("6. grasp_predictions shape:", grasp_predictions.shape)
        # Only consider grasp candidates on the target mask regions 
        grasp_predictions[seg_rot_pad[0, 0, :, :] != 255] = 0

        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )

        # For each of those grasps, we need further feasibility and collision-checks
        # Set non-tgt obj (B, 1, pad_H, pad_W) masks to 1 and tgt + background pixels to zero
        seg_non_tgt = torch.where((seg_rot_pad != 255) & (seg_rot_pad != 0), torch.ones_like(seg_rot_pad), torch.zeros_like(seg_rot_pad))
        kh = GRIPPER_GRASP_SAFE_WIDTH_PIXEL
        kw = GRIPPER_GRASP_INNER_DISTANCE_PIXEL
        x = seg_non_tgt.to(torch.float32) # Use float for convolution
        # Create an all-ones kernel like np.ones((kh, kw))
        weight = torch.ones((1, 1, kh, kw), device=x.device, dtype=x.dtype)
        pad_left   = kw // 2
        pad_top    = kh // 2
        pad_right  = kw - 1 - pad_left
        pad_bottom = kh - 1 - pad_top
        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        tgt_boundary_collision_mask = F.conv2d(x_pad, weight, stride=1, padding=0)  # (B, 1, H, W)
        

        mask_small = tgt_boundary_collision_mask > threshold_small
        mask_big   = tgt_boundary_collision_mask > threshold_big
        # print("mask_small shape:", mask_small.shape)
        grasp_predictions[mask_small[0, 0, :, :]] /= 2
        grasp_predictions[mask_big[0, 0, :, :]] = 0

        
        # non-tgt segm labels (10 objs): [60  70  80  90 100 110 120 130 140], tgt segm label (1 obj): 255
        unique_obj_seg_labels = torch.tensor([60, 70, 80, 90, 100, 110, 120, 130, 140, 255], device=seg_rot_pad.device)

        seg_non_bg_only = torch.where(torch.isin(seg_rot_pad, unique_obj_seg_labels), 
                                   torch.ones_like(seg_rot_pad)*255, torch.zeros_like(seg_rot_pad))  
        okh = GRIPPER_GRASP_WIDTH_PIXEL
        okw = GRIPPER_GRASP_OUTER_DISTANCE_PIXEL
        x_bg = seg_non_bg_only.to(torch.float32) # Use float for convolution
        left_bound = math.floor(
            (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
        )
        right_bound = (
            math.ceil(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            + 1
        )
        # Create an all-ones kernel like np.ones((kh, kw))
        weight_bg = torch.ones((1, 1, okh, okw), device=x_bg.device, dtype=x_bg.dtype)
        weight_bg[:, :, :, left_bound:right_bound] = 0
        pad_left_bg   = okw // 2
        pad_top_bg    = okh // 2
        pad_right_bg  = okw - 1 - pad_left_bg
        pad_bottom_bg = okh - 1 - pad_top_bg
        x_bg_pad = F.pad(x_bg, (pad_left_bg, pad_right_bg, pad_top_bg, pad_bottom_bg), mode="constant", value=0)
        gripper_outer_collision_mask = F.conv2d(x_bg_pad, weight_bg, stride=1, padding=0)  # (B, 1, H, W)
        # print("gripper_outer_collision_mask shape:", gripper_outer_collision_mask.shape)
        gripper_outer_collision_mask[gripper_outer_collision_mask > 5] = 255
        
        grasp_predictions[gripper_outer_collision_mask[0, 0, :, :] > 128] = 0
        
        grasp_predictions = grasp_predictions[padding_width_start:padding_width_end, padding_width_start:padding_width_end]
        # print("7. Final grasp_predictions shape:", grasp_predictions.shape)
        flat_best_px_idx = torch.argmax(grasp_predictions)
        y = flat_best_px_idx // grasp_predictions.shape[1]
        x = flat_best_px_idx % grasp_predictions.shape[1]
        best_pix_ind = (0, y.item(), x.item())  # since batch size is 1
        tile_row = best_pix_ind[1] // tile_length
        tile_col = best_pix_ind[2] // tile_length
        rot = tile_row * grid_n + tile_col  # rotation index
        # print("Best pixel index (y,x):", best_pix_ind[1:], "Rotation index:", rot)
        best_grasp_score = grasp_predictions[best_pix_ind[1], best_pix_ind[2]].item()
        # print("Best grasp score:", best_grasp_score)
        return best_grasp_score, (rot, best_pix_ind[1], best_pix_ind[2]), grasp_predictions
    
    def crop_fixed_tile_torch(self, img, tgt_centers, tile=56, pad_value=0):
        """
        Batched, pad-safe fixed crop around tgt_centers.
        
        inputs:
        img: (B, H, W, C)
        tgt_centers: (B, 2) float (cx, cy)

        outputs:
        cropped_tiles: (B, tile, tile, C)
        """
        device = img.device
        B, H, W, C = img.shape
        half = tile // 2

        # Convert to (B, C, H, W) for padding + indexing 
        if img.ndim == 4:
            img_bchw = img.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            raise ValueError("Expected img with 4 dims (B,H,W,C)")

        # pad so every crop is in-bounds
        pad = [half, half, half, half]  # left, right, top, bottom
        img_padded = F.pad(img_bchw, pad, mode="constant", value=pad_value)  # (B, C, H+2*half, W+2*half)

        # centers in padded coordinates 
        padded_tgt_centers = torch.round(tgt_centers).long() + half  # (B, 2) (cx, cy)

        # build per-batch crop indices 
        # y in [cy-half, cy+half), x in [cx-half, cx+half)]
        y_idxs = torch.arange(-half, half, device=device).view(1, tile)  # (1, tile)
        x_idxs = torch.arange(-half, half, device=device).view(1, tile)  # (1, tile)
        ys = padded_tgt_centers[:, 1].view(B, 1) + y_idxs  # (B, tile)
        xs = padded_tgt_centers[:, 0].view(B, 1) + x_idxs  # (B, tile)  

        # gather using advanced indexing
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, tile, tile)  # (B, tile, tile)
        y_idx2 = ys.view(B, tile, 1).expand(B, tile, tile)  # (B, tile, tile)
        x_idx2 = xs.view(B, 1, tile).expand(B, tile, tile)  # (B, tile, tile)

        # img_padded: (B, C, Hpad, Wpad) -> (B, C, tile, tile)
        cropped_tiles = img_padded[b_idx, :, y_idx2, x_idx2]  # (B, C, tile, tile)
        cropped_tiles = img_padded[b_idx, :, y_idx2, x_idx2]   # (B, tile, tile, C)
        cropped_tiles = cropped_tiles.permute(0, 3, 1, 2).contiguous() # now: (B, C, tile, tile)
        return cropped_tiles
    
    def get_tgt_centers(self, segm_map_bhw, target_label=255):
        """
        segm_map_bhw: (B, H, W) int
        returns:
        tgt_centers: (B, 2) float (cx, cy) for each batch item
        """
        B, H, W = segm_map_bhw.shape
        mask = (segm_map_bhw == target_label) 
        ys = torch.arange(H, device=segm_map_bhw.device).view(1, H, 1).expand(B, H, W)
        xs = torch.arange(W, device=segm_map_bhw.device).view(1, 1, W).expand(B, H, W)

        m = mask.to(torch.float32)
        denom = m.sum(dim=(1,2))  # (B,)

        # Avoid division by zero
        denom_safe = torch.clamp(denom, min=1.0)

        cx = (m * xs).sum(dim=(1,2)) / denom_safe  # (B,)
        cy = (m * ys).sum(dim=(1,2)) / denom_safe  # (B,)

        tgt_centers = torch.stack([cx, cy], dim=1)  # (B, 2)
        valid = denom > 0 
        return tgt_centers, valid     

    @torch.no_grad()
    def get_grasp_q_parallel_Bx16(self, rgb, dep, seg, post_checking=False, is_real=False):
        device = "cuda"
        # non-tgt segm labels (10 objs): [60  70  80  90 100 110 120 130 140], tgt segm label (1 obj): 255
        # RGB: (B,224,224, 3) to (B,3,224,224)
        rgb = rgb.float().permute(0, 3, 1, 2) # (B,3,224,224)

        # Depth: (B,224,224) to (B,1,224,224)
        dep = dep.float().unsqueeze(1)  # (B,1,224,224)

        seg = seg.int().float().unsqueeze(1)  # (B,1,224,224)
        tgt_centers, valid = self.get_tgt_centers(segm_map_bhw=seg.squeeze(1), target_label=255)
        if not torch.all(valid):
            raise ValueError("Target center not found in some batch items!")
            return None

        tile = 224

        rgb_crop_bctt = self.crop_fixed_tile_torch(rgb.permute(0,2,3,1), tgt_centers, tile=tile, pad_value=0)  # (B,3,t,t)
        dep_crop_bctt    = self.crop_fixed_tile_torch(dep.permute(0,2,3,1), tgt_centers, tile=tile, pad_value=0)  # (B,1,t,t)
        seg_crop_bctt    = self.crop_fixed_tile_torch(seg.permute(0 ,2,3,1), tgt_centers, tile=tile, pad_value=0)  # (B,1,t,t)
        B, C, T, _ = rgb_crop_bctt.shape
        
        # expand across batch dim -> (B*R, 2 ,3)
        theta_batch = self.theta.unsqueeze(0).expand(B, NUM_ROTATION, 2, 3).contiguous().view(B*NUM_ROTATION, 2, 3)  # (B*R,2,3)
         # repeat input across rotation dim -> (B*R, C, T, T)
        rgb_crop_bctt_rep = rgb_crop_bctt.unsqueeze(1).expand(B, NUM_ROTATION, C, T, T).contiguous().view(B*NUM_ROTATION, C, T, T)  # (B*R,C,T,T)

        # sample rotated crops
        grid = F.affine_grid(theta_batch, rgb_crop_bctt_rep.size(), align_corners=True)  # (B*R,T,T,2)
        
        rgb_crop_brctt = F.grid_sample(rgb_crop_bctt_rep, grid, mode="nearest", padding_mode="zeros", align_corners=True)  # (B*R,C,T,T)
        dep_crop_brctt = F.grid_sample(dep_crop_bctt.unsqueeze(1).expand(B, NUM_ROTATION, 1, T, T).contiguous().view(B*NUM_ROTATION, 1, T, T), grid, mode="nearest", padding_mode="zeros", align_corners=True)  # (B*R,C,T,T)
        seg_crop_brctt = F.grid_sample(seg_crop_bctt.unsqueeze(1).expand(B, NUM_ROTATION, 1, T, T).contiguous().view(B*NUM_ROTATION, 1, T, T), grid, mode="nearest", padding_mode="zeros", align_corners=True)  # (B*R,C,T,T)

        rgb_crop_brctt = rgb_crop_brctt / 255.0
        # ---- normalizing the input (x - mu)/sigma for the GPN network ----
        rgb_crop_brctt[:, 0, :, :] = rgb_crop_brctt[:, 0, :, :] - COLOR_MEAN[0]
        rgb_crop_brctt[:, 1, :, :] = rgb_crop_brctt[:, 1, :, :] - COLOR_MEAN[1]
        rgb_crop_brctt[:, 2, :, :] = rgb_crop_brctt[:, 2, :, :] - COLOR_MEAN[2]

        rgb_crop_brctt[:, 0, :, :] = rgb_crop_brctt[:, 0, :, :] / COLOR_STD[0]
        rgb_crop_brctt[:, 1, :, :] = rgb_crop_brctt[:, 1, :, :] / COLOR_STD[1]
        rgb_crop_brctt[:, 2, :, :] = rgb_crop_brctt[:, 2, :, :] / COLOR_STD[2]

        dep_crop_brctt[:, 0, :, :] = dep_crop_brctt[:, 0, :, :] - DEPTH_MEAN[0]
        dep_crop_brctt[:, 0, :, :] = dep_crop_brctt[:, 0, :, :] / DEPTH_STD[0]

        
        grasp_predictions = self.grasp_model(rgb_crop_brctt, dep_crop_brctt) # B*R, 1, pad_H, pad_W
          
        grasp_predictions = grasp_predictions[:, 0, :, :] # (B*R, pad_H, pad_W)
        IMAGE_PAD_WIDTH = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions.shape[1] - IMAGE_PAD_WIDTH
        
        grasp_predictions[seg_crop_brctt[:, 0, :, :] != 255] = 0

        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )

        # For each of those grasps, we need further feasibility and collision-checks
        # Set non-tgt obj (B*R, 1, pad_H, pad_W) masks to 1 and tgt + background pixels to zero
        seg_non_tgt = torch.where((seg_crop_brctt != 255) & (seg_crop_brctt != 0), torch.ones_like(seg_crop_brctt), torch.zeros_like(seg_crop_brctt))
        kh = GRIPPER_GRASP_SAFE_WIDTH_PIXEL
        kw = GRIPPER_GRASP_INNER_DISTANCE_PIXEL
        x = seg_non_tgt.to(torch.float32) # Use float for convolution
        # Create an all-ones kernel like np.ones((kh, kw))
        weight = torch.ones((1, 1, kh, kw), device=x.device, dtype=x.dtype)
        pad_left   = kw // 2
        pad_top    = kh // 2
        pad_right  = kw - 1 - pad_left
        pad_bottom = kh - 1 - pad_top
        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        tgt_boundary_collision_mask = F.conv2d(x_pad, weight, stride=1, padding=0)  # (B, 1, H, W)

        mask_small = tgt_boundary_collision_mask > threshold_small
        mask_big   = tgt_boundary_collision_mask > threshold_big
        grasp_predictions[mask_small[:, 0, :, :]] /= 2
        grasp_predictions[mask_big[:, 0, :, :]] = 0

        
        # non-tgt segm labels (10 objs): [60  70  80  90 100 110 120 130 140], tgt segm label (1 obj): 255
        unique_obj_seg_labels = torch.tensor([60, 70, 80, 90, 100, 110, 120, 130, 140, 255], device=seg_crop_brctt.device)

        seg_non_bg_only = torch.where(torch.isin(seg_crop_brctt, unique_obj_seg_labels), 
                                   torch.ones_like(seg_crop_brctt)*255, torch.zeros_like(seg_crop_brctt))  
        okh = GRIPPER_GRASP_WIDTH_PIXEL
        okw = GRIPPER_GRASP_OUTER_DISTANCE_PIXEL
        x_bg = seg_non_bg_only.to(torch.float32) # Use float for convolution
        left_bound = math.floor(
            (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
        )
        right_bound = (
            math.ceil(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            + 1
        )
        # Create an all-ones kernel like np.ones((kh, kw))
        weight_bg = torch.ones((1, 1, okh, okw), device=x_bg.device, dtype=x_bg.dtype)
        weight_bg[:, :, :, left_bound:right_bound] = 0
        pad_left_bg   = okw // 2
        pad_top_bg    = okh // 2
        pad_right_bg  = okw - 1 - pad_left_bg
        pad_bottom_bg = okh - 1 - pad_top_bg
        x_bg_pad = F.pad(x_bg, (pad_left_bg, pad_right_bg, pad_top_bg, pad_bottom_bg), mode="constant", value=0)
        gripper_outer_collision_mask = F.conv2d(x_bg_pad, weight_bg, stride=1, padding=0)  # (B, 1, H, W)

        gripper_outer_collision_mask[gripper_outer_collision_mask > 5] = 255
        
        # gripper_outer_collision_mask_unrotated = F.grid_sample(gripper_outer_collision_mask, unrotate_grid, mode="nearest", align_corners=True) # (R, 1, 224, 224)
        grasp_predictions[gripper_outer_collision_mask[:, 0, :, :] > 128] = 0
        # grasp_predictions[gripper_outer_collision_mask[:, 0, :, :] > 128] = 0
        
        grasp_predictions = grasp_predictions[:, padding_width_start:padding_width_end, padding_width_start:padding_width_end]
        # print("grasp_predictions shape:", grasp_predictions.shape)
        rot_q = grasp_predictions.flatten(1).amax(dim=1)      # (B*R,)
        rot_q = rot_q.view(B, NUM_ROTATION)                              # (B,R)
        q_values = rot_q.amax(dim=1)                          # (B,)
        return q_values

    @torch.no_grad()
    def get_grasp_q_parallel_x16(self, color_heightmap, depth_heightmap, seg, post_checking=False, is_real=False):
        device = "cuda"
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)
        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(
            depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )
        seg_pad = np.pad(
            seg, IMAGE_PAD_WIDTH, "constant", constant_values=0
            )
        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        seg_pad.shape = (seg_pad.shape[0], seg_pad.shape[1], 1)
        input_seg_image = np.copy(seg_pad)  
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]
        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_seg_image.shape = (
            input_seg_image.shape[0],
            input_seg_image.shape[1],
            input_seg_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(
            3, 2, 0, 1 # (B,3,H,W)
        )
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(
            3, 2, 0, 1 # (B,1,H,W)
        )
        seg_data = torch.from_numpy(input_seg_image.astype(np.float32)).permute(3, 2, 0, 1) # (B,1,H,W)
        R = NUM_ROTATION
        rgb_rep = input_color_data.repeat(R, 1, 1, 1)  # (R,3,224,224)
        dep_rep = input_depth_data.repeat(R, 1, 1, 1)  # (R,1,224,224)
        seg_rep = seg_data.repeat(R, 1, 1, 1)  # (R,1,224,224)

        grid = F.affine_grid(self.theta, rgb_rep.size(), align_corners=True)  # (R,224,224,2)

        # ---- sample both with same grid ----
        rgb_rot_pad = F.grid_sample(rgb_rep.to(self.device), grid, mode="nearest", align_corners=True)  # (R,3,224,224)
        dep_rot_pad = F.grid_sample(dep_rep.to(self.device), grid, mode="nearest",  align_corners=True)  # (R,1,224,224)
        seg_rot_pad = F.grid_sample(seg_rep.to(self.device), grid, mode="nearest", align_corners=True) # (R, 1, 224, 224)

        grasp_predictions = self.grasp_model(rgb_rot_pad, dep_rot_pad) # B, 1, pad_H, pad_W
        # print("Max grasp score from the model:", grasp_predictions.max().item())

        grasp_predictions = grasp_predictions[:, 0, :, :] # (R, pad_H, pad_W)
        # print("output_prob.shape:",grasp_predictions.shape)
        
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions.shape[1] - IMAGE_PAD_WIDTH
        
        # print("6. grasp_predictions shape:", grasp_predictions.shape)
        # Only consider grasp candidates on the target mask regions 
        grasp_predictions[seg_rot_pad[:, 0, :, :] != 255] = 0

        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )

        # For each of those grasps, we need further feasibility and collision-checks
        # Set non-tgt obj (B, 1, pad_H, pad_W) masks to 1 and tgt + background pixels to zero
        seg_non_tgt = torch.where((seg_rot_pad != 255) & (seg_rot_pad != 0), torch.ones_like(seg_rot_pad), torch.zeros_like(seg_rot_pad))
        kh = GRIPPER_GRASP_SAFE_WIDTH_PIXEL
        kw = GRIPPER_GRASP_INNER_DISTANCE_PIXEL
        x = seg_non_tgt.to(torch.float32) # Use float for convolution
        # Create an all-ones kernel like np.ones((kh, kw))
        weight = torch.ones((1, 1, kh, kw), device=x.device, dtype=x.dtype)
        pad_left   = kw // 2
        pad_top    = kh // 2
        pad_right  = kw - 1 - pad_left
        pad_bottom = kh - 1 - pad_top
        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        tgt_boundary_collision_mask = F.conv2d(x_pad, weight, stride=1, padding=0)  # (B, 1, H, W)
       
        mask_small = tgt_boundary_collision_mask > threshold_small
        mask_big   = tgt_boundary_collision_mask > threshold_big

        # print("mask_small shape:", mask_small.shape)
        grasp_predictions[mask_small[:, 0, :, :]] /= 2
        grasp_predictions[mask_big[:, 0, :, :]] = 0

        
        # non-tgt segm labels (10 objs): [60  70  80  90 100 110 120 130 140], tgt segm label (1 obj): 255
        unique_obj_seg_labels = torch.tensor([60, 70, 80, 90, 100, 110, 120, 130, 140, 255], device=seg_rot_pad.device)

        seg_non_bg_only = torch.where(torch.isin(seg_rot_pad, unique_obj_seg_labels), 
                                   torch.ones_like(seg_rot_pad)*255, torch.zeros_like(seg_rot_pad))  
        okh = GRIPPER_GRASP_WIDTH_PIXEL
        okw = GRIPPER_GRASP_OUTER_DISTANCE_PIXEL
        x_bg = seg_non_bg_only.to(torch.float32) # Use float for convolution
        left_bound = math.floor(
            (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
        )
        right_bound = (
            math.ceil(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            + 1
        )
        # Create an all-ones kernel like np.ones((kh, kw))
        weight_bg = torch.ones((1, 1, okh, okw), device=x_bg.device, dtype=x_bg.dtype)
        weight_bg[:, :, :, left_bound:right_bound] = 0
        pad_left_bg   = okw // 2
        pad_top_bg    = okh // 2
        pad_right_bg  = okw - 1 - pad_left_bg
        pad_bottom_bg = okh - 1 - pad_top_bg
        x_bg_pad = F.pad(x_bg, (pad_left_bg, pad_right_bg, pad_top_bg, pad_bottom_bg), mode="constant", value=0)
        gripper_outer_collision_mask = F.conv2d(x_bg_pad, weight_bg, stride=1, padding=0)  # (B, 1, H, W)

        gripper_outer_collision_mask[gripper_outer_collision_mask > 5] = 255
        
        grasp_predictions[gripper_outer_collision_mask[:, 0, :, :] > 128] = 0
        # print("Max grasp score after collision check:", grasp_predictions.max().item())
        grasp_predictions = grasp_predictions[:, padding_width_start:padding_width_end, padding_width_start:padding_width_end]
        # print("7. Final grasp_predictions shape:", grasp_predictions.shape)
        flat_idx = torch.argmax(grasp_predictions)
        # Convert flat index to (b, h, w)
        b = flat_idx // (grasp_predictions.shape[1] * grasp_predictions.shape[2])
        rem = flat_idx % (grasp_predictions.shape[1] * grasp_predictions.shape[2])
        h = rem // grasp_predictions.shape[2]
        w = rem % grasp_predictions.shape[2]

        best_pix_ind = (b, h, w)

        # Extract the value
        grasp_q_value = grasp_predictions[best_pix_ind]

        # best pix index needs to be rotated back to original unrotated image coord frame. 

        # Un-rotate the winning pixel about the CENTRE OF THIS IMAGE. The size
        # was hardcoded 224 (the legacy workspace heightmap), so on the 320 px
        # camera canvas the point was rotated about (112,112) instead of
        # (160,160). That misplaces the grasp by 2*|dc|*sin(theta/2) with
        # |dc| = 48*sqrt(2) px = 135.8 mm -- up to 272 mm at theta = 180 deg,
        # and it matched all 16 rotation bins in the open-loop trajectories.
        # Take the dims from the cropped predictions instead;
        # for a genuine 224 render this is identical to the old behaviour.
        _h, _w = grasp_predictions.shape[1], grasp_predictions.shape[2]
        x_rot, y_rot = self.rotate_image_point(best_pix_ind[2].item(), best_pix_ind[1].item(), _w, _h, math.radians(best_pix_ind[0].item() * 360 / 16))
        best_pix_ind_unrot = torch.tensor((b, int(y_rot), int(x_rot)), device=grasp_predictions.device)
        return grasp_q_value, best_pix_ind_unrot, grasp_predictions
    # , raw_grasp_predictions

    @torch.no_grad()
    def get_grasp_q(self, color_heightmap, depth_heightmap, segm_map, post_checking=False, is_real=False):
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)
        # print("color heightpad shape:", color_heightmap_pad.shape)
        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(
            depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )

        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )
        # print("[Grasp Model Input] color img shape:", input_color_data.shape, "depth img shape:", input_depth_data.shape)
        # Pass input data through model
        output_prob = self.grasp_model_seq(input_color_data, input_depth_data)
        # For changed grasp net with no separate rotation outputs
        # output_prob = [[None, output_prob], [None, output_prob]]
                                    #    , True, -1, False, device=device)
        # print("output_prob:", output_prob)
        # print('len(output_prob):', len(output_prob))
        # print("output_prob[0].shape:",output_prob[0].shape)
        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,]
                )
            else:
                grasp_predictions = np.concatenate(
                    (
                        grasp_predictions,
                        output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, :, :,],
                    ),
                    axis=0,
                )
        raw_grasp_predictions = np.copy(grasp_predictions)
        # post process, only grasp one object, focus on blue object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        # mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        mask = np.copy(segm_map)
        # Mask stats
        mask_uint8 = (np.array(mask) > 0).astype(np.uint8)
        area = mask_uint8.sum()

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w,h = None, None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (w, h) = rect[1]
                
            min_edge, max_edge = min(w, h), max(w, h)   
            bbox_area = h * w

            aspect_ratio = min(w, h) / max(w, h) if h > 0 else 0
            extent = area / bbox_area if bbox_area > 0 else 0

            # contour-based features
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

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
            # print("Center (cx, cy):", rect[0])
            # print("mask_stats:", mask_stats)
        else:
            print("Target recognition by color was not possible!!")
            
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # Save the image 
        # cv2.imwrite("<debug output dir>/spiral_grid/000000.txt/mask_bg_pad.png", mask_bg_pad)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask_pad != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones(
            (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
        )
        # print("2D Kernel Big shape:", kernel_big.shape)
        if (
            is_real
        ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 5
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            )
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )
        # print("threshold_big:", threshold_big, "threshold_small:", threshold_small)
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
        # save_path = "<debug output dir>/spiral_grid/000000.txt"
        for rotate_idx in range(len(grasp_predictions)):
            # print("mask_pad unique:", np.unique(mask_pad))
            color_mask = rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            color_mask[color_mask == 0] = 1
            color_mask[color_mask == 255] = 0
            no_target_mask = color_mask
            # print("mask bg pad unique:", np.unique(mask_bg_pad))
            bg_mask = rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            no_target_mask[bg_mask == 255] = 0
            # print("no_target_mask unique:", np.unique(no_target_mask))
            non_tgt_mask_vis = no_target_mask.copy()
            non_tgt_mask_vis[non_tgt_mask_vis == 1] = 255
            # cv2.imwrite(f"{save_path}/non-tgt-mask-{rotate_idx}.png", non_tgt_mask_vis)

            # only grasp one object
            invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
            invalid_mask = rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)

            # print("[after conv] non-tgt mask shape:", invalid_mask.shape)
            # conv_mask_vis = invalid_mask.copy()

            # cv2.imwrite(f"{save_path}/invalid-mask-{rotate_idx}.png", conv_mask_vis)

            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, only work for one level
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            check_kernel = np.ones(
                (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
            )
            left_bound = math.floor(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            right_bound = (
                math.ceil(
                    (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
                )
                + 1
            )
            check_kernel[:, left_bound:right_bound] = 0
            # print("check kernel shape:", check_kernel.shape)
            # print("left bound:", left_bound, "right bound:", right_bound)
            for rotate_idx in range(len(grasp_predictions)):
                object_mask = rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
                invalid_mask[invalid_mask > 5] = 255
                invalid_mask = rotate(
                    invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
                )
                grasp_predictions[rotate_idx][invalid_mask > 128] = 0
        # print("grasp_predictions shape:", grasp_predictions.shape)
        # print("padding_width_start:", padding_width_start, "padding_width_end:", padding_width_end)
        grasp_predictions = grasp_predictions[
            :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
        ]

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return torch.tensor(grasp_q_value, device="cuda"), torch.tensor(best_pix_ind, device="cuda"), torch.tensor(grasp_predictions, device="cuda"), torch.tensor(raw_grasp_predictions, device="cuda")

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, is_push=False,
                           tile_size=DEFAULT_TILE_SIZE, target_mask=None, threshold=0.7):
        """Magnified Q tiles with full-mat rotation insets; inputs are untouched."""
        return render_prediction_grid(
            predictions, color_heightmap, best_pix_ind, is_push=is_push,
            tile_size=tile_size, target_mask=target_mask, threshold=threshold,
        )
    

@torch.no_grad()
def from_maskrcnn(model, color_image, device, plot=False):
    """
    Use Mask R-CNN to do instance segmentation and output masks in binary format.
    Assume it works in real world
    """
    image = color_image.copy()
    image = TF.to_tensor(image)
    prediction = model([image.to(device)])[0]
    final_mask = np.zeros((720, 1280), dtype=np.uint8)
    labels = {}
    if plot:
        pred_mask = np.zeros((720, 1280), dtype=np.uint8)
    for idx, mask in enumerate(prediction["masks"]):
        # TODO, 0.9 can be tuned
        threshold = 0.7
        if prediction["scores"][idx] > threshold:
            # get mask
            img = mask[0].mul(255).byte().cpu().numpy()
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # too small
            if np.sum(img == 255) < 100:
                continue
            # overlap IoU 70%
            if np.sum(np.logical_and(final_mask > 0, img == 255)) > np.sum(img == 255) * 3 / 4:
                continue
            fill_pixels = np.logical_and(final_mask == 0, img == 255)
            final_mask[fill_pixels] = idx + 1
            labels[(idx + 1)] = prediction["labels"][idx].cpu().item()
            if plot:
                pred_mask[img > 0] = prediction["labels"][idx].cpu().item() * 10
                cv2.imwrite(str(idx) + "mask.png", img)
    if plot:
        cv2.imwrite("pred.png", pred_mask)
    print("Mask R-CNN: %d objects detected" % (len(np.unique(final_mask)) - 1), prediction["scores"].cpu())
    return final_mask, labels
