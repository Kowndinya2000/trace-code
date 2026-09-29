
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.tasks.vision.backbone_utils import resnet_fpn_net
from .constants import NUM_ROTATION
from torchvision.utils import save_image
import math 
import cv2
class reinforcement_net_seq(nn.Module):
    """
    The DQN Network.
    graspnet is the Grasp Network.
    pushnet is the Push Network for the DQN + GN method.
    """

    def __init__(self, device):  # , snapshot=None
        super(reinforcement_net_seq, self).__init__()
        self.device = torch.device(device)
        self.num_rotations = NUM_ROTATION
        self.graspnet = resnet_fpn_net("resnet18", trainable_layers=5).to(self.device)
        # print("max_memory_allocated (MB):", torch.cuda.max_memory_allocated() / 2 ** 20)
        # print("memory_allocated (MB):", torch.cuda.memory_allocated() / 2 ** 20)


    def forward(self, input_color_data, input_depth_data):
        # input_color_data: (B,3,H,W)
        # input_depth_data: (B,1,H,W)
        B = input_color_data.size(0)
        device = self.device

        with torch.no_grad():
            self.output_prob = []

            # Move once (avoid repeated .to(device) inside loop)
            input_color_data = input_color_data.to(device)
            input_depth_data = input_depth_data.to(device)

            for rotate_idx in range(self.num_rotations):
                rotate_theta = math.radians(rotate_idx * (360.0 / self.num_rotations))

                # ----- BEFORE rotation grid -----
                c = math.cos(-rotate_theta)
                s = math.sin(-rotate_theta)

                # theta must be (B,2,3)
                affine_mat_before = torch.tensor(
                    [[c,  s, 0.0],
                    [-s, c, 0.0]],
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).repeat(B, 1, 1)  # (B,2,3)

                flow_grid_before = F.affine_grid(
                    affine_mat_before, input_color_data.size(), align_corners=True
                )

                rotate_color = F.grid_sample(
                    input_color_data, flow_grid_before, mode="nearest", align_corners=True
                )
                rotate_depth = F.grid_sample(
                    input_depth_data, flow_grid_before, mode="nearest", align_corners=True
                )
                # cv2.imwrite(f"<debug output dir>/spiral_grid/000000.txt/color/seq_rgb_rot_{rotate_idx}.png",
                #         cv2.cvtColor((rotate_color[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
                input_data = torch.cat((rotate_color, rotate_depth), dim=1)  # (B,4,H,W)
                final_grasp_feat = self.graspnet(input_data)
                # print(f"[SEQ-before-unrotation] rotation idx: {rotate_idx}, max grasp feat value: {final_grasp_feat.max().item()}")
                # ----- AFTER rotation grid (unrotate predictions) -----
                c = math.cos(rotate_theta)
                s = math.sin(rotate_theta)

                affine_mat_after = torch.tensor(
                    [[c,  s, 0.0],
                    [-s, c, 0.0]],
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).repeat(B, 1, 1)  # (B,2,3)

                flow_grid_after = F.affine_grid(
                    affine_mat_after, final_grasp_feat.size(), align_corners=True
                )

                unrotated_feat = F.grid_sample(
                    final_grasp_feat, flow_grid_after, mode="nearest", align_corners=True
                )

                self.output_prob.append([None, unrotated_feat])
                # print(f"[SEQ-after-unrotation] rotation idx: {rotate_idx}, max unrotated feat value: {unrotated_feat.max().item()}")

            # IMPORTANT: return after processing all rotations
            return self.output_prob
