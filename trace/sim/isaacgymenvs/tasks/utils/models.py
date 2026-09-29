
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.tasks.vision.backbone_utils import resnet_fpn_net
from .constants import NUM_ROTATION
from torchvision.utils import save_image
import math 
class reinforcement_net(nn.Module):
    """
    The DQN Network.
    graspnet is the Grasp Network.
    pushnet is the Push Network for the DQN + GN method.
    """

    def __init__(self, device):  # , snapshot=None
        super(reinforcement_net, self).__init__()
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
        
            input_data = torch.cat((input_color_data, input_depth_data), dim=1)  # (B,4,H,W)
            final_grasp_feat = self.graspnet(input_data)
            return final_grasp_feat
        # [:, 0, :, :]  # (B,H,W)
