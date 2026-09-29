########################################################################################################

# Title: Fast Object Retrieval from Clutter using Continuous Pushing and Grasping

import torch


def quaternion_to_euler(q):
    """ Convert quaternion to euler angles (roll, pitch, yaw) """
    """ q shape: (num_envs, 4)"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (rx)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (ry)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))  

    # Yaw (rz)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)  # Shape (num_envs, 3)

def euler_to_quaternion(euler):
    """ Convert euler angles (roll, pitch, yaw) back to quaternion """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([x, y, z, w], dim=-1)  # Shape (num_envs, 4)

def euler_to_quat(roll: float, pitch: float, yaw: float):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qx, qy, qz, qw], dim=-1)
########################################################################################################








