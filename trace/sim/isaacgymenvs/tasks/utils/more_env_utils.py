########################################################################################################

# Title: Fast Object Retrieval from Clutter using Continuous Pushing and Grasping

import torch 

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

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quaternion_to_yaw_rotation_matrix(quat: torch.Tensor):
    """ Convert quaternion (batch of shape (N, 4)) to 2D rotation matrices (N, 2, 2). """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Compute the 2D rotation angle from the quaternion (assuming a pure Z-axis rotation)
    theta = 2 * torch.atan2(torch.sqrt(x**2 + y**2 + z**2), w)  # Extract rotation angle
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Create 2D rotation matrices
    rot_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=-1),
        torch.stack([sin_theta, cos_theta], dim=-1)
    ], dim=1)  # Shape: (N, 2, 2)

    return rot_matrices

@torch.jit.script
def quaternion_to_roll_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """ Convert quaternion (batch of shape (N, 4)) to 2D rotation matrices (N, 2, 2) for rotation along the X-axis. """
    x, w = quat[:, 0], quat[:, 3]  # Extract X and W components

    # Compute rotation angle around the X-axis
    theta = 2 * torch.atan2(x, w)  # Extract rotation angle
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Create 2D rotation matrices (for Y-Z plane rotation)
    rot_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=-1),
        torch.stack([sin_theta, cos_theta], dim=-1)
    ], dim=1)  # Shape: (N, 2, 2)

    return rot_matrices

@torch.jit.script
def rotate_rectangles(rects: torch.Tensor, quats: torch.Tensor, axis: int):
    """
    Rotate rectangles around their centers using quaternions.
    
    Args:
        rects: Tensor of shape (1000, 5, 2), where first entry is the center, and the next four are corners.
        quats: Tensor of shape (1000, 4) representing quaternions.
        
    Returns:
        Tensor of shape (1000, 5, 2) with rotated corner coordinates (center unchanged).
    """
    centers = rects[:, 0, :].clone()  # Shape: (1000, 2), remains unchanged
    corners = rects[:, 1:, :].clone()  # Shape: (1000, 4, 2)
    
    if axis == 2:
        # Convert quaternion to 2D rotation matrix
        rot_matrices = quaternion_to_yaw_rotation_matrix(quats)  # Shape: (1000, 2, 2)
    elif axis == 0:
        # Convert quaternion to 2D rotation matrix
        rot_matrices = quaternion_to_roll_rotation_matrix(quats)  # Shape: (1000, 2, 2)
    else:
        raise ValueError("rotation axis is only limited to x & z!")
    # Translate corners to local coordinate system (center at origin)
    corners_relative = corners - centers[:, None, :]  # Shape: (1000, 4, 2)
    
    # Rotate the corners
    # print('rot mat device', rot_matrices.device, 'corners device:', corners_relative.device, "axis:", axis)
    rotated_corners = torch.einsum('nij,nkj->nki', rot_matrices, corners_relative)  # Shape: (1000, 4, 2)
    # rotated_corners = torch.einsum('nij,nkj->nki', rot_matrices, corners_relative)  # Shape: (1000, 4, 2)
    
    # Translate back to the original center
    rotated_corners += centers[:, None, :]
    
    # Concatenate the unchanged center with the rotated corners
    rotated_rects = torch.cat([centers[:, None, :], rotated_corners], dim=1)  # Shape: (1000, 5, 2)
    
    return rotated_rects
@torch.jit.script
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p, q, r)
    # function returns following values
    # 0 -> p, q and r are collinear
    # 1 -> Clockwise
    # 2 -> Counterclockwise
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # if p,q,r are 1000 tensors of shape (1000, 2)
    val = (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]) - (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1])
    
    collinear = torch.zeros_like(val)
    clockwise = torch.ones_like(val)
    anticlockwise = 2 * torch.ones_like(val)

    val = torch.where(val > 0, clockwise, val)
    val = torch.where(val < 0, anticlockwise, val)
    val = torch.where(val == 0, collinear, val)

    return val

@torch.jit.script
def onSegment(p, q, r):
    x_cond = (q[:, 0] <= torch.max(p[:, 0], r[:, 0])) & (q[:, 0] >= torch.min(p[:, 0], r[:, 0]))
    y_cond = (q[:, 1] <= torch.max(p[:, 1], r[:, 1])) & (q[:, 1] >= torch.min(p[:, 1], r[:, 1]))
    return x_cond & y_cond

@torch.jit.script
def doIntersect(p1, q1, p2, q2):
    # Find the four orientations needed for general and
    # special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    cond1 =  (o1 != o2) & (o3 != o4)

    # Special Cases
    cond2 = (o1 == 0) & onSegment(p1, p2, q1)
    cond3 = (o2 == 0) & onSegment(p1, q2, q1)
    cond4 = (o3 == 0) & onSegment(p2, p1, q2)
    cond5 = (o4 == 0) & onSegment(p2, q1, q2)

    return cond1 | cond2 | cond3 | cond4 | cond5


@torch.jit.script
def generate_subrectangles(rects: torch.Tensor, num_scales: int=8):
    """
    Generate `num_scales` concentric scaled rectangles towards the center.
    rects: (N, 4, 2)
    returns: (N, num_scales, 4, 2)
    """
    center = rects.mean(dim=1, keepdim=True)  # (N, 1, 2)
    scales = torch.linspace(1/num_scales, 1.0, num_scales, device=rects.device).view(1, num_scales, 1, 1)
    centered = rects.unsqueeze(1) - center.unsqueeze(1)  # (N, 1, 4, 2)
    scaled = centered * scales + center.unsqueeze(1)
    return scaled  # (N, num_scales, 4, 2)

@torch.jit.script
def do_rects_intersect_fast(A, B):
    """
    A, B: (N, 4, 2) — N rectangles, each with 4 corner points
    Returns: (N,) bool — whether each A[i] intersects B[i]
    """
    A_edges_start = A 
    A_edges_end = torch.roll(A, shifts=-1, dims=1)  # Shift the edges to get the end points

    B_edges_start = B
    B_edges_end = torch.roll(B, shifts=-1, dims=1)  # Shift the edges to get the end points

    
    A1 = A_edges_start.unsqueeze(2).expand(-1, 4, 4, -1)
    A2 = A_edges_end.unsqueeze(2).expand(-1, 4, 4, -1)
    B1 = B_edges_start.unsqueeze(1).expand(-1, 4, 4, -1)
    B2 = B_edges_end.unsqueeze(1).expand(-1, 4, 4, -1)

    N = A.shape[0]
    A1 = A1.reshape(N * 16, 2)
    A2 = A2.reshape(N * 16, 2)
    B1 = B1.reshape(N * 16, 2)
    B2 = B2.reshape(N * 16, 2)

    intersections = doIntersect(A1, A2, B1, B2).reshape(N, 16)

    # Check any intersecting side
    # print("intersections.any(dim=1):", intersections.any(dim=1))
    return intersections.any(dim=1)

@torch.jit.script
def compute_iou(Gripper: torch.Tensor, Non_Target: torch.Tensor, num_scales: int=8) -> torch.Tensor:
    """
    Compute approximate IoU between pairs of polygons A and B on GPU.
    Args:
        A, B: Tensors of shape (N, 4, 2), corners of rectangles
    Returns:
        Tensor of shape (N, 1), IoU per pair
    """
    Gripper = Gripper
    Non_Target = Non_Target
    assert Gripper.shape[1] == 4 and Gripper.shape[2] == 2, f"Gripper should be of shape (N, 4, 2), but it is {Gripper.shape}"
    assert Non_Target.shape[1] == 4 and Non_Target.shape[2] == 2, f"Non_Target should be of shape (N, 4, 2), but it is {Non_Target.shape}"
    Gripper_sub_rects = generate_subrectangles(Gripper, num_scales=num_scales) # N, num_scales, 4, 2
    assert Gripper_sub_rects.shape[1] == num_scales, "Gripper_sub_rects should be of shape (N, num_scales, 4, 2)"
    free_area = torch.zeros(Gripper_sub_rects.shape[0], device=Gripper.device, dtype=torch.float32)  # (N,)

    for ij in range(num_scales):
        free_area = torch.where(~do_rects_intersect_fast(Gripper_sub_rects[:, ij, :, :], Non_Target), 
                                             ((ij+1)/(num_scales))**2, free_area)
        
    return free_area.unsqueeze(1)

@torch.jit.script
def compute_free_area_ratio(rectangles: torch.Tensor, gripper_clearance: torch.Tensor, num_scales: int = 8):
    """
    Compute the ratio of free area to total area of the gripper_clearance rectangle after overlaps.

    Args:
    rectangles: Tensor of shape (4, 1000, 4, 2) representing 4 rectangles per environment.
    gripper_clearance: Tensor of shape (1000, 4, 2) representing the clearance area per environment.

    Returns:
    free_area_ratio: Tensor of shape (1000,), representing the ratio for each environment.
    """
    # Get min/max x and y coordinates for gripper clearance
    ious = []
    for i in range(rectangles.shape[0]):
        # Compute IoU for each rectangle in the batch
        iou = compute_iou(Gripper=gripper_clearance[:, 1:, :], Non_Target=rectangles[i, :, 1:, :], num_scales=num_scales)
        ious.append(iou)

    # take max entry of all four ious element-wise
    iou_stacked = torch.stack(ious, dim=0).min(dim=0)  # Shape: (1000, 1)
    iou_min, _ = iou_stacked
    free_area_ratio = iou_min.clamp(min=0, max=1)  # Ensure it's between 0 and 1
    return free_area_ratio.squeeze()
########################################################################################################

