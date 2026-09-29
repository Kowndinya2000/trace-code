# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""More — the ACTIVE Isaac Gym task for the IROS26 reactive pushing policy.

Registered as task "More" in tasks/__init__.py (eval fork: more_test.py
MoreTest). Trained with rl_games a2c_discrete via train.py
(cfg/task/More.yaml + cfg/train/MorePPO.yaml).

- Scene: 11 objects (index 0 = target, segId 255) loaded per env from
  pre-authored files in test-cases/dataset/selected/... (one file per env,
  line format: objname r g b x y z roll pitch yaw). Assets: assets/urdf/more/.
- Observations (compute_observations): per object the 4 rotated bbox corners
  (8 floats) + gripper (x,y) -> 11*8+2 = 90-D (numObservations in More.yaml
  is a dummy overwritten here).
- Actions: 16 discrete EEF-relative planar primitives decoded in
  build_two_step_plan (4 cardinal + 4 diagonal x 3 execution modes); applied
  over sub-steps via damped-least-squares IK (control_ik).
- Reward (compute_more_reward_jit): grasp-network terminal reward (Q > 0.9)
  + distance/convex-hull/idle/backtrack/tight-coupling shaping terms.
  Training uses MCTSHelper.grasp_prob_B (EfficientNet); the batched Bx16
  tiled GPN path exists but is commented at the call site.

Archived forks (4-mps ablations, 16-mps migration twins) are in
../archive/dead_code/ — this file already contains the 16-primitive set.
"""
# Concave Faces: 9cm x 4.5 cm (rect face), 9cm x 4.5 cm (side cavity curve), 9 cm x 4.5 cm(front cavity curve)
# Cylinder Faces: 4.5cm x 4.5cm (top/bottom), 4.5 cm x 4.5cm (lateral/sides)
# Cube faces: 4.5cm x 4.5cm
# half-cube faces: 4.5cm x 4.5cm, 4.5cmx 2.25cm
# Triangle: base - 8.5cm x 4.5cm flipped to side - 4.5cm x 8.5cm
# rect: 9 cm x 4.5 cm
# Gripper: 12cm x 2.2/2.3 cm - you want to achieve a min clearance of this area around the target object in one of the 16 possible rotations (360 deg/16).   
# version1: fix cube,cylinder,half-cube as targets as 16 clearance rectangles are easy to compute.
# default_home: [56.50 (-90 for sim), -113.61, 151.06, -127.36, -89.75, 326.49, 0.0, 0.0]
import time
import numpy as np
import os
from colorama import Fore
from tqdm import tqdm
import torch
# torch.nn.functional
import math
import torch.nn.functional as F
import random 
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_conjugate, quat_mul
from .utils.mtcs_utils import MCTSHelper 

from .utils.more_utils import quaternion_to_euler, euler_to_quat
from .utils.more_jit_utils import (
                                rotate_rectangles,
                                compute_free_area_ratio,
                                point_in_convex_hull_gpu
)
from  .base.vec_task import VecTask
from .constants import (
    IS_REAL,
    IMAGE_OBJ_CROP_SIZE,
    IMAGE_SIZE,
    PIXEL_SIZE,
    # WORKSPACE_LIMITS,
    WORKSPACE_PUSH_BORDER,
    PUSH_LENGTH
)
# ONE workspace: the real robot's reachable 0.448 m square (the real cell), centred
# at (0.5, 0). The OOW rule (more_robust.WS_*), scene generator, symmetry
# centre and camera all derive from this. The table box is larger (TABLE_SIZE)
# so the 0.64 m camera view never shows the Isaac floor.
WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [0.0001, 0.4]])
WORKSPACE_CENTER = (0.5, 0.0)
TABLE_SIZE = 0.76

import glob
import random
random.seed(1600)


class More(VecTask):

    def __init__(self, cfg, test, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # print("===> Initializing More Task <===")
        # print("self.action_space:", self.act_space)
        # print("==============================================")
        # print("===> Initializing More Task <===")
        # print("self.action_space:", self.act_space)
        # print("==============================================")
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.overlap_scale = self.cfg["env"]["overlapScale"]
        self.distance_scale  = self.cfg["env"]["distanceScale"]
        self.target_clearance_reward = self.cfg["env"]["targetClearanceReward"]
        self.desired_eef_to_target_distance = self.cfg["env"]["desiredGripperToTargetDist"]
        self.gripper_to_block_center_dist_threshold = self.cfg["env"]["gripperToBlockThreshold"]
        self.tight_coupling_scale = self.cfg["env"]["tightCouplingScale"]
        self.tight_coupling_reward_scale = self.cfg["env"]["tightCouplingRewardScale"]
        self.num_sub_rects = self.cfg["env"]["geometry"]["numSubRects"]
        # ticks (physics frames) per waypoint phase. K = 1 means a phase is a
        # SINGLE frame, which leaves no window to hold a setpoint in — with
        # K = 1 the absolute/reach_gated modes reduce algebraically to the
        # legacy increment (measured bit-identical, cluster job 248036).
        # A plan is 2 phases, so use controlFrequencyInv >= 2K to finish a plan
        # inside one RL step and avoid dropped actions.
        self.K = max(1, int((self.cfg["env"].get("controller", {}) or {}).get("ticksPerPhase", 1)))
        # controller (cfg.env.controller): "incremental" = legacy behaviour;
        # "absolute" holds one IK setpoint per waypoint phase so the PD can
        # converge. holdOrientation restores the orientation error term the
        # legacy path zeroed out. maxStepM is the per-frame Cartesian clamp.
        # primitive size: total EEF displacement commanded per action, split
        # across the plan's two waypoint phases. 0.02 = the original 2cm step;
        # PMBS uses a 0.10 macro push. Larger primitives lose proportionally
        # less to the accel/decel transient (delivery plateaus ~47% at 2cm no
        # matter how the controller is tuned — cluster job 248067) and shorten
        # the horizon, at the cost of coarser control near the target.
        self.push_total = float(self.cfg["env"].get("pushDistanceM", 0.02))
        _ctrl = self.cfg["env"].get("controller", {}) or {}
        self.ctrl_mode = str(_ctrl.get("mode", "incremental"))
        assert self.ctrl_mode in ("incremental", "absolute", "reach_gated"), self.ctrl_mode
        # reach_gated (PMBS environment.py:1010 pattern): hold the setpoint and
        # only advance the waypoint phase once the joints actually arrive
        # (|err| < reachTolRad) or provably stall (stallFrames with no motion).
        self.ctrl_reach_tol = float(_ctrl.get("reachTolRad", 0.012))
        self.ctrl_stall_frames = int(_ctrl.get("stallFrames", 100))
        self.ctrl_stall_eps = float(_ctrl.get("stallEpsRad", 0.001))
        self.ctrl_hold_orn = bool(_ctrl.get("holdOrientation", False))
        # 0 => auto: one phase length, so a held setpoint can reach its
        # waypoint instead of being clamped short (matters once pushes grow)
        _ms = float(_ctrl.get("maxStepM", 0.01))
        self.ctrl_max_step = (self.push_total / 2.0) if _ms <= 0 else _ms
        self.ctrl_orn_ref = None
        if self.ctrl_mode != "incremental" and \
                int(cfg["env"].get("controlFrequencyInv", 1)) < 2:
            print("[More] WARNING: controller.mode=" + self.ctrl_mode +
                  " with controlFrequencyInv=1 — a 2-phase plan cannot finish in"
                  " one RL step, so every other action is dropped. Use >= 2.")

        self.num_actions = 4

        self.damping = 0.05
        self.num_objects = self.cfg["env"]["numObjects"] # 3, 6  
        # robot
        self.default_joint_angles = self.cfg["robot"]["default_joint_angles"]
        if self.num_objects == 12:
            self.default_joint_angles = [-19.32, -100.29, 147.48, -137.18, -89.72, 160.73, 0, 0]  # Ideal for twelve-obj setup. # forward + exact gripper orientation (EoH Robot URDF)
        self.default_joint_angles = [x*np.pi/180 for x in self.default_joint_angles] 
        # print('default joint angles:', self.default_joint_angles)
        # scene config
        self.test_case_dir = os.path.join(self.cfg["env"]["test_cases"]["scene_root_dir"], self.cfg["env"]["test_cases"]["difficulty_choice"])

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = self.cfg["sim"]["dt"]
      

        self.cfg["env"]["numObservations"] = self.cfg["env"].get(
            "numObservationsOverride", self.num_objects*8 + 2)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
       
        # get gym GPU state tensors
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        # print("dof state tensor shape:", self.dof_state_tensor.shape)
        self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # print("actor root state tensor shape:", self.actor_root_state_tensor.shape)
        
        self.mcts_helper = MCTSHelper(f"logs_grasp/snapshot-post-020000.reinforcement.pth", f"logs_grasp/grasp_model-89.pth", device="cuda" if torch.cuda.is_available() else "cpu")
        self.q_values_parallel = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        self.plan_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ctrl_hold_target = torch.zeros(self.num_envs, 6, device=self.device)
        self.ctrl_prev_dof = torch.zeros(self.num_envs, 6, device=self.device)
        self.ctrl_stall_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # setpoint gate: recompute the held target at an RL-step boundary or a
        # waypoint-phase change. NOT on plan_tick (K = 1 resets it every tick,
        # which silently made the hold identical to the legacy increment).
        self.ctrl_fresh_step = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # per-plan delivery, recorded when a plan completes INSIDE the substep
        # loop; tools sample plan_delivery_seq to detect new completions
        self.plan_delivery_last = torch.zeros(self.num_envs, device=self.device)
        self.plan_done_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.plan_delivery_seq = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ctrl_last_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.plan_phase  = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 0 or 1
        self.plan_tick   = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 0..(K-1)

        self.start_pos   = torch.zeros((self.num_envs, 3), device=self.device)
        self.wp1_pos     = torch.zeros((self.num_envs, 3), device=self.device)
        self.wp2_pos     = torch.zeros((self.num_envs, 3), device=self.device)


        self.refresh_env_tensors()

        self.ur5e_default_dof_pos = to_torch(self.default_joint_angles, device=self.device)
        # self.ur5e_default_dof_pos = torch.deg2rad(self.ur5e_default_dof_pos)
        
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.ur5e_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur5e_dofs]
        self.ur5e_dof_pos = self.ur5e_dof_state[..., 0] # shape: (num_envs, num_ur5e_dofs)
        self.ur5e_dof_vel = self.ur5e_dof_state[..., 1]
        self.ur5e_dof_targets = torch.zeros_like(self.ur5e_dof_pos, device=self.device) # shape: (num_envs, num_ur5e_dofs)

        self.blocks_rect_rotated = torch.zeros((self.num_objects, self.num_envs, 5, 2), device=self.device) # dim:3 -> 0-index is center and rest are four corners of the rectangle
        self.gripper_open_length = self.cfg["env"]["geometry"]["gripperOpenLength"] #  2.2 cms (along X)
        self.gripper_open_width = self.cfg["env"]["geometry"]["gripperOpenWidth"] # 0.12  - 12 cms (along Y), for rot(0, np.pi, 0) along XYZ axes respectively
        
        
        vec_actor_root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.block_state = vec_actor_root_state_tensor[:, 2:] # without wall
        # self.block_state = vec_actor_root_state_tensor[:, 3:] # with wall
        self.robot_root_state = vec_actor_root_state_tensor[:, 1].clone()

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5e") # shape [num_envs, 15, 6, 8]
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = jacobian[:, self.ur5e_gripper_index-1, :, :6]

        # Record indices

        self.global_actor_ids = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )  # 1 workspace + 1 robot + 4 target objects 
        self.global_robot_ids = self.global_actor_ids[:, 1].flatten()
        # self.global_target_obj_ids = self.global_actor_ids[:, 1:5].detach().clone()
        # self.global_robot_target_obj_ids = self.global_actor_ids[:, 0:5].detach().clone()

        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.successes = torch.zeros((self.num_envs), device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        self.prev_gripper_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.grasp_q_parallel_values = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        self.ep_start_wall = np.full(self.num_envs, time.perf_counter(), dtype=np.float64)
        self.success_recorded = torch.zeros(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.success_wall_times = []
        self.success_steps = []
        self.total_eps = 0
        self.inverse_action = torch.tensor(
                        [2,3,0,1, 13,14,15, 10,11,12, 7,8,9, 4,5,6],
                        device=self.device, dtype=torch.long
                    )
        self.prev_action_idx = -1 * torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self.reset_idx(torch.arange(self.num_envs, device=self.device), from_where="init")
        self.reset_buf = torch.zeros((self.num_envs), device=self.device)
        self.progress_buf = torch.zeros((self.num_envs), device=self.device)
        

        self.refresh_env_tensors()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim_params.dt = self.cfg["sim"]["dt"]
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.cfg["sim"]["use_gpu_pipeline"]
        if self.cfg["physics_engine"] == "physx":
            self.cfg["physics_engine"] = gymapi.SIM_PHYSX
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 24
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.001 / 4
            self.sim_params.physx.contact_offset = 0.002 / 4  # TODO: check the comparison between 0.001 and 0.002
            # self.sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP
            self.sim_params.physx.bounce_threshold_velocity = 0.2
            self.sim_params.physx.max_depenetration_velocity = 10
            self.sim_params.physx.friction_offset_threshold = 0.004 / 4
            self.sim_params.physx.friction_correlation_distance = 0.0025 / 4
            self.sim_params.physx.num_threads = self.cfg["sim"]["physx"]["num_threads"]
            self.sim_params.physx.use_gpu = self.cfg["sim"]["physx"]["use_gpu"]          
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.cfg["physics_engine"], self.sim_params)

        self.robot_urdf_option = 0 # 0 - with EoH Cam, 1 - W/o EoH but links for grippers to sense contact force

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)), self.num_objects)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row, num_objects):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.name2id_map = {
            "concave": 0,
            "cylinder": 1,
            "cube" : 2,
            "half-cube": 3,
            "rect": 4,
            "triangle": 5
        }
        self.name_id2_dims_map = {
            0: [9.0/100, 4.5/100], # along, X,Y always (in meters) for Zero rotation along X and Y axes
            1: [4.5/100, 4.5/100],
            2: [4.5/100, 4.5/100],
            3: [2.25/100, 4.5/100],
            4: [4.5/100, 9.0/100],
            5: [4.5/100, 8.5/100]
        }

        asset_root = os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
        # print('asset_root:', asset_root)
        if self.robot_urdf_option == 0:
            ur5e_asset_folder = f"{asset_root}/assets/urdf/more/ur5e_simplified"
        elif self.robot_urdf_option == 1:
            ur5e_asset_folder = f"{asset_root}/assets/urdf/more"
        
        wall_asset_folder = f"{asset_root}/assets/urdf/more/workspace"
        block_asset_folder = f"{asset_root}/assets/urdf/more/blocks-more"

        # load UR5e asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = True # If True, meshes will be switched from Z-UP left handed system to Y-UP right handed system.
        robot_asset_options.fix_base_link = True
        robot_asset_options.disable_gravity = True
        robot_asset_options.thickness = 0.001
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset_options.use_mesh_materials = True
        robot_asset_options.override_com = True 
        robot_asset_options.override_inertia = True

        if self.robot_urdf_option == 0:
            ur5e_asset = self.gym.load_asset(self.sim, ur5e_asset_folder, "ur5e_simplified_gripper_no_eoh.urdf", robot_asset_options)
        elif self.robot_urdf_option == 1:
            ur5e_asset = self.gym.load_asset(self.sim, ur5e_asset_folder, "ur5e/ur5e_gripper.urdf", robot_asset_options)
        ur5e_props = self.gym.get_asset_rigid_shape_properties(ur5e_asset)
        for prop in ur5e_props:
            prop.friction = 0.6
        self.gym.set_asset_rigid_shape_properties(ur5e_asset, ur5e_props)

        ur5e_link_dict = self.gym.get_asset_rigid_body_dict(ur5e_asset)
        if self.robot_urdf_option == 0:
            self.ur5e_gripper_index = ur5e_link_dict["zdummy_gripper_grasp_pose_link"]
        elif self.robot_urdf_option == 1:
            self.ur5e_gripper_index = ur5e_link_dict["dummy_center_indicator_link"]

        # workspace asset
        # dim = WORKSPACE_LIMITS[0][1] - WORKSPACE_LIMITS[0][0]
        # workspace_dims = gymapi.Vec3(dim, dim, 0.001)
        dim_x = TABLE_SIZE
        dim_y = TABLE_SIZE
        print(f"workspace size in pixels x pixels: {dim_x} x {dim_y}")
        workspace_dims = gymapi.Vec3(dim_x, dim_y, 0.001)


        workspace_asset_options = gymapi.AssetOptions()
        workspace_asset_options.flip_visual_attachments = False # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        workspace_asset_options.fix_base_link = True
        workspace_asset_options.linear_damping = 0.5
        workspace_asset_options.angular_damping = 0.5
        workspace_asset = self.gym.create_box(
            self.sim, workspace_dims.x, workspace_dims.y, workspace_dims.z, workspace_asset_options
        )
        workspace_props = self.gym.get_asset_rigid_shape_properties(workspace_asset)
        if IS_REAL:
            workspace_props[0].friction = 1.1
        else:
            workspace_props[0].friction = 1.1

        workspace_props[0].restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(workspace_asset, workspace_props)
        workspace_pose = gymapi.Transform()
        # workspace_pose.p = gymapi.Vec3(0.55, 0.01, 0.0005)
        workspace_pose.p = gymapi.Vec3(WORKSPACE_CENTER[0], WORKSPACE_CENTER[1], 0.0005)

        x_center = workspace_pose.p.x
        y_center = workspace_pose.p.y

        x_min = x_center - workspace_dims.x / 2
        x_max = x_center + workspace_dims.x / 2

        y_min = y_center - workspace_dims.y / 2
        y_max = y_center + workspace_dims.y / 2

        print(f"workspace x limits: {x_min} to {x_max}")
        print(f"workspace y limits: {y_min} to {y_max}")

        # wall_asset_options = gymapi.AssetOptions()
        # wall_asset_options.flip_visual_attachments = False # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        # wall_asset_options.fix_base_link = True
        # wall_asset = self.gym.load_asset(self.sim, wall_asset_folder, "wall.urdf", wall_asset_options)
        # wall_pose = gymapi.Transform()
        # wall_pose.p = gymapi.Vec3(0.55, 0.01, 0.0005)

        # load block asset
        block_asset_options = gymapi.AssetOptions()
        block_asset_options.override_com = True
        block_asset_options.override_inertia = True
        block_asset_options.thickness = 0.001        
        
        
        ur5e_dof_stiffness = to_torch(self.cfg["robot"]["stiffness"], dtype=torch.float, device=self.device)
        self.ur5e_dof_damping = to_torch(self.cfg["robot"]["damping"], dtype=torch.float, device=self.device)

        self.num_ur5e_dofs = self.gym.get_asset_dof_count(ur5e_asset)

        all_block_assets = {
            "concave": [],
            "cylinder": [],
            "cube" : [],
            "half-cube": [],
            "rect": [],
            "triangle": []
        }
        for obj_name in all_block_assets:
            if ("concave" in obj_name):
                block_asset_options.vhacd_enabled = True
                block_asset_options.vhacd_params.resolution = 64000000 # 64000000
                block_asset_options.vhacd_params.alpha = 0.005
                block_asset_options.vhacd_params.beta = 0.005
            else:
                block_asset_options.vhacd_enabled = False
            block_asset = self.gym.load_asset(self.sim, block_asset_folder, f"{obj_name}.urdf", block_asset_options)
            block_props = self.gym.get_asset_rigid_shape_properties(block_asset)
            block_props[0].friction = 0.3
            self.gym.set_asset_rigid_shape_properties(block_asset, block_props)
            all_block_assets[obj_name].append(block_asset)

        # set robot dof properties
        ur5e_dof_props = self.gym.get_asset_dof_properties(ur5e_asset)
        self.ur5e_dof_lower_limits = []
        self.ur5e_dof_upper_limits = []
        for i in range(self.num_ur5e_dofs):
            ur5e_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            ur5e_dof_props['stiffness'][i] = ur5e_dof_stiffness[i]
            ur5e_dof_props['damping'][i] = self.ur5e_dof_damping[i]

            self.ur5e_dof_lower_limits.append(ur5e_dof_props['lower'][i])
            self.ur5e_dof_upper_limits.append(ur5e_dof_props['upper'][i])

        self.ur5e_dof_lower_limits = to_torch(self.ur5e_dof_lower_limits, device=self.device)
        self.ur5e_dof_upper_limits = to_torch(self.ur5e_dof_upper_limits, device=self.device)

        ur5e_start_pose = gymapi.Transform()
        ur5e_start_pose.p = gymapi.Vec3(self.cfg["robot"]["start_pose_p"][0], 
                                        self.cfg["robot"]["start_pose_p"][1], 
                                        self.cfg["robot"]["start_pose_p"][2])
        ur5e_start_pose.r = gymapi.Quat(self.cfg["robot"]["start_pose_r"][0],
                                        self.cfg["robot"]["start_pose_r"][1], 
                                        self.cfg["robot"]["start_pose_r"][2], 
                                        self.cfg["robot"]["start_pose_r"][3])

        self.ur5es = []
        self.workspaces = []
        # self.walls= []
        self.envs = []
        # self.cameras = []
        self.cam_color_tensors = []  # populated only when env.videoLog.enabled
        self.cam_depth_tensors = []
        self.cam_segm_tensors = []
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        # videoLog.cameraRes overrides for DEBUG VIDEO ONLY: non-224 res breaks
        # the grasp network's heightmap assumptions (suppress terminals then)
        _cam_res = int(self.cfg["env"].get("videoLog", {}).get("cameraRes", 224))
        camera_props.width = _cam_res
        camera_props.height = _cam_res
        # for a WIDER view at unchanged mm/px, scale fov with res (224 -> 0.02578)
        _cam_fov = float(self.cfg["env"].get("videoLog", {}).get("cameraFov", 0.0))
        if _cam_fov > 0:
            camera_props.horizontal_fov = _cam_fov
            self._cam_fov_override = _cam_fov
        else:
            camera_props.horizontal_fov = 0.02578
        camera_props.near_plane = 999.75
        camera_props.far_plane = 1001.0
        # Separate student image camera. The original clipped camera continues
        # to define privileged graspability labels and is never changed.
        self.occluded_depth_tensors, self.occluded_seg_tensors = [], []
        occluded_props = None
        if self.cfg['env'].get('graspImages', {}).get('enabled', False):
            occluded_props = gymapi.CameraProperties()
            occluded_props.enable_tensors = True
            occluded_props.width = camera_props.width
            occluded_props.height = camera_props.height
            occluded_props.horizontal_fov = camera_props.horizontal_fov
            # Keep a narrow depth range for numerical precision, but include
            # the entire robot above the table (camera is at z=999.8 m).
            occluded_props.near_plane = 997.0
            occluded_props.far_plane = camera_props.far_plane
        # OPTIONAL high-resolution VISUALISATION camera (env.videoLog.vizRes).
        # Never feeds the policy or the grasp network: those must stay at
        # cameraRes/cameraFov, since resolution without a matching fov change
        # rescales mm/px and silently corrupts Q (workspace_geometry.md #1).
        # Keeping fov FIXED while raising the resolution holds the same 0.64 m
        # coverage and just makes the pixels finer (1280 px -> 0.5 mm/px).
        self._viz_res = int(self.cfg["env"].get("videoLog", {}).get("vizRes", 0))
        self.cam_viz_tensors = []
        viz_props = None
        if self._viz_res > 0:
            viz_props = gymapi.CameraProperties()
            viz_props.enable_tensors = True
            viz_props.width = viz_props.height = self._viz_res
            viz_props.horizontal_fov = camera_props.horizontal_fov
            viz_props.near_plane = camera_props.near_plane
            viz_props.far_plane = camera_props.far_plane
        _camera_local_transform = gymapi.Transform()
        # _camera_local_transform.p = gymapi.Vec3(0.5, 0, 999.8)
        _camera_local_transform.p = gymapi.Vec3(WORKSPACE_CENTER[0], WORKSPACE_CENTER[1], 999.8)
        _camera_local_transform.r = gymapi.Quat.from_euler_zyx(np.pi, np.pi/2, 0)

        self.gym.set_light_parameters(
            self.sim, 0, gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0, 0, 0)
        )
        self.gym.set_light_parameters(
            self.sim, 1, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0.0, 0.0, 0.0), gymapi.Vec3(0, 0, 0)
        )
        self.gym.set_light_parameters(
            self.sim, 2, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0.0, 0.0, 0.0), gymapi.Vec3(0, 0, 0)
        )
        self.gym.set_light_parameters(
            self.sim, 3, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0.0, 0.0, 0.0), gymapi.Vec3(0, 0, 0)
        )

        self.gripper_pos = []
        self.gripper_rot = []
        self.gripper_idxs = []
        self.target_block_idxs = []
        self.block1_idxs = []
        self.block2_idxs = []
        self.block3_idxs = []
        self.block4_idxs = []

        self.chosen_scenes = []
        self.num_blocks_each_env = [] # (num_envs, 1), number of objects (variable) in each env
        self.all_block_idxs = torch.zeros((num_envs, num_objects), dtype=torch.long, device=self.device) # 4 being the maximum number of blocks in any env for simpler scene setups
        self.all_block_name_ids = torch.zeros((num_envs, num_objects), device=self.device) # 0 for concave, 1 for cylinder, 2 for cube, 3 for half-cube, 4 for rect, 5 for triangle

        self.block_idxs = []
        self.default_block_state = []   

        self.object_quat = torch.zeros((num_envs, num_objects, 4), dtype=torch.float32, device=self.device)        
        # graspable_envs_to_avoid = torch.load(f"{self.test_case_dir}/graspable_env_list.pt")
        # print(graspable_envs_to_avoid)
        # Scenes are bound to envs ONCE, here: env i gets scene (offset + i).
        # There is no re-draw across episodes, so a 64-env collector sees scenes
        # 0..63 and nothing else, forever. `sceneOffset` lets successive runs
        # cover disjoint slices of the pool instead of re-visiting the same head.
        _scene_offset = int(self.cfg["env"].get("test_cases", {}).get("sceneOffset", 0))
        test_case_iter = _scene_offset - 1
        # for i in range(self.num_envs):
        for i in tqdm(range(self.num_envs), desc="Creating Gym Environments"):        
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # Workspace - first actor is this
            workspace_actor = self.gym.create_actor(env=env_ptr, asset=workspace_asset, pose=workspace_pose, name="workspace", group=i, filter=1, segmentationId=0)
            self.gym.set_rigid_body_color(
                            env_ptr, workspace_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 0.0, 0.0)
                        )

            ur5e_actor = self.gym.create_actor(env=env_ptr, asset=ur5e_asset, pose=ur5e_start_pose, name="ur5e", group=i, filter=0, segmentationId=1)
            self.gym.set_actor_dof_properties(env_ptr, ur5e_actor, ur5e_dof_props)
            self.gym.set_actor_dof_states(env_ptr, ur5e_actor, self.default_joint_angles, gymapi.STATE_POS)
            self.gym.set_actor_dof_position_targets(env_ptr, ur5e_actor, np.array(self.default_joint_angles, dtype=np.float32))

            # wall_actor = self.gym.create_actor(env=env_ptr, asset=wall_asset, pose=wall_pose, name="wall", group=i, filter=0, segmentationId=0)
            # Get initial Gripper Pose
            # find_actor_rigid_body_handle
            if self.robot_urdf_option == 0:
                gripper_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_actor, "zdummy_gripper_grasp_pose_link")
            elif self.robot_urdf_option == 1:
                gripper_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_actor, "dummy_center_indictor_link")
            gripper_pose = self.gym.get_rigid_transform(env_ptr, gripper_handle)
            self.gripper_pos.append([gripper_pose.p.x, gripper_pose.p.y, gripper_pose.p.z])
            self.gripper_rot.append([gripper_pose.r.x, gripper_pose.r.y, gripper_pose.r.z, gripper_pose.r.w])
            
            # get global index of gripper in rigid body state tensor
            if self.robot_urdf_option == 0:
                self.gripper_idxs.append(self.gym.find_actor_rigid_body_index(env_ptr, ur5e_actor, "zdummy_gripper_grasp_pose_link", gymapi.DOMAIN_SIM))
            elif self.robot_urdf_option == 1:
                self.gripper_idxs.append(self.gym.find_actor_rigid_body_index(env_ptr, ur5e_actor, "dummy_center_indictor_link", gymapi.DOMAIN_SIM))
                
           
            block_files = []
            block_poses = []
            block_colors = []
            rdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_case_iter += 1
            if not os.path.exists(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt"):
                # scan forward for the next scene file; WRAP to the first file
                # when the suite has fewer scenes than envs (previously an
                # infinite busy-loop past the last file)
                if not hasattr(self, "_scene_max_idx"):
                    _names = [int(n[:-4]) for n in os.listdir(f"{rdir}/{self.test_case_dir}")
                              if n.endswith(".txt") and n[:-4].isdigit()]
                    assert _names, f"no scene files in {self.test_case_dir}"
                    self._scene_max_idx = max(_names)
                while not os.path.exists(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt"):
                    test_case_iter += 1
                    if test_case_iter > self._scene_max_idx:
                        test_case_iter = 0
            if True:
                   with open(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt", "rb") as f:
                    # with open(f"{rdir}/{self.test_case_dir}/test05.txt", "rb") as f:
                        # print(f"loaded test case: {test_case_iter:06d}.txt....")
                        scene_info = f.readlines()
                        num_blocks = len(scene_info)
                        self.num_blocks_each_env.append(num_blocks)
                        for block_idx in range(num_blocks):
                            obj_info = scene_info[block_idx].split() 
                            obj_pose = gymapi.Transform()
                            obj_pose.p = gymapi.Vec3(
                                float(obj_info[4]),
                                float(obj_info[5]),
                                float(obj_info[6]),
                            )
                            obj_pose.r = gymapi.Quat.from_euler_zyx(
                                float(obj_info[7]),
                                float(obj_info[8]),
                                float(obj_info[9]),
                            )
                            obj_color = gymapi.Vec3(
                                float(obj_info[1]),
                                float(obj_info[2]),
                                float(obj_info[3]),
                            )
                            block_files.append(obj_info[0])
                            block_poses.append(obj_pose)
                            block_colors.append(obj_color)
                        num_blocks = min(len(scene_info), self.num_objects) # limiting to 6 objects only
                        assert num_blocks == 11, f"Expected 11 objects in the test-case file, got {num_blocks} objects, {test_case_iter:06d}.txt."
                        block_assets = []
                        block_names = []
                        unique_blocks = {}
                        for block_idx in range(num_blocks):
                            block_file = block_files[block_idx]
                            if block_file in unique_blocks:
                                block_assets.append(unique_blocks[block_file])
                            else:
                                if "concave" in block_file.decode():
                                    block_asset_options.vhacd_enabled = True
                                    block_asset_options.vhacd_params.resolution = 64000000
                                    block_asset_options.vhacd_params.alpha = 0.005
                                    block_asset_options.vhacd_params.beta = 0.005
                                else:
                                    block_asset_options.vhacd_enabled = False
                                block_asset = all_block_assets[block_file.decode().split(".")[0]][0]
                                block_props = self.gym.get_asset_rigid_shape_properties(block_asset)
                                block_props[0].friction = 0.3
                                self.gym.set_asset_rigid_shape_properties(block_asset, block_props)
                                unique_blocks[block_file] = block_asset
                                block_assets.append(block_asset)
                            block_names.append(block_file.decode().split(".")[0])

                        assert num_blocks > 0, "No Objects in the test-case!"
                        assert num_blocks == self.num_objects, f"Expected {self.num_objects} objects, got {num_blocks} objects, {test_case_file}."
                        for block_idx in range(num_blocks):

                            if block_idx == 0:
                                segm_id = 255
                            else:
                                segm_id = 50 + 10 * (block_idx - 1)
                            
                            block_actor = self.gym.create_actor(env=env_ptr, asset=block_assets[block_idx], pose=block_poses[block_idx], name=f"block{segm_id}", group=i, filter=0, segmentationId=segm_id) 
                            self.gym.set_rigid_body_color(
                                        env_ptr, block_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, block_colors[block_idx]
                                    )
                            self.default_block_state.append(
                                [block_poses[block_idx].p.x, block_poses[block_idx].p.y, block_poses[block_idx].p.z, block_poses[block_idx].r.x, block_poses[block_idx].r.y, block_poses[block_idx].r.z, block_poses[block_idx].r.w, 0, 0, 0, 0, 0, 0]
                            )

                            self.all_block_idxs[i, block_idx] = self.gym.get_actor_rigid_body_index(env_ptr, block_actor, 0, gymapi.DOMAIN_SIM)
                            self.all_block_name_ids[i, block_idx] = self.name2id_map[block_names[block_idx]]
                            self.object_quat[i, block_idx] = euler_to_quat(roll=torch.tensor(float(obj_info[9])), pitch=torch.tensor(float(obj_info[8])), yaw=torch.tensor(float(obj_info[7])))
                        
                    # break

            self.envs.append(env_ptr)
            self.ur5es.append(ur5e_actor)
            self.workspaces.append(workspace_actor)
            # self.walls.append(wall_actor)
        
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_transform(
                camera_handle, env_ptr, _camera_local_transform
            )
            # self.gym.set_camera_location(
            #     camera_handle, env, _camera_local_transform.p, gymapi.Vec3(0, 0, -1.0)
            # )
            # self.cameras.append(camera_handle)
            # color tensors only when video logging is on (adds ~200KB/env GPU)
            if self.cfg["env"].get("videoLog", {}).get("enabled", False):
                cam_color_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
                    )
                self.cam_color_tensors.append(
                    gymtorch.wrap_tensor(cam_color_tensor)[:, :, :3])
            cam_depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH
                )
            torch_cam_depth_tensor = gymtorch.wrap_tensor(cam_depth_tensor)
            
            cam_segm_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION
                )
            torch_cam_segm_tensor = gymtorch.wrap_tensor(cam_segm_tensor)

            # self.cam_color_tensors.append(torch_cam_color_tensor[:, :, :3])
            self.cam_depth_tensors.append(torch_cam_depth_tensor)
            self.cam_segm_tensors.append(torch_cam_segm_tensor)
            if occluded_props is not None:
                occluded_handle = self.gym.create_camera_sensor(env_ptr, occluded_props)
                self.gym.set_camera_transform(occluded_handle, env_ptr, _camera_local_transform)
                self.occluded_depth_tensors.append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, occluded_handle, gymapi.IMAGE_DEPTH)))
                self.occluded_seg_tensors.append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, occluded_handle, gymapi.IMAGE_SEGMENTATION)))

            if viz_props is not None:
                viz_handle = self.gym.create_camera_sensor(env_ptr, viz_props)
                self.gym.set_camera_transform(viz_handle, env_ptr, _camera_local_transform)
                self.cam_viz_tensors.append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_ptr, viz_handle, gymapi.IMAGE_COLOR))[:, :, :3])
        

        self.gripper_idxs = to_torch(self.gripper_idxs, dtype=torch.int32, device=self.device)
        self.default_block_state = to_torch(self.default_block_state, device=self.device, dtype=torch.float).view(
            self.num_envs, self.num_objects, 13
        )
        dists = []
        for ib in range(1, self.num_objects):      
            dist = torch.norm(self.default_block_state[:, ib, :3] - self.default_block_state[:, 0, :3], dim=1)  # (num_envs, 2)
            dists.append(dist.unsqueeze(1))  # (num_envs, 1)

        # Find the two closest distances for each environment
        dist = torch.stack(dists, dim=1)  # (num_envs, 5)
        # print("dist shape:", dist.shape)
        # sort distances and take the two smallest ones
        # print("sorted dists", torch.sort(dist, dim=1))
        smallest_four_distances = torch.sort(dist, dim=1).values[:, :4]  # (num_envs, 5)
        # print("self smallest two indices shape:", self.smallest_two_indices.shape)
        tight_coupling_dist = torch.sum(smallest_four_distances, dim=1)  # (num_envs, )
        self.default_min_tight_coupling_dist_for_reset = smallest_four_distances[:, 0].reshape(-1)  # (num_envs, ) — reshape not squeeze: num_envs=1 must stay 1-D (open-loop twin)
        self.prev_min_tight_coupling_dist = self.default_min_tight_coupling_dist_for_reset.clone()
        self.default_tight_coupling_dist = tight_coupling_dist.reshape(-1).clone()
        self.prev_tight_coupling_dist = self.default_tight_coupling_dist.clone()


        self.num_actors = 1 + 1 + 0 + num_objects # 1 workspace + 1 robot + 0 wall + 6 target objects
        link_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.ur5es[0])

        finger_names = [name for name in link_names if "pad" in name]
        self.gripper_handles = [
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5es[0], name) for name in finger_names
        ]
        self.init_data()

    def init_data(self):
        if self.robot_urdf_option == 0:
            gripper_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5es[0], "zdummy_gripper_grasp_pose_link")
        elif self.robot_urdf_option == 1:
            gripper_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5es[0], "dummy_center_indicator_link")
        gripper_pose = self.gym.get_rigid_transform(self.envs[0], gripper_handle)
        self.gripper_pos = to_torch([gripper_pose.p.x, gripper_pose.p.y, gripper_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.default_gripper_pos = self.gripper_pos[:, :2].clone()
        self.gripper_rot = to_torch([gripper_pose.r.x, gripper_pose.r.y, gripper_pose.r.z, gripper_pose.r.w], device=self.device).repeat((self.num_envs, 1))

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.prev_gripper_pos[:], self.successes[:], self.prev_min_tight_coupling_dist[:], self.prev_action_idx[:] = compute_more_reward_jit(
            reset_buf = self.reset_buf, progress_buf = self.progress_buf,
            blocks_rect_rotated = self.blocks_rect_rotated,
            gripper_pos=self.gripper_pos,
            num_envs = self.num_envs, 
            max_episode_length = self.max_episode_length, 
            prev_min_tight_coupling_dist = self.prev_min_tight_coupling_dist,
            distance_scale = self.distance_scale, target_clearance_reward = self.target_clearance_reward,
            desired_eef_to_target_distance = self.desired_eef_to_target_distance,
            tight_coupling_scale = self.tight_coupling_scale, 
            tight_coupling_bias=self.cfg["env"]["tightCouplingBias"], prev_gripper_pos = self.prev_gripper_pos,
            gripper_idle_scale= self.cfg["env"]["gripperIdleScale"],  
            min_tight_coupling_tolerance=self.cfg["env"]["minCouplingDifference"],
            successes = self.successes, grasp_q_values_parallel = self.grasp_q_parallel_values,
            inverse_action=self.inverse_action,
            prev_actions=self.prev_action_idx,
            actions=self.actions
        )
        self.total_resets += self.reset_buf.sum().item()
        direct_average_successes = self.total_successes + self.successes.sum()
        self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
        self.extras["successes"] = self.successes.mean()
        self.extras["direct_average_successes"] = direct_average_successes / (self.total_resets + self.num_envs)
        # The direct average shows the overall result more quickly, but slightly undershoots long term
        # policy performance.
        # print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
        


    def refresh_env_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def _refresh_task_tensors(self):
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5e") # shape [num_envs, 15, 6, 8]
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = jacobian[:, self.ur5e_gripper_index-1, :, :6]
        
        # get rigid body state tensor for the simulation
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor for the simulation
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        if self.robot_urdf_option == 0:
            self.ur5e_dof_pos = dof_states[:, 0].view(self.num_envs, 8)
        elif self.robot_urdf_option == 1:    
            self.ur5e_dof_pos = dof_states[:, 0].view(self.num_envs, 12)
        self.ur5e_dof_targets = torch.zeros_like(self.ur5e_dof_pos, device=self.device)

        for bidx in range(self.num_objects):
            block_pos = self.rb_states[self.all_block_idxs[:, bidx], :3].clone()
            block_rot = self.rb_states[self.all_block_idxs[:, bidx], 3:7].clone()
            # permute the order to 2, 1, 0, 3
            block_rot = block_rot[:, [2, 1, 0, 3]]  # (num_envs, 4) - to match the order of euler_to_quat

            block_coord = torch.zeros((self.num_envs, 4, 2), device=self.device)
            block_width = torch.zeros((self.num_envs, self.num_objects), device=self.device)
            
            block_width = torch.where(self.all_block_name_ids == 0, self.name_id2_dims_map[0][0], block_width)
            block_width = torch.where(self.all_block_name_ids == 1, self.name_id2_dims_map[1][0], block_width)
            block_width = torch.where(self.all_block_name_ids == 2, self.name_id2_dims_map[2][0], block_width)
            block_width = torch.where(self.all_block_name_ids == 3, self.name_id2_dims_map[3][0], block_width)
            block_width = torch.where(self.all_block_name_ids == 4, self.name_id2_dims_map[4][0], block_width)
            block_width = torch.where(self.all_block_name_ids == 5, self.name_id2_dims_map[5][0], block_width)
            
            block_height = torch.zeros((self.num_envs, self.num_objects), device=self.device)
            block_height = torch.where(self.all_block_name_ids == 0, self.name_id2_dims_map[0][1], block_height)
            block_height = torch.where(self.all_block_name_ids == 1, self.name_id2_dims_map[1][1], block_height)
            block_height = torch.where(self.all_block_name_ids == 2, self.name_id2_dims_map[2][1], block_height)
            block_height = torch.where(self.all_block_name_ids == 3, self.name_id2_dims_map[3][1], block_height)
            block_height = torch.where(self.all_block_name_ids == 4, self.name_id2_dims_map[4][1], block_height)
            block_height = torch.where(self.all_block_name_ids == 5, self.name_id2_dims_map[5][1], block_height)
            
            block_width[:, 0] = self.gripper_open_width # Creating the gripper clearance rectangle
            block_height[:, 0] = self.gripper_open_length # creating the gripper clearance rectangle
            
            block_coord[:, 0, 0] = block_pos[:, 0] - (block_width[:, bidx]/2)
            block_coord[:, 0, 1] = block_pos[:, 1] - (block_height[:, bidx]/2)

            block_coord[:, 1, 0] = block_pos[:, 0] + (block_width[:, bidx]/2)
            block_coord[:, 1, 1] = block_pos[:, 1] - (block_height[:, bidx]/2)


            block_coord[:, 2, 0] = block_pos[:, 0] + (block_width[:, bidx]/2)
            block_coord[:, 2, 1] = block_pos[:, 1] + (block_height[:, bidx]/2)

            block_coord[:, 3, 0] = block_pos[:, 0] - (block_width[:, bidx]/2)
            block_coord[:, 3, 1] = block_pos[:, 1] + (block_height[:, bidx]/2)

            block_rect = torch.cat([block_pos[:, :2].unsqueeze(1), block_coord], dim=1)
            
            # print("block rot chosen env: ", block_rot[2])
            # print("block_rect shape:", block_rect.shape, "block_rot shape:", block_rot.shape)
            self.blocks_rect_rotated[bidx] = rotate_rectangles(block_rect, block_rot, axis=0)

        # self.gripper_pos = self.rb_states[self.gripper_idxs, :3]  # Get positions of gripper
        # self.gripper_rot = self.rb_states[self.gripper_idxs, 3:7]  # Get orientations of gripper
        self.gripper_pos = self.rb_states[self.gripper_idxs.to(dtype=torch.long), :3]  # Get positions of gripper
        self.gripper_rot = self.rb_states[self.gripper_idxs.to(dtype=torch.long), 3:7]  # Get orientations of gripper


    def compute_observations(self):
        obs_tensors = []
        for bidx in range(self.num_objects):
            obs_tensors.append(self.blocks_rect_rotated[bidx, :, 1:5, :].view(self.num_envs, -1))  # (num_envs, 4, 2) to (num_envs, 8)
        obs_tensors.append(self.gripper_pos[:, :2]) # (num_envs, 2)
        self.obs_buf = torch.cat(obs_tensors, dim=-1) # shape: (num_envs, numObservations)
        
        assert self.obs_buf.shape[0] == self.num_envs, self.obs_buf.shape[1] == self.num_objects*8 + 2 
        return self.obs_buf

    def reset_idx(self, env_ids, from_where="init"):
        env_ids_int32 = env_ids.to(dtype=torch.long)
        robot_ids = self.global_robot_ids[env_ids_int32]

        assert self.ur5e_dof_targets.shape[1] == self.ur5e_default_dof_pos.shape[0], f"shape of dof target: {self.ur5e_dof_targets.shape}, default_dof_shape: {self.ur5e_default_dof_pos.shape}"

        self.ur5e_dof_targets[env_ids_int32, :] = self.ur5e_default_dof_pos
        self.ur5e_dof_pos[env_ids_int32, :] = self.ur5e_default_dof_pos
        self.ur5e_dof_vel[env_ids_int32, :] = torch.zeros_like(self.ur5e_dof_vel[env_ids_int32])
        block_indices = self.global_actor_ids[env_ids_int32, 2:].flatten()
        self.block_state[env_ids_int32, :] = self.default_block_state[env_ids_int32, :]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.actor_root_state_tensor, gymtorch.unwrap_tensor(block_indices), len(block_indices),
        )

        self.gym.set_dof_position_target_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.ur5e_dof_targets),
                    gymtorch.unwrap_tensor(robot_ids),
                    len(robot_ids),
                )
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(robot_ids), len(robot_ids))

        self.progress_buf[env_ids_int32] = 0
        self.grasp_q_parallel_values[env_ids_int32] = 0.0
        self.reset_buf[env_ids_int32] = 0
        self.rew_buf[env_ids_int32] = 0.0
        self.successes[env_ids_int32] = 0
        self.prev_min_tight_coupling_dist[env_ids_int32] = self.default_min_tight_coupling_dist_for_reset[env_ids_int32]
        self.prev_gripper_pos[env_ids_int32] = self.default_gripper_pos[env_ids_int32].clone()
        self.plan_active[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.bool)
        self.plan_phase[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.long)  # 0 or 1
        self.plan_tick[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.long)  # 0..(K-1)

        self.start_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.wp1_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.wp2_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.prev_action_idx[env_ids_int32] = -1 * torch.ones(len(env_ids_int32), device=self.device, dtype=torch.long)
    def control_ik(self, dpose):
        # global damping, j_eef, num_envs 
        # Solve damped least squares for inverse kinematics
        j_eef_T = torch.transpose(self.j_eef, 1, 2)  # Transpose Jacobian (a matrix )
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u
    
    def _apply_smooth_target(self):
        # choose current waypoint
        wp = torch.where(self.phase.view(-1,1) == 0, self.wp1_pos, self.wp2_pos)  # (B,3)

        # interpolation fraction within this control step
        # if control_freq_inv=4, you get 4 substeps, but here we’re setting target once per RL step
        # Better: move over multiple RL steps OR update in the inner simulate loop.
        # Since VecTask simulates control_freq_inv inside step(), we can update per substep by using self.substep.

        t = (self.substep.float() + 1.0) / float(self.control_freq_inv)  # (B,)
        t = t.view(-1,1)

        # start of this phase
        start = torch.where(self.phase.view(-1,1) == 0, self.start_pos, self.wp1_pos)
        target_pos = start * (1 - t) + wp * t

        # position error -> IK
        pos_err = target_pos - self.gripper_pos
        cc = quat_conjugate(self.gripper_rot)
        q_r = quat_mul(self.gripper_rot, cc)
        orn_err = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)  # (this is always ~0; see note below)
        dpose = torch.cat((pos_err, orn_err), dim=-1).unsqueeze(-1)

        self.ur5e_dof_targets[:, :6] = self.ur5e_dof_pos[:, :6] + self.control_ik(dpose)
        self.ur5e_dof_targets[:, 6:] = 0.0  # hold pads CLOSED (real 2F-85); was ratcheting to current pos

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets))

        # advance substep counters
        self.substep += 1
        done_substep = self.substep >= self.control_freq_inv
        if torch.any(done_substep):
            self.substep[done_substep] = 0
            self.phase[done_substep] += 1

            # finish plan after phase 2
            finished = self.phase >= 2
            self.plan_active[finished] = False
            self.phase[finished] = 0

    # def build_two_step_plan(self, action_idx, step=0.01):
    #     """
    #     action_idx: (B,) long in [0,15]
    #     returns:
    #     d1: (B,2) first (dx,dy)
    #     d2: (B,2) second (dx,dy)
    #     """
    #     B = action_idx.shape[0]
    #     device = action_idx.device
    #     d1 = torch.zeros((B,2), device=device)
    #     d2 = torch.zeros((B,2), device=device)

    #     # cardinals: 0..3 => N,E,S,W
    #     # N: (0,+s), E: (+s,0), S:(0,-s), W:(-s,0)
    #     is_card = action_idx < 4
    #     card = action_idx

    #     d = torch.tensor([[0, step], [step, 0], [0, -step], [-step, 0]], device=device, dtype=torch.float32)
    #     d1[is_card] = d[card[is_card]]
    #     d2[is_card] = d[card[is_card]]

    #     # diagonals: 4..15
    #     is_diag = ~is_card
    #     a = action_idx[is_diag] - 4  # 0..11

    #     diag_dir = a // 3            # 0:NE,1:NW,2:SE,3:SW
    #     mode     = a % 3             # 0:diag, 1:grid1, 2:grid2

    #     # define unit moves
    #     N = torch.tensor([0, step], device=device)
    #     E = torch.tensor([step, 0], device=device)
    #     S = torch.tensor([0, -step], device=device)
    #     W = torch.tensor([-step, 0], device=device)

    #     # diagonal step length: to make total displacement of one step = 1cm,
    #     # use (step/sqrt(2), step/sqrt(2)) etc.
    #     ds = step / (2**0.5)
    #     diag = torch.tensor([
    #         [ ds,  ds],   # NE
    #         [-ds,  ds],   # NW
    #         [ ds, -ds],   # SE
    #         [-ds, -ds],   # SW
    #     ], device=device)

    #     # grid orders for each diagonal:
    #     # NE: (E then N) or (N then E)
    #     # NW: (W then N) or (N then W)
    #     # SE: (E then S) or (S then E)
    #     # SW: (W then S) or (S then W)
    #     first_grid = torch.stack([E, W, E, W], dim=0)  # (4,2)
    #     second_grid= torch.stack([N, N, S, S], dim=0)  # (4,2)
    #     # swapped:
    #     first_grid_sw = second_grid
    #     second_grid_sw= first_grid

    #     # fill diagonal cases
    #     idx = torch.nonzero(is_diag, as_tuple=False).squeeze(1)

    #     # mode 0: diag,diag
    #     m0 = (mode == 0)
    #     d1[idx[m0]] = diag[diag_dir[m0]]
    #     d2[idx[m0]] = diag[diag_dir[m0]]

    #     # mode 1: grid then grid (E/W then N/S)
    #     m1 = (mode == 1)
    #     d1[idx[m1]] = first_grid[diag_dir[m1]]
    #     d2[idx[m1]] = second_grid[diag_dir[m1]]

    #     # mode 2: swapped grid (N/S then E/W)
    #     m2 = (mode == 2)
    #     d1[idx[m2]] = first_grid_sw[diag_dir[m2]]
    #     d2[idx[m2]] = second_grid_sw[diag_dir[m2]]

    #     return d1, d2

    def build_two_step_plan(self, action_idx, total=0.01):
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


    def _apply_plan_target_one_substep(self):
        if not torch.any(self.plan_active):
            return

        # choose phase start and end
        phase0 = (self.plan_phase == 0)
        start = torch.where(phase0.unsqueeze(1), self.start_pos, self.wp1_pos)  # (B,3)
        end   = torch.where(phase0.unsqueeze(1), self.wp1_pos,  self.wp2_pos)  # (B,3)

        # interpolation fraction for this tick
        t = (self.plan_tick.float() + 1.0) / float(self.K)
        t = t.unsqueeze(1)  # (B,1)

        target_pos = start*(1-t) + end*t

        # clamp max cartesian increment (prevents “throwing”)
        if self.ctrl_mode in ("absolute", "reach_gated"):
            # aim at the phase's END waypoint (a setpoint the PD converges to),
            # not an interpolated point one frame ahead
            target_pos = end
        pos_err = target_pos - self.gripper_pos
        max_norm = (self.ctrl_max_step if self.ctrl_mode in ("absolute", "reach_gated")
                    else self.ctrl_max_step / float(self.K))
        n = torch.norm(pos_err, dim=1, keepdim=True) + 1e-8
        pos_err = pos_err * torch.clamp(max_norm / n, max=1.0)

        # orientation: hold the tool perpendicular when ctrl_hold_orn, else the
        # legacy zero error (IK leaves orientation free in the null space, which
        # is the source of the observed EEF tilt drift)
        if self.ctrl_hold_orn:
            if self.ctrl_orn_ref is None:
                self.ctrl_orn_ref = self.gripper_rot.clone()
            q_err = quat_mul(self.ctrl_orn_ref, quat_conjugate(self.gripper_rot))
            orn_err = q_err[:, 0:3] * torch.sign(q_err[:, 3]).unsqueeze(-1)
        else:
            orn_err = torch.zeros((self.num_envs, 3), device=self.device)

        dpose = torch.cat([pos_err, orn_err], dim=1).unsqueeze(-1)
        dq = self.control_ik(dpose)
        if self.ctrl_mode in ("absolute", "reach_gated"):
            # Hold ONE setpoint for the whole waypoint phase instead of
            # re-deriving an increment from the current pose every tick.
            # Incremental targets give an exponential approach that never
            # converges: measured delivery of a commanded 20mm primitive was
            # only 14-27% at controlFrequencyInv 1-4 (job 248002).
            fresh = self.ctrl_fresh_step | (self.plan_phase != self.ctrl_last_phase)
            if torch.any(fresh):
                self.ctrl_hold_target[fresh] = (
                    self.ur5e_dof_pos[fresh, :6] + dq[fresh])
                self.ctrl_last_phase[fresh] = self.plan_phase[fresh]
                self.ctrl_fresh_step[fresh] = False
            self.ur5e_dof_targets[:, :6] = self.ctrl_hold_target
        else:
            self.ur5e_dof_targets[:, :6] = self.ur5e_dof_pos[:, :6] + dq
        self.ur5e_dof_targets[:, 6:] = 0.0  # hold pads CLOSED (real 2F-85); was ratcheting to current pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets))

        # advance tick/phase
        if self.ctrl_mode == "reach_gated":
            # advance only when the joints reached the held setpoint, or stalled
            err = (self.ctrl_hold_target - self.ur5e_dof_pos[:, :6]).abs().amax(dim=1)
            moved = (self.ur5e_dof_pos[:, :6] - self.ctrl_prev_dof).abs().amax(dim=1)
            self.ctrl_stall_count = torch.where(moved < self.ctrl_stall_eps,
                                                self.ctrl_stall_count + 1,
                                                torch.zeros_like(self.ctrl_stall_count))
            self.ctrl_prev_dof = self.ur5e_dof_pos[:, :6].clone()
            arrived = (err < self.ctrl_reach_tol) | (self.ctrl_stall_count > self.ctrl_stall_frames)
            self.plan_tick = torch.where(self.plan_active & arrived,
                                         self.plan_tick + 1, self.plan_tick)
            self.ctrl_stall_count = torch.where(arrived,
                                                torch.zeros_like(self.ctrl_stall_count),
                                                self.ctrl_stall_count)
        else:
            self.plan_tick += 1
        done_tick = self.plan_tick >= self.K
        if torch.any(done_tick):
            self.plan_tick[done_tick] = 0
            self.plan_phase[done_tick] += 1

            finished = self.plan_phase >= 2
            # NOTE: gripper_pos is stale inside the substep loop (refreshed only
            # in post_physics_step), so delivery is computed there instead.
            self.plan_done_flag |= finished
            self.plan_active[finished] = False
            self.plan_phase[finished] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # unique actions across all envs
        # actions_unique = torch.unique(self.actions)
        # if self.progress_buf[0] % 50 == 0:  # only print for every 50 steps
        #     print("actions shape:", self.actions.shape)
        #     print("unique actions in batch:", actions_unique)
        if self.actions.dim() == 2:
            best = self.actions.squeeze(-1).long() 
        else:
            best = self.actions.long() 
        # start a new 2-step plan only when not active 
        # a plan spans 2 waypoint phases; with controlFrequencyInv < 2 it cannot
        # finish inside one RL step, so the NEXT action is ignored while it
        # completes (only envs with plan_active == False accept a new action).
        self.ctrl_fresh_step[:] = True
        start_new = ~self.plan_active 

        if torch.any(start_new):
            best_new = best[start_new]
            d1, d2 = self.build_two_step_plan(best_new, total=self.push_total)
            # snapshot start pose once. This was commented out: start_pos stayed
            # at zeros, so (a) per-plan delivery measured distance from the
            # world origin (reported 0.0% across ~6000 plans) and (b) the
            # incremental path interpolated phase 0 from the origin for K >= 2
            # (harmless at K = 1, where t = 1 lands on `end`, which is why it
            # went unnoticed). The absolute/reach_gated modes target `end`
            # directly and were never affected.
            # build a 2-step displacement 
            # step = 0.01 / 2 # 0.5cm per sub-step
            # d1, d2 = self.build_two_step_plan(best, total=0.01)

            # dx = torch.zeros((self.num_envs, 3), device=self.device)
            # dy = torch.zeros((self.num_envs, 3), device=self.device)

            # waypoint 1 and 2
            wp1 = self.gripper_pos[start_new].clone()
            wp1[:, 0] += d1[:, 0]
            wp1[:, 1] += d1[:, 1]

            wp2 = wp1.clone()
            wp2[:, 0] += d2[:, 0]
            wp2[:, 1] += d2[:, 1]

            # clamp waypoints to the ACTUAL workspace box. The old hardcoded
            # bounds (x 0.30-0.75, y -0.224-0.284) predate the 0.448 m square
            # and are looser than it on three sides, which let the EEF walk
            # outside: measured 23-31% of episodes lost the gripper with 0%
            # recovery, since nothing penalises or terminates an EEF excursion
            # (the OOW rule only tests blocks). MoreRobust supplies ws_x/ws_y
            # so this tracks robust.wsSide; the fallback is the module default.
            _wx = getattr(self, "ws_x", (WORKSPACE_LIMITS[0][0], WORKSPACE_LIMITS[0][1]))
            _wy = getattr(self, "ws_y", (WORKSPACE_LIMITS[1][0], WORKSPACE_LIMITS[1][1]))
            wp1[:, 0] = torch.clamp(wp1[:, 0], float(_wx[0]), float(_wx[1]))
            wp1[:, 1] = torch.clamp(wp1[:, 1], float(_wy[0]), float(_wy[1]))
            wp2[:, 0] = torch.clamp(wp2[:, 0], float(_wx[0]), float(_wx[1]))
            wp2[:, 1] = torch.clamp(wp2[:, 1], float(_wy[0]), float(_wy[1]))

            self.start_pos[start_new] = self.gripper_pos[start_new].clone()
            self.wp1_pos[start_new] = wp1
            self.wp2_pos[start_new] = wp2

            self.plan_phase[start_new] = 0
            self.plan_tick[start_new] = 0
            self.plan_active[start_new] = True

        # Now: set targets for *this* control step (one substep target, not simulate here!)
        # self._apply_plan_target_one_substep()

        # direction = self.actions[:, :self.num_actions]
        # assert direction.shape == (self.num_envs, self.num_actions), f"Direction shape mismatch: {direction.shape}, expected {(self.num_envs, 4)}, got {direction.shape}"
        # best_action_idx = torch.argmax(direction, dim=1)  # (num_envs,)
        # actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        
        # pos_x =  self.gripper_pos.clone()
        # pos_x[:, 0] += 0.01
        # # (0.01 * 0.25)

        # pos_y = self.gripper_pos.clone()
        # pos_y[:, 1] += 0.01
        # # (0.01 * 0.25)

        # neg_x = self.gripper_pos.clone()
        # neg_x[:, 0] -= 0.01
        # # (0.01 * 0.25)

        # neg_y = self.gripper_pos.clone()
        # neg_y[:, 1] -= 0.01
        # # (0.01 * 0.25)

        # gripper_target_pos = self.gripper_pos.clone()  # (num_envs, 3)
        # gripper_target_pos = torch.where((best_action_idx == 0).unsqueeze(1), pos_x, gripper_target_pos)
        # gripper_target_pos = torch.where((best_action_idx == 1).unsqueeze(1), pos_y, gripper_target_pos)
        # gripper_target_pos = torch.where((best_action_idx == 2).unsqueeze(1), neg_x, gripper_target_pos)
        # gripper_target_pos = torch.where((best_action_idx == 3).unsqueeze(1), neg_y, gripper_target_pos)

        # target_gripper_rotation_roll = self.actions # (num_envs,) - planar rotation (yaw - rot around Z-axis)
       

        # gripper_target_pos[:, 0] = torch.clamp(gripper_target_pos[:, 0], min=0.30, max=0.75)
        # gripper_target_pos[:, 1] = torch.clamp(gripper_target_pos[:, 1], min=-0.224, max=0.284)

        # current_gripper_euler_rot = quaternion_to_euler(self.gripper_rot)
        # target_gripper_euler_rot = current_gripper_euler_rot.clone()
        # target_gripper_euler_rot[:, 2] += target_gripper_rotation_roll[:, 0]  # Modify roll (rx), keep pitch (ry) & yaw (rz) the same
        
        # pos_err = gripper_target_pos - self.gripper_pos
        # cc = quat_conjugate(self.gripper_rot)
        # q_r = quat_mul(self.gripper_rot, cc)
        # orn_err = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
        # dpose = torch.cat((pos_err, orn_err), dim=-1).unsqueeze(-1)  # Concatenate position and orientation error
        # self.ur5e_dof_targets[:, :6] = self.ur5e_dof_pos[:, :6] + self.control_ik(dpose)  # Compute joint positions using IK
        # self.ur5e_dof_targets[:, 6:] = 0.0  # hold pads CLOSED (real 2F-85); was ratcheting to current pos  # Keep last 2 joints (gripper finger tips) unchanged

        # # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets))

        # step physics and render each frame
        # for i in range(self.control_freq_inv):
        #     if self.force_render:
        #         self.render()
        #     self.gym.simulate(self.sim)


    def post_physics_step(self):
        self.progress_buf += 1
        # plans that completed during the substep loop: measure their realised
        # displacement now that the rigid-body state has been refreshed
        if torch.any(self.plan_done_flag):
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            f = self.plan_done_flag
            self.plan_delivery_last[f] = (
                self.gripper_pos[f, :2] - self.start_pos[f, :2]).norm(dim=-1)
            self.plan_delivery_seq[f] += 1
            self.plan_done_flag[:] = False

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # print("reset buf:", self.reset_buf)
        # print("env ids to reset:", env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids, from_where="post_physics_step")

        self.refresh_env_tensors()
        self._refresh_task_tensors()

        self.compute_observations()

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            # cam_color = torch.stack(self.cam_color_tensors, dim=0)   # (B,H,W,3) uint8?
            cam_depth = torch.stack(self.cam_depth_tensors, dim=0)   # (B,H,W)   float?
            cam_segm  = torch.stack(self.cam_segm_tensors,  dim=0)  # (B,H,W)   int?

            # self._cam_color = cam_color
            # self._cam_depth = cam_depth
            # self._cam_segm  = cam_segm

            # 1) Replace -inf with 0
            is_neginf = torch.isneginf(cam_depth)
            depth_clean = torch.where(is_neginf, torch.zeros_like(cam_depth), cam_depth)

            # 2) Subtract per-image minimum so min == 0.
            # The min is taken over the CENTRAL 224px (0.448m) sub-window so the
            # reference plane matches the GN's training-time statistics on any
            # canvas size: a wider canvas sees farther table pixels whose depth
            # is ~1mm lower, and the GN flips 1.0<->0.0 on a 1mm global offset
            # (measured; see tools/gn_ab_test.py).
            H = depth_clean.shape[1]
            c0 = max((H - 224) // 2, 0)
            central = depth_clean[:, c0:c0 + 224, c0:c0 + 224]
            depth_min = central.amin(dim=(1, 2), keepdim=True)      # (B,1,1)
            depth_shifted = depth_clean - depth_min
            
            # q_start = time.time()
            # q_values_parallel = self.mcts_helper.get_grasp_q_parallel_Bx16(
            #     cam_color, cam_depth_cropped, cam_segm
            # )
            # q_values_parallel = self.mcts_helper.get_grasp_q_parallel_Bx16(
            #     cam_color, cam_depth_cropped, cam_segm
            # )
            # cam_color,
            q_values_parallel = self.mcts_helper.grasp_prob_B(depth_shifted, cam_segm, tile_length=112)
            self.grasp_q_parallel_values = q_values_parallel
            # if self.progress_buf[0] % 50 == 0:
            #     print(f"[POST PHYSICS per 50 steps] Max grasp Q value: {q_values_parallel.max()}, Min grasp Q value: {q_values_parallel.min()}, Mean grasp Q value: {q_values_parallel.mean()}")
            #     # print the environment id with the highest q value
            #     max_q_value, max_env_id = torch.max(q_values_parallel, dim=0)
            #     min_q_value, min_env_id = torch.min(q_values_parallel, dim=0)
            #     print(f"Env id with highest Q value: {max_env_id.item()}, Q value: {max_q_value.item()}, Env id with min Q value: {min_env_id.item()}")
        finally:
            self.gym.end_access_image_tensors(self.sim)
        

        self.compute_reward()


@torch.jit.script
def compute_more_reward_jit(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, 
    blocks_rect_rotated: torch.Tensor,  
    gripper_pos: torch.Tensor,
    num_envs:int, max_episode_length: int, 
    prev_min_tight_coupling_dist: torch.Tensor,
    distance_scale: float, target_clearance_reward: float,
    desired_eef_to_target_distance: float, 
    tight_coupling_scale: float,  tight_coupling_bias: float,
    prev_gripper_pos: torch.Tensor, gripper_idle_scale: float, 
    min_tight_coupling_tolerance: float, 
    successes: torch.Tensor, grasp_q_values_parallel: torch.Tensor,
    inverse_action: torch.Tensor,
    prev_actions: torch.Tensor,
    actions: torch.Tensor
):    
    # num_grasp_rects = 8
    # overlap_penalty = torch.ones(size=(int(num_envs), num_grasp_rects), dtype=torch.float32, device=blocks_rect_rotated.device)
    is_graspable = torch.zeros(size=(int(num_envs), ), dtype=torch.bool, device=blocks_rect_rotated.device)
    gripper_to_target_center_dist = torch.norm(gripper_pos[:, :2] - blocks_rect_rotated[0, :, 0, :], dim=1)
    
    dist_penalty = gripper_to_target_center_dist.clone() 
    # print("gripper to target center dist[115]:", gripper_to_target_center_dist[115])
    # Smooth distance reward (max at 7 cm, penalize too close/far)
    too_close_threshold = 0.03
    # dist_penalty = distance_scale * (gripper_to_target_center_dist - desired_eef_to_target_distance)**2
    dist_penalty = torch.where(
                                gripper_to_target_center_dist + 0.001 > desired_eef_to_target_distance, 
                                (distance_scale * (0.001 + gripper_to_target_center_dist - desired_eef_to_target_distance)),  
                                # + non_zero_dist_penalty_bias, 
                                torch.zeros_like(dist_penalty)
                                )
    dist_penalty = torch.where(
        gripper_to_target_center_dist < too_close_threshold,
        dist_penalty + -2.0 * (too_close_threshold - gripper_to_target_center_dist),  # extra negative reward
        dist_penalty
    )   
    # print("dist penalty[115]:", dist_penalty[115].item())
    # ov_penalty = overlap_penalty.min(dim=1).values

    # epsilon = 0.01
    # is_graspable = (ov_penalty < epsilon)
    is_graspable = grasp_q_values_parallel > 0.9

    grasp_reward = torch.zeros(size=(int(num_envs), ), dtype=torch.float32, device=blocks_rect_rotated.device)
    grasp_reward = torch.where(is_graspable, target_clearance_reward, grasp_reward)
    overlap_penalty = torch.zeros_like(grasp_reward, device=grasp_reward.device)
    # print("grasp q values parallel shape:", grasp_q_values_parallel.shape)
    overlap_penalty = torch.where(
        (grasp_q_values_parallel > 0.0) & (grasp_q_values_parallel <= 0.9),
        (grasp_q_values_parallel - 0.9) / 0.9,
        torch.zeros_like(grasp_q_values_parallel)
    )
    overlap_penalty = torch.where(
        grasp_q_values_parallel < -1.0,
        torch.ones_like(overlap_penalty) * -2.0,
        overlap_penalty
    )

    overlap_penalty = torch.where(is_graspable, torch.zeros_like(overlap_penalty), overlap_penalty)


    # overlap_penalty = torch.where((grasp_q_values_parallel > 0.0) & (grasp_q_values_parallel <= 0.9), (grasp_q_values_parallel-0.9)/0.9, overlap_penalty)
    # overlap_penalty *= overlap_scale
    # overlap_penalty_min3 = torch.sort(overlap_penalty, dim=1).values[:, -3:]  # (num_envs, 3)
    # mean_overlap_penalty = torch.mean(overlap_penalty_min3, dim=1)
    # mean_overlap_penalty = torch.where(
    #     is_graspable, 
    #     0.0, 
    #     mean_overlap_penalty)
    
    
    inside_convex_hull = point_in_convex_hull_gpu(blocks_rect_rotated[:, :, 1:5, :].clone(), gripper_pos[:, :2].clone())

    out_of_hull_penalty = torch.where(
        inside_convex_hull,
        torch.zeros_like(dist_penalty),
        torch.ones_like(dist_penalty) * -1.0
    )
   
    gripper_idle_penalty = torch.zeros_like(grasp_reward, device=grasp_reward.device)
    gripper_idle_penalty = torch.where( (progress_buf > 2.0) & (~is_graspable) & (torch.norm(prev_gripper_pos[:, :2] - gripper_pos[:, :2], dim=1) < 1e-3), gripper_idle_scale, gripper_idle_penalty)
    prev_gripper_pos.copy_(gripper_pos[:, :2].clone())  # Update previous gripper position
    graspable_env_test_case_ids  = torch.nonzero(is_graspable).squeeze(1)
    # print("graspable_envs/testcases:", graspable_env_test_case_ids)
    # print("len of successful envs/testcases:", graspable_env_test_case_ids.shape)


    # find closest four distances from target to other object centers
    dists = []
    for ib in range(1, blocks_rect_rotated.shape[0]):      
        dis = torch.norm(blocks_rect_rotated[ib, :, 0, :] - blocks_rect_rotated[0, :, 0, :], dim=1)
        dists.append(dis)  # (num_envs, 1)
    dist = torch.stack(dists, dim=1)  # (num_envs, 5)


    # print("dist shape:", dist.shape)
    # sort distances and take the two smallest ones
    # smallest_two_distances = torch.sort(dist, dim=1).values[:, :2]  # (num_envs, 5)
    sorted_distances = torch.sort(dist, dim=1).values  # (num_envs, 5)
    
    smallest_dist1 = sorted_distances[:, 0]  # (num_envs, 1)
    smallest_four_distances = sorted_distances[:, :4]  # (num_envs, 1)

    margin = 0.09  # example: want each of the 4 closest >= 6cm
    viol = torch.relu(margin - smallest_four_distances)  # (N,4)
    avg_coup_penalty = - 100.0 * (viol**2).sum(dim=1)  # squared gives strong push when too close
    # print("avg tight coupling penalty[115]:", avg_coup_penalty[115].item())
    # print("smallest four distances[115]:", smallest_four_distances[115])

    # smallest_two_distances = torch.gather(dist, dim=1, index=smallest_two_indices)  # (num_envs, 2)
    # print("smallest two distances shape:", smallest_two_distances.shape)
    # avg_tight_coupling_penalty = torch.sum(smallest_four_distances, dim=1)  # (num_envs, )
    curr_min_tight_coupling_dist = smallest_dist1.clone()  # (num_envs, )
    # print("current min tight coupling dist[115]:", curr_min_tight_coupling_dist[115].item())
    # print("prev min tight coupling dist[115]:", prev_min_tight_coupling_dist[115].item())
    # print("min tight coupling difference to achieve[115]:", min_tight_coupling_tolerance)
    min_tight_coupling_penalty = torch.where(
        curr_min_tight_coupling_dist >= (prev_min_tight_coupling_dist + min_tight_coupling_tolerance), 
        0.0,
        (10 * (curr_min_tight_coupling_dist - prev_min_tight_coupling_dist - min_tight_coupling_tolerance))
    )
    prev_min_tight_coupling_dist.copy_(curr_min_tight_coupling_dist.clone())  # Update previous min tight coupling distance
    # assert torch.sum(is_graspable & (mean_overlap_penalty < 0)) == 0, "If it is graspable then mean overlap penalty should be removed!"
    assert torch.sum(is_graspable & (gripper_idle_penalty < 0)) == 0, f"If it is graspable then gripper idle penalty should be removed!, \n progress_buf: {progress_buf[is_graspable].detach().cpu()}"
    # , envs: {all_envs[is_graspable].detach().cpu()}"    
    inv_of_curr = inverse_action[actions] # (B, 1) or (B, )
    is_backtrack = (prev_actions == inv_of_curr)

    backtrack_penalty = torch.where(is_backtrack & (progress_buf > 10.0),
                                torch.ones_like(grasp_reward, device=grasp_reward.device) * -0.5,
                                torch.zeros_like(grasp_reward, device=grasp_reward.device))
    
    prev_actions.copy_(actions)
    
    # + mean_overlap_penalty 
    rewards =  grasp_reward + overlap_penalty  + (dist_penalty + out_of_hull_penalty + gripper_idle_penalty + min_tight_coupling_penalty + avg_coup_penalty + backtrack_penalty)/1
    assert min_tight_coupling_penalty.max() <= 0, "Min tight coupling penalty should be non-positive"
    assert gripper_idle_penalty.max() <= 0, "Gripper idle penalty should be non-positive"
    assert overlap_penalty.max() <= 0, "Overlap penalty should be non-positive"
    assert dist_penalty.max() <= 0, "Distance penalty should be non-positive"
    assert out_of_hull_penalty.max() <= 0, "Out of hull penalty should be non-positive"
    assert avg_coup_penalty.max() <= 0, "Avg tight coupling penalty should be non-positive"
    assert backtrack_penalty.max() <= 0, "Backtrack penalty should be non-positive"
    # + avg_tight_coupling_penalty
    # print("out of hull penalty[115]:", out_of_hull_penalty[115].item())
    # print("gripper idle penalty[115]:", gripper_idle_penalty[115].item())
    # print("min tight coupling penalty[115]:", min_tight_coupling_penalty[115].item())
    # # print("avg tight coupling penalty[115]:", avg_tight_coupling_penalty[115].item())
    # print("overlap penalty[115]:", overlap_penalty[115].item())
    # print("backtrack penalty[115]:", backtrack_penalty[115].item())
    # print("total reward[115]:", rewards[115].item())
    # print("=================================================================")
    # print("Per step overlap penalty range  - mean:", overlap_penalty.mean().item(), ", max:", overlap_penalty.max().item(), ", min:", overlap_penalty.min().item())
    # print("Per step idle penalty range  - mean:", gripper_idle_penalty.mean().item(), ", max:", gripper_idle_penalty.max().item(), ", min:", gripper_idle_penalty.min().item())
    # print("Per step grasp reward range - mean:", grasp_reward.mean().item(), ", max:", grasp_reward.max().item(), ", min:", grasp_reward.min().item())
    # print("Per step dist penalty range - mean:", dist_penalty.mean().item(), ", max:", dist_penalty.max().item(), ", min:", dist_penalty.min().item())
    # print("Per step total out of hull penalty range - mean:", out_of_hull_penalty.mean().item(), ", max:", out_of_hull_penalty.max().item(), ", min:", out_of_hull_penalty.min().item())
    # print("Per step total reward range - mean:", rewards.mean().item(), ", max:", rewards.max().item(), ", min:", rewards.min().item())
    # print("Per step min tight coupling penalty range - mean:", min_tight_coupling_penalty.mean().item(), ", max:", min_tight_coupling_penalty.max().item(), ", min:", min_tight_coupling_penalty.min().item())
    # print("Per step avg tight coupling penalty range - mean:", avg_tight_coupling_penalty.mean().item(), ", max:", avg_tight_coupling_penalty.max().item(), ", min:", avg_tight_coupling_penalty.min().item())
    # + gripper_idle_penalty + mean_overlap_penalty + avg_tight_coupling_penalty
    # + min_tight_coupling_penalty
    
    rewards = torch.where(is_graspable, grasp_reward, rewards)
    reset_buf = torch.where(is_graspable & (reset_buf == 0), torch.ones_like(reset_buf), reset_buf)
    successes = torch.where(reset_buf, torch.ones_like(is_graspable), successes)
    reset_buf = torch.where((grasp_q_values_parallel == -2.0) & (reset_buf == 0), torch.ones_like(reset_buf), reset_buf) # Reset if target obj goes out of camera view
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) & (reset_buf == 0), torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf, prev_gripper_pos, successes, prev_min_tight_coupling_dist, prev_actions
    return rewards, reset_buf, prev_gripper_pos, successes, prev_min_tight_coupling_dist, prev_actions

