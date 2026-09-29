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

# Concave Faces: 9cm x 4.5 cm (rect face), 9cm x 4.5 cm (side cavity curve), 9 cm x 4.5 cm(front cavity curve)
# Cylinder Faces: 4.5cm x 4.5cm (top/bottom), 4.5 cm x 4.5cm (lateral/sides)
# Cube faces: 4.5cm x 4.5cm
# half-cube faces: 4.5cm x 4.5cm, 4.5cmx 2.25cm
# Triangle: base - 8.5cm x 4.5cm flipped to side - 4.5cm x 8.5cm
# rect: 9 cm x 4.5 cm
# Gripper: 12cm x 2.2/2.3 cm - you want to achieve a min clearance of this area around the target object in one of the 16 possible rotations (360 deg/16).   
# version1: fix cube,cylinder,half-cube as targets as 16 clearance rectangles are easy to compute.
# default_home: [56.50 (-90 for sim), -113.61, 151.06, -127.36, -89.75, 326.49, 0.0, 0.0]
import numpy as np
import os
from colorama import Fore
from tqdm import tqdm 
import torch
import time 
# torch.nn.functional
import torch.nn.functional as F
import random 
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_conjugate, quat_mul

from  .more import More, compute_more_reward_jit
from .utils.more_utils import quaternion_to_euler, euler_to_quat
from .utils.more_jit_utils import (
                                rotate_rectangles,
                                compute_free_area_ratio,
                                point_in_convex_hull_gpu
)
from .constants import (
    IS_REAL,
    IMAGE_OBJ_CROP_SIZE,
    IMAGE_SIZE,
    PIXEL_SIZE,
    # WORKSPACE_LIMITS,
    WORKSPACE_PUSH_BORDER,
    PUSH_LENGTH
)
random.seed(1600)
WORKSPACE_LIMITS = np.asarray([[0.176, 0.724], [-0.424, 0.224], [0.0001, 0.4]])



class MoreTest(More):

    def __init__(self, cfg, test, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(
                            cfg=cfg, 
                            test=test,
                            rl_device=rl_device, 
                            sim_device=sim_device, 
                            graphics_device_id=graphics_device_id, 
                            headless=headless,
                            virtual_screen_capture=virtual_screen_capture, 
                            force_render=force_render)
        print("\n=======================================")
        print(">>>>>>>>>>>num envs for success sign initialization:", self.num_envs)
        print("=======================================\n")
        self.success_sign = torch.zeros(size=(self.num_envs, ), device=self.device)
        self.gripperTarget = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.ep_start_wall = np.full(self.num_envs, time.perf_counter(), dtype=np.float64)
        self.success_recorded = torch.zeros(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.success_wall_times = []
        self.success_steps = []
        self.total_eps = 0
        self.total_success_eps = 0




        # print("Inside MoreTest init...")

    def _create_envs(self, num_envs, spacing, num_per_row, num_objects):
        self.inverse_action = torch.tensor(
                                [2,3,0,1, 13,14,15, 10,11,12, 7,8,9, 4,5,6],
                                device=self.device, dtype=torch.long
                            )
        # repeat for B, envs
        # self.inverse_action = self.inverse_action.unsqueeze(0).repeat(num_envs, 1) # (B, 16)
        self.prev_action_idx = -1 * torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.inverse_action = torch.tensor(
                                [2,3,0,1, 13,14,15, 10,11,12, 7,8,9, 4,5,6],
                                device=self.device, dtype=torch.long
                            )
        # repeat for B, envs
        # self.inverse_action = self.inverse_action.unsqueeze(0).repeat(num_envs, 1) # (B, 16)
        self.prev_action_idx = -1 * torch.ones(self.num_envs, device=self.device, dtype=torch.long)
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
        print('asset_root:', asset_root)
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
            ur5e_asset = self.gym.load_asset(self.sim, ur5e_asset_folder, "ur5e_simplified_gripper.urdf", robot_asset_options)
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
        dim_x = WORKSPACE_LIMITS[0][1] - WORKSPACE_LIMITS[0][0]
        dim_y = WORKSPACE_LIMITS[1][1] - WORKSPACE_LIMITS[1][0]
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
        workspace_pose.p = gymapi.Vec3(0.55, 0.01, 0.0005)

        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.flip_visual_attachments = False # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        wall_asset_options.fix_base_link = True
        wall_asset = self.gym.load_asset(self.sim, wall_asset_folder, "wall.urdf", wall_asset_options)
        wall_pose = gymapi.Transform()
        wall_pose.p = gymapi.Vec3(0.55, 0.01, 0.0005)

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
        self.walls= []
        self.envs = []

        self.cameras = []
        # self.cam_color_tensors = []
        self.cam_depth_tensors = []
        self.cam_segm_tensors = []
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True 
        camera_props.width = 224
        camera_props.height = 224
        camera_props.horizontal_fov = 0.02578
        camera_props.near_plane = 999.75
        camera_props.far_plane = 1001.0
        _camera_local_transform = gymapi.Transform()
        # _camera_local_transform.p = gymapi.Vec3(0.5, 0, 999.8)
        _camera_local_transform.p = gymapi.Vec3(0.55, 0.05, 999.8)
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
        test_case_iter = -1 # 0000001.txt .... 000690.txt are 512 envs used for training
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
            # if not os.path.exists(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt"):
            #     while not os.path.exists(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt"):
            #         test_case_iter += 1
            if True:
                    self.chosen_scenes.append(test_case_iter)
                    with open(f"{rdir}/{self.test_case_dir}/{test_case_iter:06d}.txt", "rb") as f:
                        # print(f"loaded test case: {test_case_iter:06d}.txt....")
                    # chosen_id = 1474
                    # with open(f"{rdir}/{self.test_case_dir}/{chosen_id:06d}.txt", "rb") as f:
                    #     print(f"loaded test case: {chosen_id:06d}.txt....")
                        scene_info = f.readlines()
                        num_blocks = len(scene_info)
                        self.num_blocks_each_env.append(num_blocks)
                        for block_idx in range(num_blocks):
                            obj_info = scene_info[block_idx].split() 
                            # print("obj_info:", obj_info)
                            # print("obj_info:", obj_info)
                            obj_pose = gymapi.Transform()
                            obj_pose.p = gymapi.Vec3(
                                float(obj_info[4]),
                                float(obj_info[5]),
                                float(obj_info[6]),
                            )
                            # if "triangle" in obj_info[0].decode():
                            #     if float(obj_info[9]) <= -2.3: 
                            #         angle_x = 0.785
                            #     elif float(obj_info[9]) <= -0.77:
                            #         angle_x = 2.356
                            #     elif float(obj_info[9]) >= 2.30:
                            #         angle_x = -0.785 
                            #     elif float(obj_info[9]) >= 0.77:
                            #         angle_x = -2.356
                            #     print("given rx:", float(obj_info[9]), "modified rx:", angle_x)
                            #     obj_pose.r = gymapi.Quat.from_euler_zyx(
                            #         float(obj_info[7]),
                            #         float(obj_info[8]),
                            #         angle_x
                            #     )
                            # else:
                            obj_pose.r = gymapi.Quat.from_euler_zyx(
                                float(obj_info[7]),
                                float(obj_info[8]),
                                float(obj_info[9]),
                                # float(obj_info[10]),
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

                        block_assets = []
                        block_names = []
                        unique_blocks = {}
                        for block_idx in range(num_blocks):
                            block_file = block_files[block_idx]
                            if block_file in unique_blocks:
                                block_assets.append(unique_blocks[block_file])
                            else:
                            #     if "concave" in block_file.decode():
                            #         block_asset_options.vhacd_enabled = True
                            #         block_asset_options.vhacd_params.resolution = 64000000
                            #         block_asset_options.vhacd_params.alpha = 0.005
                            #         block_asset_options.vhacd_params.beta = 0.005
                            #     else:
                            #         block_asset_options.vhacd_enabled = False

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
            # cam_color_tensor = self.gym.get_camera_image_gpu_tensor(
            #         self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
            #     )
            # torch_cam_color_tensor = gymtorch.wrap_tensor(cam_color_tensor)
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
        

        self.gripper_idxs = to_torch(self.gripper_idxs, dtype=torch.long, device=self.device)
        self.default_block_state = to_torch(self.default_block_state, device=self.device, dtype=torch.float).view(
            self.num_envs, self.num_objects, 13
        )
        dists = []
        print("num objects:", self.num_objects)
        print("num objects:", self.num_objects)
        for ib in range(1, self.num_objects):      
            dist = torch.norm(self.default_block_state[:, ib, :3] - self.default_block_state[:, 0, :3], dim=1)  # (num_envs, 2)
            dists.append(dist.unsqueeze(1))  
            dists.append(dist.unsqueeze(1))  

        # Find the two closest distances for each environment
        dist = torch.stack(dists, dim=1)  
        print("dist shape:", dist.shape)
        dist = torch.stack(dists, dim=1)  
        print("dist shape:", dist.shape)
        # sort distances and take the two smallest ones
        smallest_four_distances = torch.sort(dist, dim=1).values[:, :4]  
        print("smallest four distances shape:", smallest_four_distances.shape)
        tight_coupling_dist = torch.sum(smallest_four_distances, dim=1) 
        self.default_min_tight_coupling_dist_for_reset = smallest_four_distances[:, 0, 0]  
        self.prev_min_tight_coupling_dist = self.default_min_tight_coupling_dist_for_reset.clone()
        smallest_four_distances = torch.sort(dist, dim=1).values[:, :4]  
        print("smallest four distances shape:", smallest_four_distances.shape)
        tight_coupling_dist = torch.sum(smallest_four_distances, dim=1) 
        self.default_min_tight_coupling_dist_for_reset = smallest_four_distances[:, 0, 0]  
        self.prev_min_tight_coupling_dist = self.default_min_tight_coupling_dist_for_reset.clone()
        self.default_tight_coupling_dist = tight_coupling_dist.squeeze().clone()
        self.prev_tight_coupling_dist = self.default_tight_coupling_dist.clone()


        self.num_actors = 1 + 1 + 0+ num_objects # 1 workspace + 1 robot + 1 wall + 6 target objects
        link_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.ur5es[0])

        finger_names = [name for name in link_names if "pad" in name]
        self.gripper_handles = [
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5es[0], name) for name in finger_names
        ]
        self.init_data()

    def make_inverse_action(self, device):
        # net displacement for each action id in free space
        all_a = torch.arange(16, device=device)
        d1, d2 = self.build_two_step_plan(all_a, total=0.02)
        net = d1 + d2  # (16,2)

        inv = torch.empty(16, dtype=torch.long, device=device)
        for i in range(16):
            # choose j that best cancels i
            j = torch.argmin(torch.norm(net + net[i:i+1], dim=1))
            inv[i] = j
        return inv
    

    def make_inverse_action(self, device):
        # net displacement for each action id in free space
        all_a = torch.arange(16, device=device)
        d1, d2 = self.build_two_step_plan(all_a, total=0.02)
        net = d1 + d2  # (16,2)

        inv = torch.empty(16, dtype=torch.long, device=device)
        for i in range(16):
            # choose j that best cancels i
            j = torch.argmin(torch.norm(net + net[i:i+1], dim=1))
            inv[i] = j
        return inv
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        print("actions[0]:", self.actions[0].item())
        if self.actions.dim() == 2:
            best = self.actions.squeeze(-1).long() 
        else:
            best = self.actions.long() 

        # start a new 2-step plan only when not active 
        start_new = ~self.plan_active 

        if torch.any(start_new):
            best_new = best[start_new]
            d1, d2 = self.build_two_step_plan(best_new, total=0.02)
            # snapshot start pose once 
            # self.start_pos[start_new] = self.gripper_pos[start_new].clone()

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

            # clamp workspace
            wp1[:,0] = torch.clamp(wp1[:,0], 0.30, 0.75)
            wp1[:,1] = torch.clamp(wp1[:,1], -0.224, 0.284)
            wp2[:,0] = torch.clamp(wp2[:,0], 0.30, 0.75)
            wp2[:,1] = torch.clamp(wp2[:,1], -0.224, 0.284)

            self.start_pos[start_new] = self.gripper_pos[start_new].clone()
            self.wp1_pos[start_new] = wp1
            self.wp2_pos[start_new] = wp2

            self.plan_phase[start_new] = 0
            self.plan_tick[start_new] = 0
            self.plan_active[start_new] = True

        
        success_env_ids = (self.success_sign > 0).nonzero(as_tuple=False).squeeze(-1)
        # print("success env ids shape:", success_env_ids.shape, success_env_ids)
        
        fail_env_ids = (self.success_sign <= 0).nonzero(as_tuple=False).squeeze(-1)
        # print("fail env ids shape:", fail_env_ids.shape, fail_env_ids)

        total_cases = success_env_ids.shape[0] + fail_env_ids.shape[0]

        self.success_rate = (success_env_ids.shape[0] / total_cases * 100)
        self.fail_rate = (fail_env_ids.shape[0] / total_cases * 100)
        # print("Action idx of env 0:", self.prev_action_idx[0].item())
        # max_progress = self.progress_buf.max().item()
        # print("current max progress:", max_progress)
        # if max_progress > 160: # episode_length - 4
        #     print("==========================================================")
        #     print("success env ids:", success_env_ids)
        #     print("fail env ids:", fail_env_ids)
        #     print("===========================================================")

        # print("=======================CASES===================================")
        # print(f"total cases: {total_cases}")
        # # print(f"success cases: {success_env_ids.shape[0]}")
        # # print(f"fail cases: {fail_env_ids.shape[0]}")
        # print(f"success rate: {self.success_rate}")
        # print(f"fail rate: {self.fail_rate}")
        # print("==========================================================")

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
        
        # Mark the evironment successful when a (positive reward) graspable clearance is achieved once
        # For correctness, repeat success due to resets before max_episode_length are not counted more than once.  
        self.success_sign = torch.where(self.rew_buf > 0.0, torch.ones_like(self.rew_buf), self.success_sign)

        success_now = (self.success_sign > 0) 
        to_record = success_now & (~self.success_recorded)
        if torch.any(to_record):
            now = time.perf_counter()
            ids = to_record.nonzero(as_tuple=False).squeeze(-1) # (B, )
            ids_cpu = ids.cpu().numpy()
            wall = now - self.ep_start_wall[ids_cpu] # (B, )
            steps = self.progress_buf[ids].cpu().numpy() # (B, )

            self.success_wall_times.extend(wall.astype(float).tolist())
            self.success_steps.extend(steps.astype(int).tolist())
            self.success_recorded[ids] = True
            self.total_success_eps += ids.numel()
        
    def reset_idx(self, env_ids, from_where="init"):
        env_ids_int32 = env_ids.to(dtype=torch.long)
        # print(f"reset_idx MoreTest, from_where: {from_where}, len(env_ids): {len(env_ids_int32.tolist())}, env_ids: {env_ids_int32.tolist()}")
        # print(f"progress_buf: {self.progress_buf[env_ids_int32].tolist()}, reset_buf: {self.reset_buf[env_ids_int32].tolist()}, rew_buf: {self.rew_buf[env_ids_int32].tolist()}")
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
        self.reset_buf[env_ids_int32] = 0
        self.rew_buf[env_ids_int32] = 0.0
        self.successes[env_ids_int32] = 0.0       
        # print("default_min_tight_coupling_dist_for_reset shape:", self.default_min_tight_coupling_dist_for_reset.shape)
        self.prev_min_tight_coupling_dist[env_ids_int32] = self.default_min_tight_coupling_dist_for_reset[env_ids_int32] 
        self.prev_gripper_pos[env_ids_int32] = self.default_gripper_pos[env_ids_int32].clone()
        self.plan_active[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.bool)
        self.plan_phase[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.long)  # 0 or 1
        self.plan_tick[env_ids_int32] = torch.zeros(len(env_ids_int32), device=self.device, dtype=torch.long)  # 0..(K-1)

        self.start_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.wp1_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.wp2_pos[env_ids_int32] = torch.zeros((len(env_ids_int32), 3), device=self.device)
        self.prev_action_idx[env_ids_int32] = -1 * torch.ones(len(env_ids_int32), device=self.device, dtype=torch.long)
        
        now = time.perf_counter()
        ids = env_ids_int32.cpu().numpy()
        self.ep_start_wall[ids] = now
        self.success_recorded[env_ids_int32] = False
        self.total_eps += len(env_ids_int32)