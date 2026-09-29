"""Explicit-reset evaluation task; no change to legacy training environments."""
from pathlib import Path
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply
from isaacgymenvs.tasks.more_open_loop import MoreOpenLoop


class MoreEvaluation(MoreOpenLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_resets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Actual collision mesh vertices, including target geometry. The
        # teacher's target token instead describes a gripper-clearance rectangle.
        asset = Path(__file__).resolve().parents[2] / "assets/urdf/more/blocks-more"
        polys = []
        for name in ["concave", "cylinder", "cube", "half-cube", "rect", "triangle"]:
            vertices = [[float(x) for x in line.split()[1:4]]
                        for line in (asset / (name+".obj")).read_text().splitlines()
                        if line.startswith("v ")]
            polys.append(np.asarray(vertices, np.float32))
        nv = max(map(len, polys))
        padded = np.stack([np.pad(p, ((0,nv-len(p)),(0,0)), mode="edge") for p in polys])
        self.mesh_vertices = torch.tensor(padded, device=self.device)[self.all_block_name_ids.long()]
        self.requested_initial_oow = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.evaluation_cartesian_paths = torch.zeros((self.num_envs,2,2),dtype=torch.float32,device=self.device)
        self.evaluation_cartesian_active = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.evaluation_tcp_arc = torch.zeros(self.num_envs,dtype=torch.float64,device=self.device)
        self.evaluation_tcp_last = self.gripper_pos[:,:2].clone()
        self._evaluation_tcp_arc_enabled = False

    def reset_idx(self, env_ids, from_where="init"):
        if from_where == "post_physics_step":
            raise RuntimeError("Automatic reset attempted in a single-attempt evaluation")
        super().reset_idx(env_ids, from_where)
        if hasattr(self, "explicit_resets"):
            self.explicit_resets[env_ids.long()] += 1

    def post_physics_step(self):
        # Parent implements delayed reset from the previous step's reset_buf.
        # Preserve current terminal observations, but never execute that reset.
        self.reset_buf.zero_()
        super().post_physics_step()
        if hasattr(self, "step_mesh_oow"):
            self.step_mesh_oow |= self.physical_oow()

    def pre_physics_step(self, actions):
        self.step_mesh_oow = self.physical_oow().clone()
        super().pre_physics_step(actions)
        selected=self.evaluation_cartesian_active&~self.frozen&~self.replay_active
        if selected.any():
            # Replace XY only. The parent's safe height and fixed orientation
            # references remain untouched, and both waypoints stay inside the
            # same canonical EEF workspace used for every discrete primitive.
            start=self.start_pos[selected,:2]
            path=self.evaluation_cartesian_paths[selected]
            wp1=start+path[:,0]
            wp2=start+path[:,1]
            wx,wy=self.ws_x,self.ws_y
            wp1[:,0]=torch.clamp(wp1[:,0],float(wx[0]),float(wx[1]))
            wp1[:,1]=torch.clamp(wp1[:,1],float(wy[0]),float(wy[1]))
            wp2[:,0]=torch.clamp(wp2[:,0],float(wx[0]),float(wx[1]))
            wp2[:,1]=torch.clamp(wp2[:,1],float(wy[0]),float(wy[1]))
            self.wp1_pos[selected,:2]=wp1
            self.wp2_pos[selected,:2]=wp2
            for i in selected.nonzero(as_tuple=False).flatten().tolist():
                self.recording['waypoints'][i][-1].update(
                    wp1=self.wp1_pos[i].cpu().tolist(),wp2=self.wp2_pos[i].cpu().tolist(),
                    cartesian_override=True,
                    requested_relative_path=self.evaluation_cartesian_paths[i].cpu().tolist())
        scale=getattr(self,'evaluation_action_scale',None)
        if scale is not None:
            selected=(scale<1)&~self.frozen&~self.replay_active
            if selected.any():
                # Simulator-only short-step extension. Never alter the nominal
                # or baseline 4-cm action path, height, orientation or workspace.
                s=scale[selected,None]
                start=self.start_pos[selected,:2]
                self.wp1_pos[selected,:2]=start+s*(self.wp1_pos[selected,:2]-start)
                self.wp2_pos[selected,:2]=start+s*(self.wp2_pos[selected,:2]-start)
                for i in selected.nonzero(as_tuple=False).flatten().tolist():
                    self.recording['waypoints'][i][-1].update(
                        wp1=self.wp1_pos[i].cpu().tolist(),wp2=self.wp2_pos[i].cpu().tolist(),
                        action_scale=float(scale[i]))

    def set_evaluation_action_scale(self, scale):
        scale=torch.as_tensor(scale,device=self.device,dtype=torch.float32)
        if scale.shape!=(self.num_envs,) or not torch.isfinite(scale).all() or ((scale<=0)|(scale>1)).any():
            raise ValueError('Expected one finite action scale in (0,1] per environment')
        self.evaluation_action_scale=scale

    def set_evaluation_cartesian_paths(self, paths, active):
        paths=torch.as_tensor(paths,device=self.device,dtype=torch.float32)
        active=torch.as_tensor(active,device=self.device,dtype=torch.bool)
        if paths.shape!=(self.num_envs,2,2) or active.shape!=(self.num_envs,):
            raise ValueError('Expected one two-waypoint XY path and active flag per environment')
        if active.any() and not torch.isfinite(paths[active]).all():
            raise ValueError('Active Cartesian paths must be finite')
        self.evaluation_cartesian_paths.copy_(paths)
        self.evaluation_cartesian_active.copy_(active)
        self.enable_evaluation_tcp_arc()

    def enable_evaluation_tcp_arc(self):
        """Measure planar TCP travel for any active evaluation actor."""
        if not self._evaluation_tcp_arc_enabled:
            self.evaluation_tcp_last=self.gripper_pos[:,:2].clone()
            self._evaluation_tcp_arc_enabled=True

    def _after_simulate(self):
        super()._after_simulate()
        if not self._evaluation_tcp_arc_enabled:
            return
        # Physics-frame, not decision-frame, TCP arc length. Restore the task's
        # cached rigid-body tensor after reading so controller dynamics retain
        # the exact same observation timing as the original task.
        old=self.rb_states.clone()
        self.gym.fetch_results(self.sim,True)
        try:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            current=self.rb_states[self.gripper_idxs.long(),:2].clone()
        finally:
            self.rb_states.copy_(old)
        moving=~self.frozen&~self.replay_active
        self.evaluation_tcp_arc[moving]+=(current[moving]-self.evaluation_tcp_last[moving]).norm(dim=1).double()
        self.evaluation_tcp_last=current

    def _apply_plan_target_one_substep(self):
        # Check containment at physics-frame boundaries as well as at decision
        # boundaries, so crossing out and back within a primitive still fails.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.step_mesh_oow |= self.physical_oow()
        super()._apply_plan_target_one_substep()

    def physical_oow(self):
        st = self.block_state
        v = self.mesh_vertices
        quat = st[:, :, None, 3:7].expand(*v.shape[:-1], 4)
        xyz = quat_apply(quat.reshape(-1,4), v.reshape(-1,3)).reshape_as(v)
        xyz = xyz + st[:, :, None, :3]
        return ((xyz[...,0] < self.ws_x[0]) | (xyz[...,0] > self.ws_x[1]) |
                (xyz[...,1] < self.ws_y[0]) | (xyz[...,1] > self.ws_y[1])).any(dim=-1).any(dim=-1)

    def invalid_state(self):
        return (~torch.isfinite(self.block_state).flatten(1).all(dim=1) |
                ~torch.isfinite(self.gripper_pos).all(dim=1))

    def begin_attempt(self, block_state, permutations):
        """Reset, hold for one control interval, then read fresh physics/render.

        The four-frame initialization hold is part of the declared protocol;
        initial graspability is measured afterward and reported separately.
        It is identical for every compared actor and does not run a policy.
        """
        ids = torch.arange(self.num_envs, device=self.device)
        self.evaluation_action_scale=None
        self.evaluation_cartesian_active.zero_()
        self.evaluation_cartesian_paths.zero_()
        self.evaluation_tcp_arc.zero_()
        self._evaluation_tcp_arc_enabled=False
        self.frozen[:] = True
        self.replay_active[:] = False
        self.replay_done[:] = False
        self.default_block_state[:] = block_state
        self.reset_idx(ids, from_where="explicit_evaluation")
        self.t_obs_perm[:] = torch.as_tensor(permutations, device=self.device)
        self.requested_initial_oow = self.physical_oow().clone()
        self.plan_done_flag[:] = False
        self.ctrl_orn_ref = None
        self.ctrl_last_phase[:] = -1
        self.step(torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        self.progress_buf.zero_()
        self.reset_buf.zero_()
        self.t_needs_init[:] = True
        self.plan_active[:] = False
        self.frozen[:] = False
        self.ctrl_orn_ref = self.gripper_rot.clone()
        self.evaluation_tcp_last=self.gripper_pos[:,:2].clone()
        self.ctrl_last_phase[:] = -1
        self.clear_recording()
        self.compute_observations()
        self.obs_dict["obs"] = self.obs_buf.clamp(-self.clip_obs, self.clip_obs).to(self.rl_device)
        # Geometry and root state must describe the same CURRENT reset scene.
        centers = self.blocks_rect_rotated[:, :, 0, :].permute(1,0,2)
        if not torch.allclose(centers, self.block_state[:,:,:2], atol=1e-5):
            raise RuntimeError("Stale geometry after reset")
        return self.obs_dict
