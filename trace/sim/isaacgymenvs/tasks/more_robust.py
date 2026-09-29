"""MoreRobust — More variant fixing the bulldozer failure (Aug 26 2026 audit).

Changes vs More (each toggleable via cfg["env"]["robust"], defaults in code):

1. OUT-OF-WORKSPACE termination + penalty: pushing any non-exempt block out
   of the 55x45 cm workspace ends the episode as a FAILURE (the missing rule
   that made bulldozing reward-optimal). Blocks already out at scene load
   (boundary-padding dummies) are exempt.
2. POTENTIAL-BASED distance shaping: the parent's absolute per-step distance
   penalty (accumulates with time -> rewards charging straight at the
   target) is replaced by the potential difference Phi(s')-Phi(s),
   Phi = distance_scale * relu(d - desired). Path-independent: no rush/
   loiter incentive (Ng et al. 1999). The too-close penalty is kept as is.
3. RESET RANDOMIZATION:
   - scene symmetry: each reset applies one of {identity, rot180, mirror-y,
     mirror-x} about the workspace center to the whole scene (equivalent to
     approaching from a different direction; skipped for envs containing
     non-flat blocks) + per-block pose noise (~perception noise).
   - home randomization: shoulder-pan noise (height-safe) arcs the EEF
     start sideways.
   Together these break the fixed straight-East sweep.
4. DOMAIN RANDOMIZATION (init-time, per block actor): friction U(0.2,0.5),
   mass scale U(0.7,1.4) — also the knob the open-loop certifier perturbs.

Evaluate with tools/eval_strict.py:
  python tools/eval_strict.py task=MoreRobust test=True headless=True \
      num_envs=128 task.env.test_cases.difficulty_choice=test-128 \
      task.env.robust.randomizeReset=False checkpoint=...
"""
import math
import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch  # noqa: F401 (gymapi used for props)

from isaacgymenvs.tasks.more import More

# workspace = the real robot's reachable 0.448 m square (fort REAL limits,
# PMBS): x in [0.276, 0.724], y in [-0.224, 0.224], centre (0.5, 0)
WS_CENTER = (0.5, 0.0)
WS_SIDE_DEFAULT = 0.448             # cfg env.robust.wsSide overrides per run
WS_X = (WS_CENTER[0] - WS_SIDE_DEFAULT / 2, WS_CENTER[0] + WS_SIDE_DEFAULT / 2)
WS_Y = (WS_CENTER[1] - WS_SIDE_DEFAULT / 2, WS_CENTER[1] + WS_SIDE_DEFAULT / 2)
WS_CX = WS_CENTER[0]


class MoreRobust(More):
    def __init__(self, cfg, test, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        rb = cfg["env"].get("robust", {})
        self.rb_oow_penalty = float(rb.get("oowPenalty", -5.0))
        self.rb_terminate_oow = bool(rb.get("terminateOnOow", True))
        self.rb_randomize_reset = bool(rb.get("randomizeReset", True))
        self.rb_scene_symmetry = bool(rb.get("sceneSymmetry", True))
        # certifier hooks: envs with idx % stride == 0 get ZERO pose noise
        # (nominal replicas); suppressResets records terminals w/o resetting
        self.rb_no_noise_stride = int(rb.get("noNoiseEnvStride", 0))
        self.rb_suppress_resets = bool(rb.get("suppressResets", False))
        self.rb_pose_noise_pos = float(rb.get("poseNoisePos", 0.003))
        self.rb_pose_noise_yaw = math.radians(float(rb.get("poseNoiseYawDeg", 2.0)))
        self.rb_pan_noise = math.radians(float(rb.get("homePanNoiseDeg", 8.0)))
        self.rb_domain_rand = bool(rb.get("domainRand", True))
        self.rb_friction_range = tuple(rb.get("frictionRange", (0.2, 0.5)))
        self.rb_mass_scale_range = tuple(rb.get("massScaleRange", (0.7, 1.4)))
        self.rb_potential_shaping = bool(rb.get("potentialShaping", True))
        # random-entry starts: reset the arm to one of the K reachable
        # workspace-perimeter anchors (tools/gen_entry_poses.py) instead of the
        # fixed home. Anchors whose xy sits too close to a scene object are
        # masked per episode; anchor 0 (classic home side) is the fallback.
        # Pan noise is skipped in this mode — anchor choice supplies the
        # start diversity, and pan arcs can sweep edge anchors into clutter.
        # workspace box side (m), centred at WS_CENTER. 0.448 = PMBS / paper;
        # 0.54 verified reachable on the UR5e (trace/hardware/reach_test.py).
        # The generator inset (0.348) is independent, so the dataset is shared.
        side = float(rb.get("wsSide", WS_SIDE_DEFAULT))
        self.ws_x = (WS_CENTER[0] - side / 2, WS_CENTER[0] + side / 2)
        self.ws_y = (WS_CENTER[1] - side / 2, WS_CENTER[1] + side / 2)
        self.ws_cx = WS_CENTER[0]
        self.rb_entry_anchors = bool(rb.get("entryAnchors", False))
        self.rb_entry_clear = float(rb.get("entryClearance", 0.09))

        super().__init__(cfg, test, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        if self.rb_entry_anchors:
            import json as _json
            pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(pkg, "cfg", "entry_poses.json")) as f:
                ep = _json.load(f)
            usable = [a for a in ep["anchors"] if a["reachable"]]
            self.entry_xy = torch.tensor([a["xy"] for a in usable],
                                         device=self.device)          # (K, 2)
            self.entry_joints = torch.tensor([a["joints"] for a in usable],
                                             device=self.device)      # (K, 6)
            print(f"[MoreRobust] random-entry starts: {len(usable)} anchors")

        # pristine scene poses: reset randomization transforms FROM these so
        # noise/symmetry never accumulates across episodes
        self._pristine_block_state = self.default_block_state.clone()

        # OOW exemption: blocks out of bounds already at scene load
        xy0 = self._pristine_block_state[:, :, 0:2]
        self.oow_exempt = self._oow_of(xy0)

        # envs where every block is flat (pure-z quat) are symmetry-transformable
        q = self._pristine_block_state[:, :, 3:7]
        tilt = torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2)
        self.symmetry_ok = (tilt < 0.05).all(dim=1)
        n_sym = int(self.symmetry_ok.sum())
        print(f"[MoreRobust] symmetry-transformable envs: {n_sym}/{self.num_envs}")

        # potential-shaping memory
        self.prev_target_dist = torch.norm(
            self.default_gripper_pos[:, :2] - self._pristine_block_state[:, 0, 0:2], dim=1)

        if self.rb_domain_rand:
            self._apply_domain_randomization()

        # periodic rollout-video logging (env.videoLog) — tiles the first few
        # envs' color cameras into an mp4 and logs to wandb when a run exists
        vl = cfg["env"].get("videoLog", {})
        self.vid_enabled = bool(vl.get("enabled", False)) and len(self.cam_color_tensors) > 0
        self.vid_num_envs = min(int(vl.get("numEnvs", 4)), self.num_envs, 4)
        self.vid_window = int(vl.get("windowSteps", 150))
        self.vid_interval = int(vl.get("intervalSteps", 25000))
        self.vid_fps = int(vl.get("fps", 20))
        self.vid_dir = str(vl.get("outDir", "videos"))
        self._vid_step = 0
        self._vid_frames = []
        self._vid_recording = False
        self._vid_next_start = int(vl.get("firstStart", 500))
        if self.vid_enabled:
            print(f"[MoreRobust] video logging ON: {self.vid_num_envs} envs, "
                  f"{self.vid_window}-step window every {self.vid_interval} steps")

        # re-randomize the initial reset done by More.__init__
        self.reset_idx(torch.arange(self.num_envs, device=self.device), from_where="robust_init")

    # ------------------------------------------------------------------ util
    def _oow_of(self, xy):
        return ((xy[..., 0] < self.ws_x[0]) | (xy[..., 0] > self.ws_x[1]) |
                (xy[..., 1] < self.ws_y[0]) | (xy[..., 1] > self.ws_y[1]))

    def _apply_domain_randomization(self):
        lo_f, hi_f = self.rb_friction_range
        lo_m, hi_m = self.rb_mass_scale_range
        rng = np.random.default_rng(self.cfg.get("seed", 0))
        for i, env_ptr in enumerate(self.envs):
            for b in range(self.num_objects):
                actor = self.gym.get_actor_handle(env_ptr, 2 + b)  # 0 ws, 1 robot
                sp = self.gym.get_actor_rigid_shape_properties(env_ptr, actor)
                for s in sp:
                    s.friction = float(rng.uniform(lo_f, hi_f))
                self.gym.set_actor_rigid_shape_properties(env_ptr, actor, sp)
                bp = self.gym.get_actor_rigid_body_properties(env_ptr, actor)
                for body in bp:
                    body.mass *= float(rng.uniform(lo_m, hi_m))
                self.gym.set_actor_rigid_body_properties(env_ptr, actor, bp,
                                                         recomputeInertia=True)
        print(f"[MoreRobust] domain randomization applied: friction U{self.rb_friction_range}, "
              f"mass x U{self.rb_mass_scale_range}")

    @staticmethod
    def _yaw_of(q):
        # yaw of (x,y,z,w) quats; blocks are flat so this is exact for them
        return torch.atan2(2.0 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                           1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2))

    def _randomized_default_state(self, env_ids):
        """Pristine scene poses -> symmetry transform + pose noise (per env)."""
        st = self._pristine_block_state[env_ids].clone()      # (E, B, 13)
        E, B = st.shape[0], st.shape[1]
        x, y = st[:, :, 0], st[:, :, 1]
        yaw = self._yaw_of(st[:, :, 3:7])

        t = torch.randint(0, 4, (E,), device=st.device)
        if hasattr(self, "_pick_symmetry"):
            t = self._pick_symmetry(env_ids, st, t)
        if not self.rb_scene_symmetry:
            t = torch.zeros_like(t)
        t = torch.where(self.symmetry_ok[env_ids], t, torch.zeros_like(t))
        te = t.unsqueeze(1).expand(E, B)
        # 1: rot180   2: mirror-y   3: mirror-x   (about workspace center)
        x = torch.where((te == 1) | (te == 3), 2 * self.ws_cx - x, x)
        y = torch.where((te == 1) | (te == 2), -y, y)
        yaw = torch.where(te == 1, yaw + math.pi, yaw)
        yaw = torch.where(te == 2, -yaw, yaw)
        yaw = torch.where(te == 3, math.pi - yaw, yaw)

        if self.rb_pose_noise_pos > 0:
            nx = (torch.rand_like(x) * 2 - 1) * self.rb_pose_noise_pos
            ny = (torch.rand_like(y) * 2 - 1) * self.rb_pose_noise_pos
            nyaw = (torch.rand_like(yaw) * 2 - 1) * self.rb_pose_noise_yaw
            if self.rb_no_noise_stride > 0:
                nominal = (env_ids % self.rb_no_noise_stride == 0).unsqueeze(1)
                nx = torch.where(nominal, torch.zeros_like(nx), nx)
                ny = torch.where(nominal, torch.zeros_like(ny), ny)
                nyaw = torch.where(nominal, torch.zeros_like(nyaw), nyaw)
            x, y, yaw = x + nx, y + ny, yaw + nyaw

        st[:, :, 0], st[:, :, 1] = x, y
        # rebuild flat (pure-z) quats; non-flat envs kept t=0 but noise still
        # rebuilds their quat as flat — restore originals for those envs
        st[:, :, 3] = 0.0
        st[:, :, 4] = 0.0
        st[:, :, 5] = torch.sin(yaw / 2)
        st[:, :, 6] = torch.cos(yaw / 2)
        not_flat = ~self.symmetry_ok[env_ids]
        if not_flat.any():
            st[not_flat, :, 3:7] = self._pristine_block_state[env_ids][not_flat, :, 3:7]
        return st

    # ----------------------------------------------------------------- reset
    def reset_idx(self, env_ids, from_where="init"):
        env_ids = env_ids.to(dtype=torch.long)
        if len(env_ids) == 0:
            return
        # guard: More.__init__ resets before our attributes exist
        randomize = self.rb_randomize_reset and hasattr(self, "_pristine_block_state")
        if randomize:
            self.default_block_state[env_ids] = self._randomized_default_state(env_ids)

        super().reset_idx(env_ids, from_where)

        if not hasattr(self, "_pristine_block_state"):
            return

        if self.rb_entry_anchors:
            # per-env: uniform among anchors clear of every object center;
            # anchor 0 (home side, dataset keep-out gated) as fallback
            obj_xy = self.default_block_state[env_ids, :, 0:2]        # (N,O,2)
            d = (self.entry_xy.view(1, -1, 1, 2) -
                 obj_xy.unsqueeze(1)).norm(dim=-1)                    # (N,K,O)
            free = (d.amin(dim=2) > self.rb_entry_clear).float()      # (N,K)
            score = torch.rand_like(free) * free
            pick = torch.where(free.sum(dim=1) > 0, score.argmax(dim=1),
                               torch.zeros_like(score.argmax(dim=1)))
            joints = self.entry_joints[pick]                          # (N,6)
            self.ur5e_dof_pos[env_ids, :6] = joints
            self.ur5e_dof_targets[env_ids, :6] = joints
            self.ur5e_dof_pos[env_ids, 6:] = 0.0
            self.ur5e_dof_targets[env_ids, 6:] = 0.0
            robot_ids = self.global_robot_ids[env_ids]
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets),
                gymtorch.unwrap_tensor(robot_ids), len(robot_ids))
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(robot_ids), len(robot_ids))
        elif randomize and self.rb_pan_noise > 0:
            pan = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * self.rb_pan_noise
            self.ur5e_dof_pos[env_ids, 0] += pan
            self.ur5e_dof_targets[env_ids, 0] += pan
            robot_ids = self.global_robot_ids[env_ids]
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets),
                gymtorch.unwrap_tensor(robot_ids), len(robot_ids))
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(robot_ids), len(robot_ids))

        # shaping baseline for the new episode (gripper default + fresh target)
        self.prev_target_dist[env_ids] = torch.norm(
            self.default_gripper_pos[env_ids, :2] -
            self.default_block_state[env_ids, 0, 0:2], dim=1)

    # ----------------------------------------------------------------- video
    def post_physics_step(self):
        super().post_physics_step()
        if self.vid_enabled:
            self._video_tick()

    def _video_tick(self):
        self._vid_step += 1
        if not self._vid_recording:
            if self._vid_step >= self._vid_next_start:
                self._vid_recording = True
                self._vid_frames = []
            else:
                return
        self.gym.start_access_image_tensors(self.sim)
        try:
            tiles = [self.cam_color_tensors[i].clone() for i in range(self.vid_num_envs)]
        finally:
            self.gym.end_access_image_tensors(self.sim)
        while len(tiles) < 4:
            tiles.append(torch.zeros_like(tiles[0]))
        top = torch.cat(tiles[:2], dim=1)
        bot = torch.cat(tiles[2:4], dim=1)
        self._vid_frames.append(torch.cat([top, bot], dim=0).cpu().numpy())

        if len(self._vid_frames) >= self.vid_window:
            self._vid_recording = False
            self._vid_next_start = self._vid_step + self.vid_interval
            self._flush_video()

    @staticmethod
    def _to_h264(path):
        """cv2's mp4v (MPEG-4 part 2) does not play in browsers, so wandb's
        player buffers forever. Transcode to H.264/yuv420p with the ffmpeg
        bundled in imageio-ffmpeg (has libx264); keep mp4v if unavailable."""
        import subprocess
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return path
        out = path[:-4] + "_h264.mp4"
        r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", out])
        if r.returncode != 0 or not os.path.exists(out):
            return path
        os.replace(out, path)
        return path

    def _flush_video(self):
        import os
        import cv2
        os.makedirs(self.vid_dir, exist_ok=True)
        # include the experiment name: parallel runs otherwise overwrite one
        # another's clips in a shared videos/ directory
        _exp = str(self.cfg.get("experiment", "") or
                   os.environ.get("EXPERIMENT_NAME", "run")).replace("/", "_")
        path = os.path.join(self.vid_dir,
                            f"rollout_{_exp}_step{self._vid_step:08d}.mp4")
        h, w = self._vid_frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.vid_fps, (w, h))
        for f in self._vid_frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        self._vid_frames = []
        path = self._to_h264(path)
        print(f"[MoreRobust] wrote {path}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"rollout_video": wandb.Video(path, fps=self.vid_fps,
                                                        format="mp4")})
        except Exception as e:  # video must never kill training
            print(f"[MoreRobust] wandb video log skipped: {e}")

    # ---------------------------------------------------------------- reward
    def compute_reward(self):
        super().compute_reward()  # parent jit fills rew_buf/reset_buf/successes

        d = torch.norm(self.gripper_pos[:, :2] - self.blocks_rect_rotated[0, :, 0, :], dim=1)
        is_graspable = self.grasp_q_parallel_values > 0.9

        if self.rb_potential_shaping:
            # exactly the parent's linear distance term, recomputed to remove it
            desired = self.desired_eef_to_target_distance
            old_linear = torch.where(
                d + 0.001 > desired,
                self.distance_scale * (0.001 + d - desired),
                torch.zeros_like(d))
            shaping = self.distance_scale * (torch.relu(d - desired) -
                                             torch.relu(self.prev_target_dist - desired))
            adj = shaping - old_linear
            self.rew_buf = torch.where(is_graspable, self.rew_buf, self.rew_buf + adj)
        self.prev_target_dist = d

        # out-of-workspace: penalty + terminal failure
        viol = (self._oow_of(self.block_state[:, :, 0:2]) & ~self.oow_exempt).any(dim=1)
        viol = viol & (~is_graspable)
        if viol.any():
            self.rew_buf = torch.where(viol, self.rew_buf + self.rb_oow_penalty, self.rew_buf)
            if self.rb_terminate_oow:
                self.reset_buf = torch.where(viol, torch.ones_like(self.reset_buf),
                                             self.reset_buf)
        if self.rb_suppress_resets:
            self.reset_buf = torch.zeros_like(self.reset_buf)
