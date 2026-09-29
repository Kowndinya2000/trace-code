"""MoreOpenLoop — MoreTeacher variant for the open-loop digital-twin pipeline.

Parented on MoreTeacher (not More) so the twin runs the SAME MDP the policy was
trained on: 94-D egocentric observation, absolute-setpoint controller (K=2,
controlFrequencyInv=4), 4 cm primitive and 20000/400 drives. Under `More` the
twin silently used a 90-D observation and the legacy incremental controller at
2 cm, so a teacher checkpoint could not even load, and the solved trajectory
would not have been the one the policy intended.

Two roles, both used by isaacgymenvs/open_loop/:

  SOLVE  (solve_in_twin.py): the trained PPO policy is rolled out in the
         twin; every commanded waypoint and every actual EEF position is
         recorded PER ENV, so a whole suite of scenes can be solved in one
         batched sim (num_envs = #scenes) or a single perceived real scene
         (num_envs = 1).

  REPLAY (replay_in_twin.py): the policy is OUT of the loop. Each env is
         driven through a fixed list of absolute EEF targets (what the real
         robot will do with moveL) — the EEF is servoed to each target until
         it is within `replayTol` (or a tick budget expires), then advances.
         Combined with `perturb()` this is the lockstep certifier for a
         stored trajectory: N replicas of the perceived scene, identical arm
         motion, different object poses / friction.

Common hooks: `freeze(env_ids)` (hold pose, never auto-reset — needed for
batched first-episode-only solving), `compute_final_grasp(env_idx, colors)`
(tiled 16-rotation GPN on one env's render).
"""
import math

import numpy as np
import torch
from omegaconf import OmegaConf
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from isaacgymenvs.tasks.more_teacher import MoreTeacher
from isaacgymenvs.tasks.more_robust import WS_X, WS_Y   # noqa: F401  (canonical OOW box, kept for reference)


def _yaw_of(q):
    return torch.atan2(2.0 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                       1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2))


class MoreOpenLoop(MoreTeacher):
    def __init__(self, cfg, test, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, test, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)
        E, dev = self.num_envs, self.device
        self.frozen = torch.zeros(E, dtype=torch.bool, device=dev)
        self.replay_active = torch.zeros(E, dtype=torch.bool, device=dev)
        self.replay_done = torch.zeros(E, dtype=torch.bool, device=dev)
        self.replay_targets = torch.zeros((E, 1, 3), device=dev)
        self.replay_len = torch.zeros(E, dtype=torch.long, device=dev)
        self.replay_idx = torch.zeros(E, dtype=torch.long, device=dev)
        self.replay_ticks = torch.zeros(E, dtype=torch.long, device=dev)
        ol = self.cfg["env"].get("openLoop", {})
        self.replay_tol = float(ol.get("replayTol", 0.002))        # m, "waypoint reached"
        self.replay_max_ticks = int(ol.get("replayMaxTicks", 60))  # per waypoint, anti-stall
        # replay servo uses the SAME controller as the solve (absolute setpoint
        # hold + orientation hold + maxStepM); a waypoint plays the role a plan
        # phase plays for the policy, so the setpoint is held until the waypoint
        # index advances. Previously this branch was hardcoded to the legacy
        # incremental form (10 mm clamp, orn_err = 0), so the arm was driven a
        # different way on replay than during the solve.
        self.replay_hold_target = torch.zeros((E, 6), device=dev)
        self.replay_last_idx = torch.full((E,), -1, dtype=torch.long, device=dev)
        self.timed_replay = torch.zeros(E, dtype=torch.bool, device=dev)
        self.timed_modes = torch.zeros(E, dtype=torch.long, device=dev)
        self._timed_enabled = False
        self.timed_commands = None
        self.timed_eef = None
        self.timed_start_eef = None
        self.record_physics = False
        self.record_replay_physics = False
        self._physics_live = torch.zeros(E, dtype=torch.bool, device=dev)
        self.pristine_block_state = self.default_block_state.clone()
        self.clear_recording()

    # ------------------------------------------------------------ recording
    def clear_recording(self):
        E = self.num_envs
        self.recording = {"start_eef": [None] * E,
                          "steps": [[] for _ in range(E)],
                          "waypoints": [[] for _ in range(E)],
                          "physics": [None] * E}

    def recording_of(self, i):
        return {"start_eef": self.recording["start_eef"][i],
                "steps": self.recording["steps"][i],
                "waypoints": self.recording["waypoints"][i],
                "physics": self.recording["physics"][i]}

    def enable_physics_recording(self, enabled=True, include_replay=False):
        """Opt in for trajectory exports/audits; training collectors pay no cost."""
        self.record_physics = bool(enabled)
        self.record_replay_physics = bool(include_replay)

    def _snapshot_physics(self):
        """Read fresh sensors without changing the teacher's cached control state.

        Its DOF tensors are views into the refresh buffer. Leaving a refreshed
        buffer in place changes phase-two IK and therefore the trained MDP.
        """
        root = gymtorch.wrap_tensor(self.actor_root_state_tensor)
        old_rb, old_dof, old_root = self.rb_states.clone(), self.dof_state.clone(), root.clone()
        self.gym.fetch_results(self.sim, True)
        try:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            return {"eef_state": self.rb_states[self.gripper_idxs.long()].clone().cpu().tolist(),
                    "joint_pos": self.ur5e_dof_pos.clone().cpu().tolist(),
                    "joint_vel": self.ur5e_dof_vel.clone().cpu().tolist(),
                    "block_state": self.block_state.clone().cpu().tolist(),
                    "robot_root_state": root.view(self.num_envs, -1, 13)[:, 1].clone().cpu().tolist()}
        finally:
            self.rb_states.copy_(old_rb)
            self.dof_state.copy_(old_dof)
            root.copy_(old_root)

    def physics_config(self):
        return OmegaConf.to_container(OmegaConf.create({"sim": self.cfg["sim"], "robot": self.cfg["robot"],
                              "controller": self.cfg["env"].get("controller", {}),
                              "asset": self.cfg["env"]["asset"],
                              "control_frequency_inv": self.control_freq_inv}), resolve=True)

    def _begin_physics_tick(self):
        if not self.record_physics:
            return
        self._physics_live[:] = ~self.frozen & (~self.replay_active |
            (self.record_replay_physics & ~self.replay_done))
        new = [i for i in self._physics_live.nonzero(as_tuple=False).flatten().tolist()
               if self.recording["physics"][i] is None]
        if new:
            states = self._snapshot_physics()
            for i in new:
                initial = {k: v[i] for k, v in states.items()}
                initial["time_s"] = 0.0
                self.recording["physics"][i] = {
                    "dt": float(self.cfg["sim"]["dt"]), "frame": "sim",
                    "quaternion_order": "xyzw", "config": self.physics_config(),
                    "initial": initial, "samples": []}

    def _after_simulate(self):
        # Called after each gym.simulate, before the next tick changes targets.
        if not self.record_physics and not self._timed_enabled:
            return
        if self.record_physics and self._physics_live.any():
            states = self._snapshot_physics()
            commands = self.ur5e_dof_targets.detach().cpu().tolist()
            for i in self._physics_live.nonzero(as_tuple=False).flatten().tolist():
                trace = self.recording["physics"][i]
                sample = {k: v[i] for k, v in states.items()}
                sample["time_s"] = (len(trace["samples"]) + 1) * trace["dt"]
                sample["joint_targets"] = commands[i]
                trace["samples"].append(sample)
        if self._timed_enabled:
            running = self.timed_replay & ~self.replay_done & ~self.frozen
            self.replay_idx[running] += 1
            self.replay_done |= running & (self.replay_idx >= self.replay_len)

    # ------------------------------------------------------------ utilities
    # _oow_of is inherited from MoreRobust so robust.wsSide remains authoritative.

    def oow_violation(self):
        """(E,) bool: a block that STARTED inside the workspace (per the reset
        scene; boundary-padded dummies are exempt) is now outside it."""
        return (self._oow_of(self.block_state[:, :, :2]) &
                ~self._oow_of(self.default_block_state[:, :, :2])).any(dim=1)

    def freeze(self, env_ids):
        self.frozen[env_ids] = True

    def settle(self, steps=30):
        """Settle-freeze (gen-v1 recipe): hold the arm, step physics so
        interpenetrating (perceived) blocks pop apart, then freeze the settled
        poses as the new default/pristine scene and reset. Returns (E,) max
        block xy displacement in metres — large values flag a twin whose
        perception was inconsistent (objects overlapping).

        CALL THIS ONCE, at startup. It REDEFINES pristine_block_state to be the
        current block state. Calling it once per collection episode would make
        each episode start from the previous episode's final state and cause
        cumulative scene drift."""
        before = self.block_state[:, :, :2].clone()
        was_frozen = self.frozen.clone()
        self.frozen[:] = True
        zero = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for _ in range(int(steps)):
            self.step(zero)
        self.frozen = was_frozen
        disp = (self.block_state[:, :, :2] - before).norm(dim=-1).amax(dim=1)
        st = self.block_state.clone()
        st[:, :, 7:] = 0.0                      # zero velocities
        self.default_block_state[:] = st
        self.pristine_block_state = st.clone()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.clear_recording()
        return disp

    def perturb(self, env_ids, pos_noise=0.0, yaw_noise_deg=0.0, friction_range=None,
                mass_scale_range=None, seed=0):
        """Perturb the DEFAULT (reset) scene of env_ids around the pristine
        scene file poses: uniform xy/yaw noise + per-actor friction / mass DR.
        Call reset_idx(env_ids) afterwards."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        st = self.pristine_block_state[env_ids].clone()          # (N, B, 13)
        if pos_noise > 0 or yaw_noise_deg > 0:
            n = st.shape[:2]
            st[:, :, 0] += (torch.rand(n, generator=g, device=self.device) * 2 - 1) * pos_noise
            st[:, :, 1] += (torch.rand(n, generator=g, device=self.device) * 2 - 1) * pos_noise
            yaw = _yaw_of(st[:, :, 3:7]) + (torch.rand(n, generator=g, device=self.device) * 2 - 1) \
                * math.radians(yaw_noise_deg)
            st[:, :, 3] = 0.0
            st[:, :, 4] = 0.0
            st[:, :, 5] = torch.sin(yaw / 2)
            st[:, :, 6] = torch.cos(yaw / 2)
        self.default_block_state[env_ids] = st
        if friction_range or mass_scale_range:
            rng = np.random.default_rng(int(seed))
            for i in env_ids.tolist():
                env_ptr = self.envs[i]
                for b in range(self.num_objects):
                    actor = self.gym.get_actor_handle(env_ptr, 2 + b)   # 0 workspace, 1 robot
                    if friction_range:
                        sp = self.gym.get_actor_rigid_shape_properties(env_ptr, actor)
                        for s in sp:
                            s.friction = float(rng.uniform(*friction_range))
                        self.gym.set_actor_rigid_shape_properties(env_ptr, actor, sp)
                    if mass_scale_range:
                        bp = self.gym.get_actor_rigid_body_properties(env_ptr, actor)
                        for body in bp:
                            body.mass *= float(rng.uniform(*mass_scale_range))
                        self.gym.set_actor_rigid_body_properties(env_ptr, actor, bp,
                                                                 recomputeInertia=True)

    def _densify(self, path):
        """Sub-sample so no segment exceeds one clamp step (controller.maxStepM).

        The absolute-hold controller commands ONE setpoint per target, capped at
        maxStepM. A sparse path (e.g. the Douglas-Peucker `executed` moveL list)
        has segments far longer than that, so the arm advances one clamp step,
        holds, and never reaches the 2 mm advance tolerance — it stalls until the
        tick budget forces it on. Densifying makes every replay target reachable
        in one hold, exactly as a waypoint phase is during the solve. The STORED
        trajectory is untouched: this only affects how the twin servos it.
        """
        step = max(self.ctrl_max_step, 1e-4)
        out = [list(path[0])]
        for a, b in zip(path[:-1], path[1:]):
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            d = float(np.linalg.norm(b - a))
            n = max(int(np.ceil(d / step)), 1)
            for k in range(1, n + 1):
                out.append(list(a + (b - a) * (k / n)))
        return out

    def set_replay(self, paths):
        """paths: list (len num_envs) of [[x,y,z], ...] absolute sim targets or
        None (env keeps being policy-driven)."""
        self.timed_replay[:] = False
        self._timed_enabled = False
        paths = [self._densify(p) if p else p for p in paths]
        L = max([len(p) for p in paths if p] + [1])
        tgt = torch.zeros((self.num_envs, L, 3), device=self.device)
        self.replay_active[:] = False
        self.replay_len[:] = 0
        for i, p in enumerate(paths):
            if p:
                tgt[i, :len(p)] = torch.as_tensor(p, dtype=torch.float32, device=self.device)
                self.replay_len[i] = len(p)
                self.replay_active[i] = True
        self.replay_targets = tgt
        self.replay_idx[:] = 0
        self.replay_ticks[:] = 0
        self.replay_done[:] = False
        self.replay_last_idx[:] = -1

    def set_timed_replay(self, trajectories, mode="joint"):
        """One command per physics tick; no point thinning or arrival gating.

        joint is the exact simulator baseline. cartesian is the direct pose
        tracking diagnostic. cartesian_feedforward combines the original
        motor commands with EEF tracking correction at the current timestamp,
        preserving the motor response as well as the desired pose sequence.
        None entries remain policy-driven. Initial states are not overwritten.
        """
        from isaacgymenvs.open_loop.trajectory import validate_physics_trace
        modes = [mode] * self.num_envs if isinstance(mode, str) else list(mode)
        mode_ids = {"joint": 1, "cartesian": 2, "cartesian_feedforward": 3}
        if (len(trajectories) != self.num_envs or len(modes) != self.num_envs
                or any(m not in mode_ids for m in modes)):
            raise ValueError("Expected one trajectory per environment and a timed replay mode")
        for t in trajectories:
            if t is None:
                continue
            if t.physics is None:
                raise ValueError("Timed replay needs a physics trace; regenerate the legacy trajectory")
            validate_physics_trace(t.physics)
            if len(t.physics["samples"]) % self.control_freq_inv:
                raise ValueError("Timed trace must end at a replay RL-step boundary")
            if abs(t.physics["dt"] - float(self.cfg["sim"]["dt"])) > 1e-9:
                raise ValueError("Replay simulator dt differs from recorded physics dt")
            if len(t.physics["initial"]["joint_pos"]) != self.ur5e_dof_pos.shape[1]:
                raise ValueError("Recorded robot DOF count differs from replay robot")
            config = t.physics.get("config")
            if config:
                current = self.physics_config()
                for key in ("robot", "asset"):
                    if config[key] != current[key]:
                        raise ValueError("Recorded " + key + " configuration differs from replay")
                if config["sim"]["substeps"] != current["sim"]["substeps"]:
                    raise ValueError("Recorded PhysX substeps differ from replay")
        self.set_replay([None] * self.num_envs)
        L = max([len(t.physics["samples"]) for t in trajectories if t is not None] + [1])
        self.timed_commands = torch.zeros((self.num_envs, L, self.ur5e_dof_pos.shape[1]), device=self.device)
        self.timed_eef = torch.zeros((self.num_envs, L, 13), device=self.device)
        self.timed_start_eef = torch.zeros((self.num_envs, L, 13), device=self.device)
        self.timed_modes[:] = torch.tensor([mode_ids[m] for m in modes], device=self.device)
        self._timed_enabled = any(t is not None for t in trajectories)
        for i, t in enumerate(trajectories):
            if t is None:
                continue
            samples = t.physics["samples"]
            self.timed_commands[i, :len(samples)] = torch.tensor([s["joint_targets"] for s in samples], device=self.device)
            self.timed_eef[i, :len(samples)] = torch.tensor([s["eef_state"] for s in samples], device=self.device)
            self.timed_start_eef[i, :len(samples)] = torch.tensor(
                [s["eef_state"] for s in [t.physics["initial"]] + samples[:-1]], device=self.device)
            self.replay_len[i] = len(samples)
            self.replay_active[i] = self.timed_replay[i] = True

    # ------------------------------------------------------------ step hooks
    def pre_physics_step(self, actions):
        actions = actions.reshape(self.num_envs)   # rl_games squeezes a 1-env batch to 0-d
        held = self.replay_active | self.frozen
        was_idle = ~self.plan_active
        self.plan_active[held] = True          # More must not start plans on held envs
        super().pre_physics_step(actions)
        started = was_idle & self.plan_active & ~held
        if started.any():
            best = actions.long()
            gp0 = self.gripper_pos.cpu().numpy()
            wp1, wp2 = self.wp1_pos.cpu().numpy(), self.wp2_pos.cpu().numpy()
            t = self.progress_buf.cpu().numpy()
            for i in started.nonzero(as_tuple=False).squeeze(-1).tolist():
                if self.recording["start_eef"][i] is None:
                    self.recording["start_eef"][i] = gp0[i].tolist()
                self.recording["waypoints"][i].append({
                    "t": int(t[i]), "action": int(best[i]),
                    "wp1": wp1[i].tolist(), "wp2": wp2[i].tolist()})

    def _apply_plan_target_one_substep(self):
        self._begin_physics_tick()
        super()._apply_plan_target_one_substep()
        r = self.replay_active & ~self.replay_done & ~self.timed_replay
        if r.any():
            ar = torch.arange(self.num_envs, device=self.device)
            tgt = self.replay_targets[ar, self.replay_idx.clamp(max=self.replay_targets.shape[1] - 1)]
            err = tgt - self.gripper_pos
            reached = (err.norm(dim=1) < self.replay_tol) | (self.replay_ticks >= self.replay_max_ticks)
            last = self.replay_idx >= self.replay_len - 1
            self.replay_done |= r & reached & last
            adv = r & reached & ~last
            self.replay_idx[adv] += 1
            self.replay_ticks[adv] = 0
            self.replay_ticks[r & ~adv] += 1
            tgt = self.replay_targets[ar, self.replay_idx.clamp(max=self.replay_targets.shape[1] - 1)]
            pos_err = tgt - self.gripper_pos
            n = torch.norm(pos_err, dim=1, keepdim=True) + 1e-8
            pos_err = pos_err * torch.clamp(self.ctrl_max_step / n, max=1.0)   # maxStepM, as in the solve
            if self.ctrl_hold_orn:                                             # holdOrientation, as in the solve
                if self.ctrl_orn_ref is None:
                    self.ctrl_orn_ref = self.gripper_rot.clone()
                q_err = quat_mul(self.ctrl_orn_ref, quat_conjugate(self.gripper_rot))
                orn_err = q_err[:, 0:3] * torch.sign(q_err[:, 3]).unsqueeze(-1)
            else:
                orn_err = torch.zeros_like(pos_err)
            dpose = torch.cat([pos_err, orn_err], dim=1).unsqueeze(-1)
            u = self.control_ik(dpose)
            if self.ctrl_mode in ("absolute", "reach_gated"):
                # hold ONE setpoint per waypoint instead of re-deriving an
                # increment from the pose just reached (the exponential approach
                # that never converges — see SWEEP "controller settled")
                fresh = r & (self.replay_idx != self.replay_last_idx)
                if torch.any(fresh):
                    self.replay_hold_target[fresh] = self.ur5e_dof_pos[fresh, :6] + u[fresh]
                    self.replay_last_idx[fresh] = self.replay_idx[fresh]
                self.ur5e_dof_targets[r, :6] = self.replay_hold_target[r]
            else:
                self.ur5e_dof_targets[r, :6] = self.ur5e_dof_pos[r, :6] + u[r]
            self.ur5e_dof_targets[r, 6:] = 0.0
        timed = self.timed_replay & ~self.replay_done & ~self.frozen
        if self._timed_enabled and timed.any():
            ar = torch.arange(self.num_envs, device=self.device)
            idx = self.replay_idx.clamp(max=self.timed_commands.shape[1] - 1)
            joint = timed & (self.timed_modes == 1)
            direct = timed & (self.timed_modes == 2)
            feedforward = timed & (self.timed_modes == 3)
            if joint.any():
                self.ur5e_dof_targets[joint] = self.timed_commands[ar, idx][joint]
            if feedforward.any():
                self._apply_timed_cartesian(feedforward, self.timed_start_eef[ar, idx],
                                            self.timed_commands[ar, idx])
            if direct.any():
                self._apply_timed_cartesian(direct, self.timed_eef[ar, idx])
        hold = self.frozen | (self.replay_active & self.replay_done)
        if hold.any():
            self.ur5e_dof_targets[hold, :6] = self.ur5e_dof_pos[hold, :6]
            self.ur5e_dof_targets[hold, 6:] = 0.0
        if r.any() or (self._timed_enabled and timed.any()) or hold.any():
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur5e_dof_targets))

    def _apply_timed_cartesian(self, mask, target, feedforward=None):
        # Refresh independent buffers and restore DOFs afterward so a policy
        # sharing this simulator retains the same cached inputs as before.
        old_rb, old_dof, old_j = self.rb_states.clone(), self.dof_state.clone(), self.j_eef.clone()
        self.gym.fetch_results(self.sim, True)
        try:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            actual = self.rb_states[self.gripper_idxs.long()]
            qerr = quat_mul(target[:, 3:7], quat_conjugate(actual[:, 3:7]))
            error = torch.cat([target[:, :3] - actual[:, :3],
                               qerr[:, :3] * torch.sign(qerr[:, 3:4])], dim=1)
            dq = self.control_ik(error.unsqueeze(-1))
            if feedforward is None:
                self.ur5e_dof_targets[mask, :6] = (self.ur5e_dof_pos[:, :6] + dq)[mask]
                self.ur5e_dof_targets[mask, 6:] = 0.0
            else:
                self.ur5e_dof_targets[mask] = feedforward[mask]
                self.ur5e_dof_targets[mask, :6] += dq[mask]
        finally:
            self.rb_states.copy_(old_rb)
            self.dof_state.copy_(old_dof)
            self.j_eef.copy_(old_j)

    def post_physics_step(self):
        held = self.replay_active | self.frozen
        self.reset_buf[held] = 0               # held envs never auto-reset
        super().post_physics_step()
        rec = ~held
        if rec.any():
            gp = self.gripper_pos.cpu().numpy()
            q = self.grasp_q_parallel_values.cpu().numpy()
            t = self.progress_buf.cpu().numpy()
            # per-step object centres, SIM object order (target first). The
            # student's plan block (student_obs.encode_plan, PLAN_PRED_OBJ) is
            # "where the twin predicted every object would be at this step";
            # the hardware loop reads it from the saved trajectory, so it has
            # to be recorded here, not only inside collect_student_data.
            cen = self.blocks_rect_rotated[:, :, 0, :].cpu().numpy()      # (O, E, 2)
            for i in rec.nonzero(as_tuple=False).squeeze(-1).tolist():
                wps = self.recording["waypoints"][i]
                self.recording["steps"][i].append({
                    "t": int(t[i]), "eef": gp[i].tolist(),
                    "action": wps[-1]["action"] if wps else -1, "grasp_q": float(q[i]),
                    "obj_xy": cen[:, i, :].tolist()})

    # ------------------------------------------------------------ final grasp
    def render_offset(self, i):
        """(d_row, d_col) between the target's render-mask centroid and
        frames.sim_to_pix(target xy) — a render/state consistency probe."""
        from isaacgymenvs.open_loop import frames
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            segm = self.cam_segm_tensors[i].cpu().numpy()
        finally:
            self.gym.end_access_image_tensors(self.sim)
        r, c = np.nonzero(segm == 255)
        px, py = frames.sim_to_pix(*self.target_pose(i)[:2])
        return (float(r.mean() - px), float(c.mean() - py)) if len(r) else (float("nan"), float("nan"))

    def target_pose(self, i):
        """(x, y, yaw) of the target block in env i (sim frame)."""
        st = self.block_state[i, 0]
        return float(st[0]), float(st[1]), float(_yaw_of(st[3:7]))

    @torch.no_grad()
    def compute_final_grasp(self, env_idx, block_colors):
        """Tiled 16-rotation GPN (rl_policy.py convention) on env `env_idx`.

        block_colors: (11, 3) floats in [0, 1], scene-file order (0 = target).
        The camera color stream is not captured by More, so RGB is composed
        flat from the segmentation map + block colors (training renders are
        flat-colored too). Segm ids are remapped from More's [255, 50..140]
        to the closed-loop [255, 60..150] scheme the x16 post-check expects.
        Returns {"q", "rotation_idx", "px", "py"} with (px, py) = (row, col)
        in the shared heightmap convention (frames.pix_to_sim).
        """
        from isaacgymenvs.open_loop import frames
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            depth = self.cam_depth_tensors[env_idx].clone()
            segm = self.cam_segm_tensors[env_idx].clone()
        finally:
            self.gym.end_access_image_tensors(self.sim)
        depth = torch.where(torch.isneginf(depth), torch.zeros_like(depth), depth)
        depth = (depth - depth.min()).cpu().numpy().astype(np.float32)   # height above table (m)
        segm = segm.cpu().numpy().astype(np.int32)

        # render convention check (row = sim x, col = sim y, workspace-corner
        # origin): target mask centroid vs. the target's sim position
        rows, cols = np.nonzero(segm == 255)
        assert len(rows) > 0, "target not visible in the final render"
        tx, ty, _ = self.target_pose(env_idx)
        # This is a CAMERA render, so validate in the canvas frame. The camera is
        # 320 px (videoLog.cameraRes, inherited from MoreTeacher); sim_to_pix is
        # the legacy 224 workspace-relative heightmap, which sits CANVAS_WS_OFFSET
        # = 48 px inside the canvas on both axes. The two frames coincided only
        # while this task ran the old 224 px camera.
        cpx, cpy = frames.sim_to_canvas_pix(tx, ty)
        d = abs(rows.mean() - cpx) + abs(cols.mean() - cpy)
        assert d < 6.0, (f"render/canvas axis mismatch: target centroid "
                         f"(row {rows.mean():.1f}, col {cols.mean():.1f}) vs sim_to_canvas_pix {(cpx, cpy)}")

        segm_cl = np.zeros_like(segm)
        rgb = np.zeros((*segm.shape, 3), dtype=np.float32)
        for b in range(self.num_objects):
            sid = 255 if b == 0 else 50 + 10 * (b - 1)
            m = segm == sid
            segm_cl[m] = 255 if b == 0 else 60 + 10 * (b - 1)
            rgb[m] = np.asarray(block_colors[b], dtype=np.float32) * 255.0
        # the closed-loop (rl_policy.py) GPN helper: numpy in, un-rotated best
        # pixel out — tasks/utils' variant returns only the Q value
        if not hasattr(self, "_x16_helper"):
            from utils.mtcs_utils import MCTSHelper as X16Helper
            self._x16_helper = X16Helper("logs_grasp/snapshot-post-020000.reinforcement.pth",
                                         "logs_grasp/grasp_model-89.pth", device=str(self.device))
        # the 16-rotation batch of 320x320 through the ResNet-FPN GPN wants
        # ~8.6 GB at fp32 — run it in fp16 autocast (GPU is shared with training)
        for attempt in range(2):
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    q, best, _ = self._x16_helper.get_grasp_q_parallel_x16(
                        rgb, depth, segm_cl, post_checking=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 1:
                    print(f"[MoreOpenLoop] WARNING: final grasp OOM in env {env_idx}; grasp left empty")
                    return None
        # best[1:3] are CANVAS pixels (the GPN saw the 320 px render). The
        # documented contract for px/py is the legacy workspace heightmap
        # (frames.pix_to_sim), so shift by CANVAS_WS_OFFSET; canvas coords are
        # returned alongside so no caller has to guess which frame it holds.
        return {"q": float(q), "rotation_idx": int(best[0]),
                "px": int(best[1]) - frames.CANVAS_WS_OFFSET,
                "py": int(best[2]) - frames.CANVAS_WS_OFFSET,
                "px_canvas": int(best[1]), "py_canvas": int(best[2])}
