"""MoreTeacher — Stage 1 of the teacher-student pipeline (the method description).

Privileged teacher task: MoreRobust's randomization/DR machinery + the fully
redesigned potential-based reward. The old reward jit is bypassed entirely.

Reward (main.tex §12, review T2/T6/T7 applied):
    Phi(s) = lamG * g(s) + lamC * c(s) - lamB * B(s)
      g = target grasp-network confidence in [0,1] (dense proxy; physical
          grasp+lift verification happens out-of-band, review T7)
      c = min footprint distance target vs clutter (oriented-rect corner +
          edge-midpoint point sets, clamped at cMax so Phi stays bounded)
      B = sum_i softplus(dSafe - dist(footprint_i, workspace boundary)),
          boundary-pad-exempt — gradient BEFORE violation
    r_t = gamma*Phi(s') - Phi(s) - lamStep - lamDisturb * D_t + terminals
      D_t = sum of clutter center displacements this step
      terminals: +rSuccess (g > 0.9), -rOow (footprint corner out, terminate),
                 -rOutOfView (target lost to camera), timeout at horizon.
    Loop-farming constraint (review T2): lamStep > (1-gamma) * max|Phi|.

Recurrence comes from rl_games' built-in GRU (cfg/train/MoreTeacherPPO.yaml);
the permutation-invariant set encoder is Stage 1b (custom network builder).
eval_strict.py works unchanged (reset_buf/successes semantics preserved).
"""
import torch
import torch.nn.functional as F

from isaacgymenvs.tasks.more_robust import MoreRobust


class MoreTeacher(MoreRobust):
    def __init__(self, cfg, test, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        tc = cfg["env"].get("teacher", {})
        self.t_lam_g = float(tc.get("lambdaG", 2.0))
        self.t_lam_c = float(tc.get("lambdaC", 5.0))
        self.t_lam_c3 = float(tc.get("lambdaC3", 2.0))     # 3NN mean gap term
        self.t_lam_arc = float(tc.get("lambdaArc", 1.0))   # free-arc (seam) term
        self.t_arc_radius = float(tc.get("arcRadius", 0.13))
        self.t_lam_b = float(tc.get("lambdaB", 10.0))
        self.t_d_safe = float(tc.get("dSafe", 0.03))
        self.t_c_max = float(tc.get("cMax", 0.20))
        self.t_lam_step = float(tc.get("lambdaStep", 0.05))
        self.t_lam_disturb = float(tc.get("lambdaDisturb", 0.1))
        self.t_r_success = float(tc.get("rSuccess", 10.0))
        self.t_r_oow = float(tc.get("rOow", 5.0))
        self.t_r_view = float(tc.get("rOutOfView", 2.0))
        self.t_gamma = float(tc.get("gamma", 0.99))  # keep equal to train gamma
        # free-arc start alignment: with this prob, pick the scene-symmetry
        # variant whose largest free arc around the target faces the robot's
        # home approach (fixes clamp scenes unsolvable from a fixed side);
        # remaining prob keeps random symmetry for exploration
        self.t_arc_align_prob = float(tc.get("arcAlignProb", 0.7))
        # EEF working-zone term: zero inside eefWorkRadius of the target, then a
        # linear pull-back saturating at eefSpan. Potential-based (inside Phi),
        # so it cannot be farmed and cannot change the optimum; a deadband so it
        # does not dictate WHICH side to approach from (arc/c3 do that).
        # Without it the reward has no opinion on EEF position at all, and an
        # excursion outside the workspace is absorbing (0% recovery measured).
        self.t_lam_eef = float(tc.get("lambdaEef", 3.0))
        self.t_eef_work_r = float(tc.get("eefWorkRadius", 0.15))
        self.t_eef_span = float(tc.get("eefSpan", 0.20))
        # training OOW rule: "corner" (any footprint corner outside, strict) or
        # "center" (object centre outside = the paper's eval metric)
        self.t_oow_rule = str(tc.get("oowRule", "corner"))
        assert self.t_oow_rule in ("corner", "center"), self.t_oow_rule

        super().__init__(cfg, test, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        n = self.num_envs
        self.t_prev_phi = torch.zeros(n, device=self.device)
        self.t_prev_clutter = torch.zeros(n, self.num_objects - 1, 2, device=self.device)
        self.t_needs_init = torch.ones(n, dtype=torch.bool, device=self.device)
        self.t_obs_perm = torch.stack([torch.randperm(self.num_objects - 1,
                                                      device=self.device)
                                       for _ in range(n)])

        # loop-farming guard (review T2): bound |Phi| over the pre-terminal
        # reachable set (bdist >= 0 — OOW terminates otherwise), using the
        # same scaled softplus as _phi_terms
        max_risk = float(F.softplus(torch.tensor(self.t_d_safe / 0.01))) * 0.01
        phi_bound = (self.t_lam_g + (self.t_lam_c + self.t_lam_c3) * self.t_c_max +
                     self.t_lam_arc + self.t_lam_b * self.num_objects * max_risk +
                     self.t_lam_eef * self.t_eef_span)
        min_step = (1.0 - self.t_gamma) * phi_bound
        # `violateBounds` exists ONLY to demonstrate the failure the T2/T2b
        # bounds prevent. It downgrades the guards to warnings so an ablation
        # can show reward farming / bail-out empirically instead of asserting
        # the inequality. Never set it for a run whose policy will be used.
        self.t_violate = bool(tc.get("violateBounds", False))
        if self.t_lam_step <= min_step and self.t_violate:
            print(f"[T2 VIOLATED ON PURPOSE] lambdaStep={self.t_lam_step} <= "
                  f"(1-gamma)*max|Phi|~={min_step:.4f}: closed loops can farm reward")
        else:
            assert self.t_lam_step > min_step, (
                f"lambdaStep={self.t_lam_step} must exceed (1-gamma)*max|Phi|"
                f"~={min_step:.4f} or closed loops can farm reward (review T2)")
        # T2b: a failure terminal must cost MORE than the discounted step cost
        # it lets the policy avoid, or bailing out becomes reward-optimal on
        # any scene it judges unwinnable (the bulldozer incentive, re-entering
        # through the terminal instead of the shaping).
        avoidable = self.t_lam_step * (1.0 - self.t_gamma ** self.max_episode_length) \
            / (1.0 - self.t_gamma)
        if (self.t_r_oow <= avoidable or self.t_r_view <= avoidable) and self.t_violate:
            print(f"[T2b VIOLATED ON PURPOSE] rOow={self.t_r_oow} / "
                  f"rOutOfView={self.t_r_view} vs avoidable step cost "
                  f"~={avoidable:.2f}: bailing out may be reward-optimal")
        else:
            assert self.t_r_oow > avoidable and self.t_r_view > avoidable, (
                f"rOow={self.t_r_oow} / rOutOfView={self.t_r_view} must exceed the "
                f"avoidable discounted step cost ~={avoidable:.2f}, else deliberately "
                f"failing is worth ~{avoidable - min(self.t_r_oow, self.t_r_view):+.2f}")

    # -------------------------------------------------------- start alignment
    def _pick_symmetry(self, env_ids, st, t_rand):
        """Choose per-env symmetry {0:id,1:rot180,2:mirror-y,3:mirror-x} so
        the target's largest free angular arc maps toward the home side
        (mapped arc direction ~ pi). st: (E, B, 13) pristine poses."""
        if self.t_arc_align_prob <= 0:
            return t_rand
        xy = st[:, :, 0:2]
        rel = xy[:, 1:] - xy[:, 0:1]                       # (E, B-1, 2)
        dist = rel.norm(dim=-1)
        ang = torch.atan2(rel[..., 1], rel[..., 0])        # (E, B-1)
        near = dist < self.t_arc_radius
        nearest = ang.gather(1, dist.argmin(dim=1, keepdim=True))
        ang = torch.where(near, ang, nearest.expand_as(ang))
        ang_s, _ = ang.sort(dim=1)
        gaps = torch.roll(ang_s, -1, dims=1) - ang_s
        gaps[:, -1] += 2 * torch.pi
        gi = gaps.argmax(dim=1)
        lo = ang_s.gather(1, gi.unsqueeze(1)).squeeze(1)
        phi = lo + 0.5 * gaps.gather(1, gi.unsqueeze(1)).squeeze(1)  # arc mid
        # mapped arc direction per transform; score = cos(mapped - pi)
        cands = torch.stack([phi, phi + torch.pi, -phi, torch.pi - phi], dim=1)
        score = torch.cos(cands - torch.pi)
        t_best = score.argmax(dim=1).to(t_rand.dtype)
        use = (torch.rand(len(env_ids), device=st.device) < self.t_arc_align_prob)
        few_near = near.sum(dim=1) < 2                     # open scene: any side
        return torch.where(use & ~few_near, t_best, t_rand)

    # ------------------------------------------------------------- observations
    def compute_observations(self):
        """Teacher obs v2 (94-D): egocentric object tokens + boundary features.
        - per object: 4 rotated bbox corners RELATIVE to the EEF (8) — matches
          the EEF-relative action space; clutter token ORDER is shuffled per
          episode (permutation augmentation, stands in for a set encoder)
        - global: EEF absolute (2) + EEF distance to the 4 workspace walls (4)
          (boundary awareness for the OOW objective)
        Enabled via numObservationsOverride: 94 in MoreTeacher.yaml."""
        if self.num_obs != self.num_objects * 8 + 6:
            return super().compute_observations()
        eef = self.gripper_pos[:, :2]
        rel = self.blocks_rect_rotated[:, :, 1:5, :] - eef.unsqueeze(0).unsqueeze(2)
        toks = [rel[0].reshape(self.num_envs, 8)]
        perm = self.t_obs_perm                                    # (E, O-1)
        clut = rel[1:].permute(1, 0, 2, 3).reshape(self.num_envs,
                                                   self.num_objects - 1, 8)
        clut = torch.gather(clut, 1, perm.unsqueeze(-1).expand(-1, -1, 8))
        toks.append(clut.reshape(self.num_envs, -1))
        walls = torch.stack([eef[:, 0] - self.ws_x[0], self.ws_x[1] - eef[:, 0],
                             eef[:, 1] - self.ws_y[0], self.ws_y[1] - eef[:, 1]], dim=1)
        self.obs_buf = torch.cat(toks + [eef, walls], dim=-1)
        # NaN/inf guard: a physics blow-up in one env must not poison the
        # policy logits for the whole batch (t3-set-s1 died on NaN logits)
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.obs_buf

    # ---------------------------------------------------------------- geometry
    def _footprint_points(self):
        """(n_obj, n_env, 8, 2): oriented-rect corners + edge midpoints."""
        corners = self.blocks_rect_rotated[:, :, 1:5, :]          # (O, E, 4, 2)
        mids = 0.5 * (corners + torch.roll(corners, -1, dims=2))  # (O, E, 4, 2)
        return torch.cat([corners, mids], dim=2)

    def _phi_terms(self):
        pts = self._footprint_points()                            # (O, E, 8, 2)

        # c: min point-set distance target vs each clutter object
        tgt = pts[0].unsqueeze(0)                                 # (1, E, 8, 2)
        clut = pts[1:]                                            # (O-1, E, 8, 2)
        d = torch.norm(clut.unsqueeze(3) - tgt.unsqueeze(2), dim=-1)  # (O-1,E,8,8)
        per_obj = d.amin(dim=(2, 3))                              # (O-1, E)
        c = per_obj.amin(dim=0).clamp(max=self.t_c_max)           # (E,)
        # mean gap to the 3 nearest clutter objects: smoother unlock signal
        c3 = (-torch.topk(-per_obj, k=min(3, per_obj.shape[0]), dim=0).values
              ).clamp(max=self.t_c_max).mean(dim=0)               # (E,)

        # largest free angular arc around the target (seam/opening detector):
        # widening the seam between two clamping objects grows this directly
        centers = self.blocks_rect_rotated[1:, :, 0, :]           # (O-1, E, 2)
        rel = centers - self.blocks_rect_rotated[0:1, :, 0, :]
        dist = rel.norm(dim=-1)                                   # (O-1, E)
        ang = torch.atan2(rel[..., 1], rel[..., 0])
        near = dist < self.t_arc_radius
        nearest = ang.gather(0, dist.argmin(dim=0, keepdim=True))
        ang = torch.where(near, ang, nearest.expand_as(ang))      # far -> collapse
        ang_s, _ = ang.sort(dim=0)
        gaps = torch.roll(ang_s, -1, dims=0) - ang_s
        gaps[-1] += 2 * torch.pi
        arc = gaps.max(dim=0).values
        arc = torch.where(near.sum(dim=0) < 2,
                          torch.full_like(arc, 2 * torch.pi), arc)

        # B: boundary risk over footprint corners (signed inside-distance)
        corners = self.blocks_rect_rotated[:, :, 1:5, :]          # (O, E, 4, 2)
        inside = torch.minimum(
            torch.minimum(corners[..., 0] - self.ws_x[0], self.ws_x[1] - corners[..., 0]),
            torch.minimum(corners[..., 1] - self.ws_y[0], self.ws_y[1] - corners[..., 1]))
        bdist = inside.amin(dim=2)                                # (O, E)
        risk = F.softplus((self.t_d_safe - bdist) / 0.01) * 0.01  # smooth, ~m units
        risk = risk * (~self.oow_exempt.T).float()                # exempt pads
        B = risk.sum(dim=0)                                      # (E,)

        g = self.grasp_q_parallel_values.clamp(0.0, 1.0)
        # E: EEF distance beyond the working radius around the target
        d_eef = (self.gripper_pos[:, :2] -
                 self.blocks_rect_rotated[0, :, 0, :]).norm(dim=-1)
        E = (d_eef - self.t_eef_work_r).clamp(0.0, self.t_eef_span)
        phi = (self.t_lam_g * g + self.t_lam_c * c + self.t_lam_c3 * c3 +
               self.t_lam_arc * arc / (2 * torch.pi) - self.t_lam_b * B
               - self.t_lam_eef * E)
        # raw per-term values for offline diagnostics (tools/diag_teacher_compare.py)
        self.diag_terms = {"g": g, "c": c, "c3": c3, "arc": arc, "B": B, "E": E}
        return phi, bdist

    # ----------------------------------------------------------------- reward
    def compute_reward(self):
        # full replacement: the old jit (and MoreRobust's adjustment) is bypassed
        q = self.grasp_q_parallel_values
        phi, bdist = self._phi_terms()

        clutter = self.blocks_rect_rotated[1:, :, 0, :].permute(1, 0, 2)  # (E,O-1,2)
        D = torch.norm(clutter - self.t_prev_clutter, dim=-1).sum(dim=1)  # (E,)

        shaping = self.t_gamma * phi - self.t_prev_phi
        fresh = self.t_needs_init
        shaping = torch.where(fresh, torch.zeros_like(shaping), shaping)
        D = torch.where(fresh, torch.zeros_like(D), D)

        success = q > 0.9
        # OOW: any non-exempt footprint corner outside the workspace
        if self.t_oow_rule == "center":
            centers = self.blocks_rect_rotated[:, :, 0, :]              # (O, E, 2)
            out_c = self._oow_of(centers)                              # (O, E)
            oow = (out_c & (~self.oow_exempt.T)).any(dim=0) & (~success)
        else:
            oow = ((bdist < 0.0) & (~self.oow_exempt.T)).any(dim=0) & (~success)
        out_of_view = (q == -2.0) & (~success) & (~oow)
        timeout = (self.progress_buf >= self.max_episode_length - 1)

        r = shaping - self.t_lam_step - self.t_lam_disturb * D
        r = torch.where(success, torch.full_like(r, self.t_r_success), r)
        r = torch.where(oow, r - self.t_r_oow, r)
        r = torch.where(out_of_view, r - self.t_r_view, r)

        reset = (success | oow | out_of_view | timeout).float()

        self.rew_buf = r
        self.reset_buf = reset
        self.successes = success.float()   # eval_strict reads this at reset

        self.t_prev_phi = phi
        self.t_prev_clutter = clutter.clone()
        self.t_needs_init = torch.zeros_like(self.t_needs_init)

        # ---- diagnostics -> RLGPUAlgoObserver -> tensorboard -> wandb -------
        # Any 0-dim tensor in self.extras is logged as a scalar; nested dicts are
        # flattened with "/". Per-term Phi and the reward decomposition make it
        # possible to see WHICH term is driving (or stalling) learning.
        with torch.no_grad():
            d = self.diag_terms
            self.extras["phi"] = {
                "total": phi.mean(), "g": d["g"].mean(), "c": d["c"].mean(),
                "c3": d["c3"].mean(), "arc": d["arc"].mean(),
                "B": d["B"].mean(), "E": d["E"].mean(),
            }
            self.extras["rew"] = {
                "total": r.mean(),
                "shaping": shaping.mean(),
                "step_cost": torch.as_tensor(self.t_lam_step, device=r.device).float(),
                "disturb": (self.t_lam_disturb * D).mean(),
                "clutter_disturb_m": D.mean(),
            }
            self.extras["term"] = {
                "success": success.float().mean(),
                "oow": oow.float().mean(),
                "out_of_view": out_of_view.float().mean(),
                "timeout": timeout.float().mean(),
                "reset_rate": reset.mean(),
            }
            eef = self.gripper_pos[:, :2]
            tgt = self.blocks_rect_rotated[0, :, 0, :]
            outside = ((eef[:, 0] < self.ws_x[0]) | (eef[:, 0] > self.ws_x[1]) |
                       (eef[:, 1] < self.ws_y[0]) | (eef[:, 1] > self.ws_y[1]))
            self.extras["eef"] = {
                "dist_to_target_m": (eef - tgt).norm(dim=-1).mean(),
                "boundary_contact": outside.float().mean(),
                "min_bdist_m": bdist.min(dim=0).values.mean(),
            }
            fin = reset.bool()
            if fin.any():
                self.extras["ep"] = {"len_at_reset": self.progress_buf[fin].float().mean()}


    def reset_idx(self, env_ids, from_where="init"):
        super().reset_idx(env_ids, from_where)
        if hasattr(self, "t_needs_init"):
            ids = env_ids.to(dtype=torch.long)
            self.t_needs_init[ids] = True
            for i in ids.tolist():      # fresh clutter permutation per episode
                self.t_obs_perm[i] = torch.randperm(self.num_objects - 1,
                                                    device=self.device)
