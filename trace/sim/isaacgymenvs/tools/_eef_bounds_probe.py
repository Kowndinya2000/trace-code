"""Does the EEF leave the workspace, and does it get stuck at a joint limit?
Nothing clamps the gripper to the workspace (more.py:1254-55 is commented out)
and the collected joint limits (more.py:453-464) are never applied to the DOF
targets, so the PD can command into a hard stop. Quantifies both."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa: F401
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv
    steps = int(cfg.get("gap_steps", 450)); tag = str(cfg.get("gap_tag", "eef"))
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    set_np_formatting(); cfg.seed = set_seed(42, False)

    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
                                 cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                                 cfg.headless, cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg, **kw)
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver()); runner.load(omegaconf_to_dict(cfg.train)); runner.reset()
    player = runner.create_player(); player.restore(cfg.checkpoint)
    env = player.env; obses = player.env_reset(env); player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()

    E = env.reset_buf.shape[0]; dev = env.reset_buf.device
    lo = env.ur5e_dof_lower_limits[:6]; hi = env.ur5e_dof_upper_limits[:6]
    done = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_out = torch.zeros(E, dtype=torch.bool, device=dev)
    out_steps = torch.zeros(E, device=dev); live_steps = torch.zeros(E, device=dev)
    recovered = torch.zeros(E, dtype=torch.bool, device=dev)
    at_limit_while_out = torch.zeros(E, dtype=torch.bool, device=dev)
    margin_min = torch.full((E,), 1e9, device=dev)
    depth_all = []                      # how far OUTSIDE, in metres (0 when inside)

    for _ in range(steps):
        a = player.get_action(obses, is_deterministic=True)
        a = a.squeeze(-1) if a.dim() > 1 else a
        obses, _, _, _ = player.env_step(env, a)
        eef = env.gripper_pos[:, :2]
        out = ((eef[:, 0] < env.ws_x[0]) | (eef[:, 0] > env.ws_x[1]) |
               (eef[:, 1] < env.ws_y[0]) | (eef[:, 1] > env.ws_y[1])) & ~done
        dx = torch.maximum(env.ws_x[0] - eef[:, 0], eef[:, 0] - env.ws_x[1]).clamp(min=0)
        dy = torch.maximum(env.ws_y[0] - eef[:, 1], eef[:, 1] - env.ws_y[1]).clamp(min=0)
        depth_all.append(torch.where(done, torch.zeros_like(dx), torch.maximum(dx, dy)))
        q = env.ur5e_dof_pos[:, :6]
        m = torch.minimum((q - lo).abs().amin(dim=1), (hi - q).abs().amin(dim=1))  # rad to nearest stop
        margin_min = torch.where(~done, torch.minimum(margin_min, m), margin_min)
        at_limit_while_out |= out & (m < 0.05)
        recovered |= ever_out & ~out & ~done
        ever_out |= out
        out_steps += out.float(); live_steps += (~done).float()
        done |= env.reset_buf.bool()

    out = {"tag": tag, "envs": E, "steps": steps,
           "mode": str(cfg.task.env.controller.mode), "cfi": int(cfg.task.env.controlFrequencyInv),
           "envs_eef_left_workspace_pct": 100.0 * float(ever_out.float().mean()),
           "steps_with_eef_outside_pct": 100.0 * float(out_steps.sum() / live_steps.sum().clamp(min=1)),
           "of_those_recovered_pct": (100.0 * float(recovered.sum()) / max(1, int(ever_out.sum()))),
           "envs_at_joint_limit_while_outside_pct": 100.0 * float(at_limit_while_out.float().mean()),
           "min_rad_to_joint_stop_median": float(margin_min.median()),
           "excursion_depth_mm_max": float(torch.stack(depth_all).max()) * 1000,
           "excursion_depth_mm_p95": float(torch.stack(depth_all)[torch.stack(depth_all) > 0].quantile(0.95)) * 1000
                                     if bool((torch.stack(depth_all) > 0).any()) else 0.0,
           "excursion_depth_mm_median_when_out": float(torch.stack(depth_all)[torch.stack(depth_all) > 0].median()) * 1000
                                     if bool((torch.stack(depth_all) > 0).any()) else 0.0}
    os.makedirs("tools/qa_out", exist_ok=True)
    json.dump(out, open(f"tools/qa_out/eef_bounds_{tag}.json", "w"), indent=1)
    print("EEF_BOUNDS " + json.dumps(out))


if __name__ == "__main__":
    main()
