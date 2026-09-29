"""Where does solve_in_twin's rollout actually spend its time?

Wraps the three candidates in the per-step path -- camera rendering, the tiled
grasp network, and the physics/policy remainder -- with CUDA-synchronised
timers. Run from isaacgymenvs/.
"""
import os, sys, time, collections
sys.path.insert(0, os.getcwd())
import isaacgym  # noqa
import tools._net_compat  # noqa
import tools._ckpt_compat  # noqa
import hydra, torch
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

T = collections.defaultdict(float)
N = collections.defaultdict(int)


def timed(key, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); T[key] += time.perf_counter() - t; N[key] += 1
        return r
    return wrap


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    import isaacgymenvs
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv
    set_np_formatting(); cfg.seed = set_seed(42, False)

    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.test, cfg.task.env.numEnvs,
                                 cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                                 cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                 cfg.force_render, cfg, **kw)
    vecenv.register("RLGPU", lambda name, num, **kw: RLGPUEnv(name, num, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": thunk})
    runner = Runner(RLGPUAlgoObserver()); runner.load(omegaconf_to_dict(cfg.train)); runner.reset()
    player = runner.create_player(); player.restore(to_absolute_path(cfg.checkpoint))
    env = player.env

    # the pybind Gym object's attributes are read-only, so wrap it in a proxy
    WATCH = {"render_all_camera_sensors": "camera render",
             "simulate": "physics simulate",
             "step_graphics": "step graphics",
             "fetch_results": "fetch results"}

    class GymProxy:
        def __init__(self, g): object.__setattr__(self, "_g", g)
        def __getattr__(self, n):
            a = getattr(object.__getattribute__(self, "_g"), n)
            return timed(WATCH[n], a) if n in WATCH and callable(a) else a

    env.gym = GymProxy(env.gym)
    env.mcts_helper.grasp_prob_B = timed("grasp network", env.mcts_helper.grasp_prob_B)

    obses = player.env_reset(env); player.get_batch_size(obses, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()

    T.clear(); N.clear()
    torch.cuda.synchronize(); t = time.perf_counter()
    env.settle(int(cfg.get("settle_steps", 30)))
    torch.cuda.synchronize(); t_settle = time.perf_counter() - t
    settle_T, settle_N = dict(T), dict(N)
    obses = player.env_reset(env); env.clear_recording()
    T.clear(); N.clear()          # settle's calls must not land in the rollout

    steps = int(cfg.get("profile_steps", 21))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        torch.cuda.synchronize(); a = time.perf_counter()
        act = player.get_action(obses, is_deterministic=True)
        torch.cuda.synchronize()
        T["policy inference"] += time.perf_counter() - a; N["policy inference"] += 1
        obses, _, _, _ = player.env_step(env, act)
    torch.cuda.synchronize(); total = time.perf_counter() - t0

    KEYS = ("camera render", "grasp network", "physics simulate", "step graphics",
            "fetch results", "policy inference")
    print("\n=== rollout profile ===")
    print(f"  settle({cfg.get('settle_steps', 30)} steps)".ljust(34) + f"{t_settle:6.2f} s")
    for k in KEYS:
        if settle_T.get(k):
            print(f"    {k:30s}{settle_T[k]:6.2f} s  {100 * settle_T[k] / t_settle:5.1f}%  "
                  f"({settle_N[k]} calls)")
    print(f"  rollout, {steps} env steps".ljust(34) + f"{total:6.2f} s"
          f"   ({1000 * total / steps:.0f} ms/step)")
    acc = 0.0
    for k in KEYS:
        print(f"    {k:30s}{T[k]:6.2f} s  {100 * T[k] / total:5.1f}%  "
              f"({N[k]} calls, {1000 * T[k] / max(N[k], 1):.0f} ms each)")
        acc += T[k]
    print(f"    {'physics + everything else':30s}{total - acc:6.2f} s  "
          f"{100 * (total - acc) / total:5.1f}%")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
