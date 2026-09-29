"""Register project-custom rl_games networks for standalone tools.

train.py registers `token_set` inside launch_rlg_hydra, but every evaluation
tool (eval_strict, certify, measure_step_gap, diag_teacher_compare, the
open_loop scripts, the val-curve daemon) builds its own Runner and would die
with `ValueError: token_set` on any set-encoder checkpoint. Import this before
`runner.create_player()`.
"""
from rl_games.algos_torch import model_builder

try:
    from isaacgymenvs.learning.token_set_builder import TokenSetBuilder
except ImportError:                       # tools run with cwd=isaacgymenvs/
    from learning.token_set_builder import TokenSetBuilder

if "token_set" not in getattr(model_builder.NETWORK_REGISTRY, "_builders", {}):
    try:
        model_builder.register_network("token_set", lambda **kwargs: TokenSetBuilder())
    except Exception:                     # already registered: harmless
        pass
