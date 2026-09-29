"""Open-loop digital-twin pipeline (post-IROS26 direction, Aug 2026).

Perceive the real tabletop ONCE with a single external RGB-D camera, build
the scene in Isaac Gym, solve it entirely in simulation with the trained PPO
push policy + grasp network, record the full end-effector trajectory, and
replay it on the real UR5e without online feedback.

Stages (one script per stage, see README.md):
  1. perceive_scene.py    real RGB-D -> twin scene file (test-case format)
  2. solve_in_twin.py     PPO rollout in sim -> trajectory JSON
  3. execute_trajectory.py  trajectory JSON -> UR5e (open loop)

Shared pieces: frames.py (workspace/frame conversions, single source of
truth), trajectory.py (the trajectory file format).
"""
