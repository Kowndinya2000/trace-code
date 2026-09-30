# Plan-Conditioned Imitation for Robust Object Retrieval under Self-Occlusion in Dense Clutter

**TRACE** — *Teacher Rollouts for Adaptive Closed-loop Execution* — is the method introduced in that paper.
**Kowndinya Boyalakuntla<sup>1</sup>, Ajinkya Pawar<sup>2</sup>, Abdeslam Boularias<sup>1</sup>, Jingjin Yu<sup>1</sup>**<br>
<sup>1</sup> Rutgers University<br>
<sup>2</sup> Indian Institute of Technology Bombay

This repository is the reference implementation, with the data and evaluation protocol.
Project page, with video walkthroughs of the method and the hardware trials:
**https://trace-retrieval.github.io/**

Retrieving a target object from dense clutter with non-prehensile pushes, then grasping it.
A privileged teacher solves the scene once inside a digital twin built from a single RGB-D
view; a student then executes on the real robot from partial observations, conditioned on
that nominal plan and free to deviate from it.

This repository contains the full method, the baselines it is compared against, and scripts
that reproduce the simulation tables of the paper end to end.

```
trace/sim/         task, policies, digital twin, evaluation protocol, baselines
trace/hardware/    UR5e pipeline: calibration, perception, the four controllers, recording
trace/common/      path configuration shared by both
scripts/           data download and one-command reproduction
docs/              installation, reproduction, hardware build
```

## Quick start (simulation)

```bash
git clone https://github.com/Kowndinya2000/trace-code.git && cd trace-code
conda env create -f environment.yml && conda activate trace   # see docs/INSTALL.md for Isaac Gym
python scripts/download_data.py                               # assets, scenes, checkpoints (~336 MB)
python -m trace.common.paths                                  # confirm the layout resolves
python scripts/reproduce_sim.py main --gpu 0                  # main table, ~1 h on one RTX 3090
```

`scripts/reproduce_sim.py` writes `RESULTS.md` next to the raw per-scene records under
`$TRACE_RUNS`, so every reported number can be traced back to the episodes behind it.

The assets, scene sets, checkpoints and training labels live in a companion dataset,
**https://huggingface.co/datasets/Kowndi/trace**. `download_data.py` fetches them and
checks each archive against a recorded SHA-256, so a truncated or altered download fails rather
than quietly changing a result; `--only collections` adds the label sets needed to retrain a
student. `--list` prints the payload table with sizes and checksums. Terms differ per payload and
are stated on the dataset page: the grasp-quality network is not trained in this work, and `NOTICE`
records the same attributions for the code.

## What the method does

1. **Perceive once.** One top-down RGB-D frame is segmented into the eleven blocks, and each
   silhouette is matched to a known shape to recover its pose.
2. **Solve in a twin.** The scene is rebuilt in simulation and a privileged reinforcement-learning
   teacher, which sees the complete state, pushes until the target becomes graspable. Its
   end-effector path is the nominal plan.
3. **Execute closed loop.** The student sees only what the camera can see past the arm, keeps a
   recurrent memory of objects it can no longer observe, and is conditioned on a short look-ahead
   of the nominal plan. It re-perceives before every 4 cm push and may leave the plan.
4. **Grasp.** A grasp-quality network scores sixteen orientations; the robot grasps when the
   target clears the threshold with enough clearance for the jaws.

Execution is bounded by a teacher-relative budget: the plan's length plus twenty decisions, and
its nominal travel plus 15 cm. Exhausting either is a failure, not a reset.

## Reproducing the paper

| Table | Command | Cost |
|---|---|---|
| Plan-conditioned policies and references | `python scripts/reproduce_sim.py main` | ~1 GPU-hour |
| Planning and heuristic baselines | `python scripts/reproduce_sim.py baselines` | ~6 GPU-hours |
| Plan-window and recurrent-memory ablations | `python scripts/reproduce_sim.py ablations` | ~1 GPU-hour |

All of them evaluate the same 511 development scenes, with arm occlusion from the robot's own
links, 10% detection dropout, a five-step observation blackout, and the teacher-relative budget.
Confidence intervals are tier-stratified scene bootstraps over 2,000 resamples.
See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the expected numbers and per-table runtimes.

## Running on a robot

The hardware pipeline is included for anyone with the same setup: a UR5e with a Robotiq 2F-85,
one RealSense D455 looking down at the workspace, and an optional third-person webcam for
recordings. [docs/HARDWARE.md](docs/HARDWARE.md) covers the bill of materials, the ChArUco
calibration board and procedure, the workspace geometry, how to launch each controller, and how
the annotated review videos are produced.

Nothing in this repository requires a robot: the full evaluation, including every baseline and
ablation, runs in simulation.

## Layout and configuration

No path is hard-coded. `trace/common/paths.py` resolves everything from the repository root and
these variables:

| Variable | Meaning | Default |
|---|---|---|
| `TRACE_ROOT` | repository root | directory of this file |
| `TRACE_DATA` | downloaded assets, scenes, checkpoints | `$TRACE_ROOT/data` |
| `TRACE_RUNS` | where runs write results | `$TRACE_ROOT/runs` |
| `TRACE_PYTHON` | interpreter for child processes | the current one |
| `TRACE_ROBOT_IP`, `TRACE_D455_SERIAL`, `TRACE_D415_SERIAL`, `TRACE_CALIB` | hardware only | see docs/HARDWARE.md |

## License and third-party code

The simulation environment builds on Isaac Gym Preview 4 and its `IsaacGymEnvs` examples; the
parallel-search baseline derives from the authors' published implementation, retained under
`trace/sim/pmbs_baseline` with its original attribution. See [NOTICE](NOTICE).

## Citation

```bibtex
@misc{boyalakuntla2026trace,
  title = {Plan-Conditioned Imitation for Robust Object Retrieval under Self-Occlusion in Dense Clutter},
  author = {Boyalakuntla, Kowndinya and Pawar, Ajinkya and Boularias, Abdeslam and Yu, Jingjin},
  year = {2026},
  url = {https://trace-retrieval.github.io/}
}
```
