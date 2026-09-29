#!/usr/bin/env python3
"""Generate reproducible real/sim target poses for closed-loop trials."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "closed-loop exps" / "target_pose_protocol.json"
HOVER_Z_M = 0.05

# Base sim coordinates and the corresponding real hover coordinates. Scene 3
# and Scene 4 retain the calibrated grasp/placement correction in real space.
BASES = {
    "scene1": {
        "sim_xy_m": [0.472510, 0.053438],
        "real_xy_m": [0.053438, -0.472510],
        "source": "saved target pose for this scene",
    },
    "scene2": {
        "sim_xy_m": [0.547424, 0.054246],
        "real_xy_m": [0.054246, -0.547424],
        "source": "validated Scene 2 initial hover record",
    },
    "scene3": {
        "sim_xy_m": [0.500272, -0.002650],
        "real_xy_m": [-0.005817941717689327, -0.5026719227807319],
        "source": "calibrated placement record",
    },
    "scene4": {
        "sim_xy_m": [0.489270, 0.014942],
        "real_xy_m": [0.011774058282310674, -0.49166992278073185],
        "source": "validated hover record",
    },
    "scene5": {
        "sim_xy_m": [0.514662, -0.005646],
        "real_xy_m": [-0.005646, -0.514662],
        "source": "saved target pose for this scene",
    },
    "scene6": {
        "sim_xy_m": [0.501048, 0.002496],
        "real_xy_m": [0.002496, -0.501048],
        "source": "saved target pose for this scene",
    },
    "scene8": {
        "sim_xy_m": [0.515189, 0.019672],
        "real_xy_m": [0.019672, -0.515189],
        "source": "saved target pose for this scene",
    },
}


def rounded(values: list[float]) -> list[float]:
    return [round(value, 9) for value in values]


def main() -> None:
    scenes = {}
    for scene, base in BASES.items():
        base_sim = base["sim_xy_m"]
        base_real = base["real_xy_m"]
        trials = {}
        for trial in ("trial1", "trial2"):
            trials[trial] = {
                "disposition": "fixed base pose",
                "offset_real_xy_m": [0.0, 0.0],
                "offset_radius_m": 0.0,
                "sim_xy_m": rounded(base_sim),
                "real_xy_m": rounded(base_real),
                "hover_tcp_z_m": HOVER_Z_M,
            }
        scenes[scene] = {
            "base": base,
            "trials": trials,
        }

    document = {
        "protocol_version": 2,
        "randomization": "none",
        "random_seed": None,
        "policy": {
            "target_pose": "fixed saved base pose for every scene, method, and trial",
            "perturbation_radius_m": 0.0,
            "comparison_rule": "all compared methods use the identical scene-by-trial pose manifest",
            "selection_rule": "never change the target pose in response to trial outcomes",
            "sim_to_real_xy": "real_x=sim_y; real_y=-sim_x, plus retained calibration correction",
        },
        "scenes": scenes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
