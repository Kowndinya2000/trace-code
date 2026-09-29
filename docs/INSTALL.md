# Installation

Tested on Ubuntu 22.04 with CUDA 11.7 and an RTX 3090 (24 GB). One GPU is enough for everything
in this repository; the evaluation uses about 6 GB and training about 8 GB.

## 1. Isaac Gym Preview 4

Isaac Gym is not redistributable, so it has to be fetched once from NVIDIA and installed into the
environment by hand.

```bash
# download IsaacGym_Preview_4_Package.tar.gz from https://developer.nvidia.com/isaac-gym
tar -xf IsaacGym_Preview_4_Package.tar.gz
```

Keep the extracted `isaacgym/` directory; step 2 installs from it.

## 2. Environment

```bash
conda env create -f environment.yml      # python 3.8, torch 1.13.1+cu117, and the rest
conda activate trace
pip install -e /path/to/isaacgym/python  # the package extracted in step 1
pip install -e .                         # this repository
```

The renderer needs a working Vulkan driver even in headless mode, because the grasp network is
scored on rendered depth. `vulkaninfo | head` should print a device. On a headless machine, run
under `xvfb-run -a`.

## 3. Data

The archives are hosted at **https://huggingface.co/datasets/Kowndi/trace**, and every
one is verified against the SHA-256 recorded in `scripts/download_data.py`, so a truncated or
altered download fails instead of quietly changing a result. `TRACE_DATA_URL` overrides the
location if you mirror them.

```bash
python scripts/download_data.py --list   # payloads, sizes and checksums
python scripts/download_data.py          # ~336 MB: assets, scenes, checkpoints, grasp models
```

Add `--only collections` if you intend to retrain a student. The hardware perception stack
also needs Mask R-CNN weights, which are not published here because they are specific to one
camera viewpoint; see docs/HARDWARE.md.

## 4. Check

```bash
python -m trace.common.paths             # every row should read "ok"
python scripts/selftest.py               # loads the task, a checkpoint, and solves two scenes
```

The self-test takes about two minutes and needs no robot. If it prints `self-test passed` with two
completed scenes, the installation is complete; the scenes are real evaluation episodes, so an
individual episode may legitimately end in failure.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError: libpython3.8.so.1.0` | Isaac Gym needs the conda environment active; re-run `conda activate trace` |
| `[Error] [carb.gym.plugin] Failed to create a valid Vulkan device` | no Vulkan driver, or no display; install the driver or use `xvfb-run -a` |
| `Segmentation fault` on the first `gym.create_sim` | mismatched CUDA driver; Isaac Gym Preview 4 needs driver 515 or newer |
| Evaluation reports `Missing .../scenes/development.json` | data step skipped; run `python scripts/download_data.py` |
