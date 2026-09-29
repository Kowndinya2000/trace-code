"""Cross-cluster checkpoint loading.

rl_games calls ``torch.load`` without ``map_location``, so a checkpoint saved
on (say) GPU 4 of a multi-GPU node fails to deserialize on a machine with fewer:
    Attempting to deserialize object on CUDA device 4 but
    torch.cuda.device_count() is 1
Import this module before ``runner.create_player()`` / ``player.restore()`` in
any tool that may load a checkpoint produced elsewhere.
"""
import torch

_orig_load = torch.load


def _safe_load(*args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = "cpu"
    return _orig_load(*args, **kwargs)


def patch_torch_load_cpu():
    """Force map_location='cpu' for every torch.load in this process."""
    if getattr(torch.load, "_ckpt_compat", False):
        return
    _safe_load._ckpt_compat = True
    torch.load = _safe_load


patch_torch_load_cpu()
