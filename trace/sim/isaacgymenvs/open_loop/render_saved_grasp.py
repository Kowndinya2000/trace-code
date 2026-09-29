"""Regenerate GN visuals from saved predictions, without inference or hardware.

python -m isaacgymenvs.open_loop.render_saved_grasp capture_gn16_predictions.npz --tiles
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from isaacgymenvs.open_loop.grasp_artifacts import load_grasp_record
from isaacgymenvs.open_loop.real_grasp import render_rotation_panel
from isaacgymenvs.utils.prediction_vis import render_prediction_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, help="new grid path (PNG)")
    parser.add_argument("--tiles", action="store_true",
                        help="export individual tiles even when the recorded grasp was rejected")
    args = parser.parse_args()
    dbg, metadata = load_grasp_record(args.record)
    if metadata["grasp"] is None or "preds" not in dbg:
        parser.error("This capture has no GN predictions; its source frame is preserved for later inference.")
    dbg["helper"] = SimpleNamespace(get_prediction_vis=render_prediction_grid)
    stem = args.record.stem
    if stem.endswith("_predictions"):
        stem = stem[:-len("_predictions")]
    output = args.output or args.record.with_name(stem + "_regenerated.png")
    render_rotation_panel(dbg, metadata["grasp"], output,
                          threshold=metadata["threshold"], tile_size=metadata["tile_size"],
                          save_tiles=True if args.tiles else None)
    print(f"Grid: {output}")
    tiles = output.with_name(output.stem + "_tiles")
    if any(tiles.glob("bin_*.png")):
        print(f"Tiles: {tiles}")


if __name__ == "__main__":
    main()
