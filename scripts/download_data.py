#!/usr/bin/env python3
"""Fetch the assets, scenes, checkpoints and training data TRACE needs.

    python scripts/download_data.py                 # everything needed to reproduce the paper
    python scripts/download_data.py --only scenes checkpoints
    python scripts/download_data.py --list          # payloads, sizes and checksums
    python scripts/download_data.py --verify        # re-check what is already installed

Each payload is a .tar.gz verified against the SHA-256 recorded below, so a truncated download
fails loudly instead of quietly changing a result. Simulation assets extract into the simulator
package (the task loads them by a path relative to its own file); everything else lands under
$TRACE_DATA, by default <repo>/data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace.common import paths  # noqa: E402

BASE_URL = os.environ.get("TRACE_DATA_URL", "https://huggingface.co/datasets/Kowndi/trace/resolve/main")

PAYLOADS = {
    "assets": dict(file="assets.tar.gz", size_mb=107, sha256="c8e79a67cf71cc0d96dcca81a18a6d83ede9d0eb06e8eb3ad9f923c574347aa8", into="sim",
                   note="UR5e, workspace and block URDFs with their collision meshes"),
    "scenes": dict(file="scenes.tar.gz", size_mb=1, sha256="9d0b72c9275f2cad5a81227e883a53614d52807bc4437950051716c1e79be110", into="data",
                   note="1,783 fit / 314 validation / 511 paper-evaluation scenes and manifests"),
    "checkpoints": dict(file="checkpoints.tar.gz", size_mb=68, sha256="7fa3e736270272159807510b61d490190946277256a11dab333c216f7aa606ea", into="data",
                        note="teacher, students for three seeds, label controls and ablation fits"),
    "grasp_models": dict(file="grasp_models.tar.gz", size_mb=160, sha256="7315d513e59786d312641515a9447c8f9c02608d9208749cfc0648b516aa048a", into="sim_pkg",
                         note="grasp-quality network and its classifier (see NOTICE)"),
    "collections": dict(file="collections.tar.gz", size_mb=985, sha256="8bdcca8dd1bcfe851879aa1166eb15bccfe5b3101eeafac52efe6f56cb2f99ec", into="data",
                        note="expert and DAgger label sets for all three seeds; needed only to retrain"),
}
DEFAULT = ["assets", "scenes", "checkpoints", "grasp_models"]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def destination(name: str) -> Path:
    into = PAYLOADS[name]["into"]
    if into == "sim":
        return paths.SIM                      # assets/ sits beside the isaacgymenvs package
    if into == "sim_pkg":
        return paths.SIM / "isaacgymenvs"     # the baseline code loads these by their upstream name
    return paths.DATA


def relocate_manifests() -> int:
    """Rewrite manifest scene paths to absolute ones and verify every scene against its hash.

    Manifests ship with paths relative to the scenes directory so the archive is portable. The
    evaluator deliberately refuses relative paths -- a scene has to be identified by content, not
    by whatever happens to sit at a relative location -- so this runs once after extraction.
    """
    total = 0
    for manifest in sorted(paths.SCENES.glob("*.json")):
        payload = json.loads(manifest.read_text())
        if "scenes" not in payload:
            continue
        for row in payload["scenes"]:
            scene = Path(row["path"])
            if not scene.is_absolute():
                scene = (paths.SCENES / scene).resolve()
            if not scene.exists():
                raise SystemExit(f"{manifest.name}: missing scene {scene}")
            if digest(scene) != row["sha256"]:
                raise SystemExit(f"{manifest.name}: {scene} does not match its recorded hash")
            row["path"] = str(scene)
        manifest.write_text(json.dumps(payload, indent=1) + "\n")
        total += len(payload["scenes"])
        print(f"[scenes] {manifest.name}: {len(payload['scenes'])} scenes verified")
    return total


def fetch(name: str) -> None:
    payload = PAYLOADS[name]
    target = destination(name)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / payload["file"]
    url = f"{BASE_URL}/{payload['file']}"
    if not archive.exists():
        print(f"[{name}] downloading {payload['size_mb']} MB from {url}")
        urllib.request.urlretrieve(url, archive)
    if payload["sha256"] != "TBD":
        got = digest(archive)
        if got != payload["sha256"]:
            raise SystemExit(f"[{name}] checksum mismatch\n  expected {payload['sha256']}\n  got      {got}")
    print(f"[{name}] extracting into {target}")
    shutil.unpack_archive(str(archive), str(target))
    archive.unlink()
    if name == "scenes":
        relocate_manifests()
    print(f"[{name}] ready")


def verify() -> None:
    ok = True
    for label, path in (("assets", paths.ASSETS), ("scenes", paths.SCENES), ("checkpoints", paths.CHECKPOINTS),
                        ("grasp models", paths.GRASP_MODELS), ("collections", paths.COLLECTIONS)):
        present = Path(path).is_dir() and any(Path(path).iterdir())
        print(f"{'ok  ' if present else 'MISSING'}  {label:<12} {path}")
        ok &= present or label == "collections"
    for manifest in sorted(paths.SCENES.glob("*.json")) if paths.SCENES.is_dir() else []:
        rows = json.loads(manifest.read_text()).get("scenes", [])
        bad = [r for r in rows if not Path(r["path"]).is_absolute()]
        if bad:
            print(f"MISSING   {manifest.name} still holds relative paths; rerun --only scenes")
            ok = False
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(PAYLOADS), default=DEFAULT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.list:
        width = max(len(n) for n in PAYLOADS)
        for name, payload in PAYLOADS.items():
            mark = " " if name in DEFAULT else "*"
            print(f"{mark} {name:<{width}}  {payload['size_mb']:>4} MB  -> {destination(name)}  {payload['note']}")
        print("\n* not downloaded by default")
        return
    if args.verify:
        verify()
    for name in args.only:
        fetch(name)
    print(f"\ndata root:      {paths.DATA}\nsimulator root: {paths.SIM}")


if __name__ == "__main__":
    main()
