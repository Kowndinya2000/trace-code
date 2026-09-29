"""Third-person video from the C615 webcam, same protocol as record_cameras.py.

  python record_webcam.py --name webcam_scene --out runs/demoN --t0 <epoch> \
      --label "Real robot - open-loop retrieval" --subtitle "<checkpoint>"

Control files in <out>/<name>.ctrl/, identical to the RealSense recorder so the
orchestrators drive both the same way:
  ready  written once frames are flowing
  stop   touch it -> encode and exit
(No `dump`: a webcam has no depth, so it serves no re-sense.)

Quality. The camera produces MJPEG; the reader hands those bytes out verbatim
(webcam.WebcamSource raw_jpeg) and they are spooled as-is -- 1920x1080 at a
measured 30 fps, ~183 KB per frame, ~5.5 MB/s, no decode and no re-encode at
capture. Decoding, the overlay and the x264 encode (crf 14, preset slow) all
happen after `stop`, so the robot never pays for them. That is the best this
camera can give: what it recorded is exactly what the sensor emitted.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webcam import WebcamSource, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS   # noqa: E402
from record_cameras import encode, load_phases, numeric_path_key             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None, help="V4L2 path (default: C615 by-id)")
    ap.add_argument("--name", default="webcam_scene")
    ap.add_argument("--out", default="runs/real")
    ap.add_argument("--t0", type=float, default=0.0, help="shared epoch for the wall-clock")
    ap.add_argument("--label", default="Real robot - third-person view")
    ap.add_argument("--subtitle", default="", help="second line, e.g. the checkpoint")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--layout", choices=("corner", "band"), default="corner")
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--endcard-seconds", type=float, default=4.0)
    ap.add_argument("--exec-label", default=None, help="EXECUTE group header")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ctrl = os.path.join(args.out, args.name + ".ctrl")
    os.makedirs(ctrl, exist_ok=True)
    for f in ("ready", "stop"):
        try:
            os.remove(os.path.join(ctrl, f))
        except OSError:
            pass
    spool = os.path.join(args.out, args.name + "_frames")
    shutil.rmtree(spool, ignore_errors=True)
    os.makedirs(spool)

    cam = WebcamSource(device=args.device, width=args.width, height=args.height,
                       fps=args.fps, raw_jpeg=True).start()
    stamps, t0, seq = [], None, 0
    open(os.path.join(ctrl, "ready"), "w").close()
    print(f"[rec] {args.name}: {args.width}x{args.height} MJPG @ {args.fps:.0f} fps, spooling raw JPEG")
    try:
        while not os.path.exists(os.path.join(ctrl, "stop")):
            jpeg, seq = cam.wait_for_color_bgr(seq, timeout=1.0)
            if jpeg is None:
                continue
            now = time.time()
            if t0 is None:
                t0 = args.t0 if args.t0 > 0 else now
            with open(os.path.join(spool, "%06d.jpg" % len(stamps)), "wb") as f:
                f.write(jpeg)
            stamps.append(now)
    finally:
        cam.stop()

    out_mp4 = os.path.join(args.out, args.name + ".mp4")
    cards = sorted(glob.glob(os.path.join(args.out, "*_gn16.png")), key=numeric_path_key)
    if not cards:
        cards = sorted(glob.glob(os.path.join(args.out, "perception", "*_gn16.png")))
        cards.sort(key=numeric_path_key)
    written = encode(spool, stamps, out_mp4, args.fps, t0,
                     args.label or args.name.replace("_", " "),
                     events=load_phases(args.out), subtitle=args.subtitle,
                     endcard=cards[-1] if cards else None,
                     endcard_s=args.endcard_seconds,
                     crf=args.crf, preset=args.preset, layout=args.layout, ext="jpg",
                     exec_label=args.exec_label)
    meas = ((len(stamps) - 1) / (stamps[-1] - stamps[0])) if len(stamps) > 1 else args.fps
    meta = {"name": args.name, "device": args.device or "C615 by-id", "frames": len(stamps),
            "stream_fps": args.fps, "encoded_fps": round(meas, 3),
            "width": args.width, "height": args.height, "source": "MJPG raw passthrough",
            "t0_epoch": t0, "duration_s": round((stamps[-1] - t0), 3) if stamps else 0.0,
            "video": written, "dumps": 0}
    json.dump(meta, open(os.path.join(args.out, args.name + "_rec.json"), "w"), indent=2)
    if not written or not os.path.isfile(written) or os.path.getsize(written) == 0:
        raise RuntimeError(f"final video missing or empty; preserving source frames in {spool}")
    shutil.rmtree(spool)
    print(f"[rec] {args.name}: {len(stamps)} frames, {meta['duration_s']} s -> {written}; source frames removed")


if __name__ == "__main__":
    main()
