"""Talk to a running record_cameras.py process.

A RealSense can only be opened once, so while the cameras are recording the
recorder OWNS them: anything that needs a frame -- perception at the start of a
run, the re-sense after the pushes -- asks the recorder for one instead of
opening the device. That also guarantees every frame in the pipeline comes out
of an identically configured sensor. Opening the device separately let the two
paths disagree about white balance.

Phase markers live here for the same reason: several processes (perception, the
twin solver, the executor) each contribute events to one phases.json that the
recorders burn into the video, so appending has to be the shared behaviour --
whoever writes last must not erase what came before.
"""
import glob
import json
import os
import time

import cv2
import numpy as np

PHASES = "phases.json"


def request_dump(rec_dir, name="d455_topdown", timeout=45.0):
    """Ask the recorder for one aligned RGB-D frame. -> (color, depth, K, base).

    Waits for a .done file that was not there before the request, rather than
    predicting its index. The recorder numbers dumps from a counter that starts
    at 0 when IT starts, so predicting from the files already on disk only
    agrees when the output directory began empty -- re-running into a directory
    that already held two dumps had this waiting for dump2 while the recorder
    wrote dump0, and it timed out with the cameras rolling.

    The D455 may spend more than 20 seconds settling exposure on first start;
    45 seconds lets that initialization finish while preserving a finite fault
    timeout. Subsequent dumps normally complete in under a second.
    """
    ctrl = os.path.join(rec_dir, name + ".ctrl")
    if not os.path.isdir(ctrl):
        raise RuntimeError(f"no recorder control dir at {ctrl} — is record_cameras.py running?")
    pat = os.path.join(rec_dir, f"{name}_dump*.done")
    before = set(glob.glob(pat))
    t0 = time.time()
    open(os.path.join(ctrl, "dump"), "w").close()
    while True:
        now = set(glob.glob(pat))
        # A NEW PATH, or an existing one rewritten since the request. The second
        # case is not hypothetical: the recorder numbers dumps from 0 each time
        # it starts, so re-running into a directory that already holds dump0
        # overwrites that exact filename -- a set difference alone stays empty
        # and this waits out the timeout on a dump that was already served.
        fresh = [f for f in now if f not in before or os.path.getmtime(f) >= t0]
        if fresh:
            base = max(fresh, key=os.path.getmtime)[: -len(".done")]
            # A completion marker is published only after the recorder has
            # atomically installed all payloads. Still validate the payloads
            # here so a damaged external-drive write never reaches perception.
            try:
                color = cv2.imread(base + "_color.png")
                if color is None or color.size == 0:
                    raise OSError("color PNG is absent or unreadable")
                depth = np.load(base + "_depth.npy")
                if depth.ndim != 2 or depth.size == 0:
                    raise OSError("depth array is empty or malformed")
                with open(base + "_K.json") as stream:
                    K = json.load(stream)
                if not all(key in K for key in ("fx", "fy", "cx", "cy")):
                    raise OSError("intrinsics JSON is incomplete")
                return color, depth, K, base
            except (OSError, ValueError, EOFError):
                # External media can expose directory entries slightly before
                # reads settle. Retry within the same bounded request window.
                pass
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"recorder produced no complete readable {name} dump within {timeout}s")
        time.sleep(0.02)


def read_phases(rec_dir):
    """Events already logged by other stages of the same run."""
    try:
        with open(os.path.join(rec_dir, PHASES)) as f:
            d = json.load(f)
        return d.get("events", []), d.get("t0")
    except (OSError, ValueError):
        return [], None


def mark(rec_dir, phase, detail="", t0=None, events=None):
    """Append one phase event. APPENDS -- the log is shared across processes."""
    if rec_dir is None:
        return events
    prev, prev_t0 = read_phases(rec_dir)
    if events is None:                       # cross-process: re-read every time
        events = prev
    elif len(prev) > len(events):            # someone else wrote while we ran
        events = prev + [e for e in events if e not in prev]
    # Only log a CHANGE. The push loop calls this every 20 ms, which wrote ~300
    # identical entries for a 7 s push -- the file and every reader of it drown
    # in duplicates, and the overlay renders the same row either way.
    if events and events[-1]["phase"] == phase and events[-1].get("detail", "") == detail:
        return events
    events = list(events) + [{"t": time.time(), "phase": phase, "detail": detail}]
    events.sort(key=lambda e: e["t"])
    from isaacgymenvs.open_loop.trial_timing import write_json_atomic, write_trial_timing
    document = {"t0": t0 or prev_t0 or events[0]["t"], "events": events}
    try:
        write_json_atomic(os.path.join(rec_dir, PHASES), document)
        write_trial_timing(rec_dir, document)
    except OSError as exc:
        print(f"Could not persist trial timing: {exc}", flush=True)
    return events


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="append a phase marker to a run's phases.json")
    ap.add_argument("rec_dir")
    ap.add_argument("phase")
    ap.add_argument("detail", nargs="?", default="")
    ap.add_argument("--t0", type=float, default=None)
    a = ap.parse_args()
    mark(a.rec_dir, a.phase, a.detail, t0=a.t0)


def ensure_control(rtde_c, ip):
    """Return a control interface with a LIVE script.

    ur_rtde's control script is dropped by its watchdog when the client goes
    quiet, and perception is exactly that: Mask R-CNN plus the grasp network is
    seconds of silence. isConnected() still reports True afterwards -- the
    socket is fine and only the script is gone -- so every subsequent moveL
    returns without moving and the gripper closes wherever it happens to be.
    That is the whole of the "narrow miss": the same grasp that holds at 121/255
    with the script alive closes empty at 228/255 without it.
    """
    def running(control, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if control.isConnected() and control.isProgramRunning():
                return True
            time.sleep(.02)
        return False

    try:
        if not rtde_c.isConnected() and not rtde_c.reconnect():
            raise RuntimeError("RTDE control reconnect failed")
        if not rtde_c.isProgramRunning():
            accepted = rtde_c.reuploadScript()
            if accepted is False or not running(rtde_c):
                raise RuntimeError("RTDE control script reupload failed")
        if not running(rtde_c, timeout=.1):
            raise RuntimeError("RTDE control script is not running")
        return rtde_c
    except Exception:
        try:
            rtde_c.disconnect()
        except Exception:
            pass
        from rtde_control import RTDEControlInterface
        replacement = RTDEControlInterface(ip)
        if not running(replacement):
            replacement.disconnect()
            raise RuntimeError("Could not start a verified RTDE control script")
        return replacement
