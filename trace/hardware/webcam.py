"""Plain UVC webcam as a frame source.

Used here by record_webcam.py for the third-person view.

Used for the third-person view of the robot. Unlike the RealSense, this is an
ordinary V4L2 device with no depth, no filters and no calibration, so it is just
cv2.VideoCapture on its own reader thread. It exposes the same one-method
contract Camera does, so VideoRecorder takes either without changes:

    wait_for_color_bgr(last_seq, timeout) -> (frame_bgr | None, seq)

    from ur_tools.camera.webcam import WebcamSource
    from ur_tools.camera.video_recorder import VideoRecorder

    cam = WebcamSource().start()
    rec = VideoRecorder(cam, "pov.mp4").start()
    ...
    rec.stop(); cam.stop()

To frame the camera before a run, preview it:

    python -m ur_tools.camera.webcam
"""

import atexit
import glob
import os
import threading
import time
import traceback
from collections import namedtuple

import cv2

# Addressed by by-id rather than /dev/videoN: the numeric index shifts whenever
# devices are replugged, the by-id symlink does not.
DEFAULT_DEVICE_GLOB = "/dev/v4l/by-id/usb-046d_HD_Webcam_C615_*-video-index0"

DEFAULT_WIDTH = 1920      # the C615 does 1920x1080 MJPG at a measured 30 fps
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0

_Frame = namedtuple("_Frame", ["bgr", "seq"])


def default_device():
    """The C615 by its stable by-id path, or None if it is not plugged in."""
    matches = sorted(glob.glob(DEFAULT_DEVICE_GLOB))
    return matches[0] if matches else None


class WebcamSource:
    """Reads a UVC webcam on a background thread and publishes the latest frame."""

    def __init__(self, device=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 fps=DEFAULT_FPS, raw_jpeg=False):
        # raw_jpeg: hand out the camera's own MJPEG frame bytes (CONVERT_RGB off)
        # instead of a decoded BGR image. The recorder spools those verbatim --
        # zero decode cost and zero re-encode loss at capture; the camera's JPEG
        # IS the source quality. 183 KB/frame at 1920x1080 on the C615.
        self.raw_jpeg = bool(raw_jpeg)
        self.device = device
        self.width = width
        self.height = height
        self.fps = float(fps)

        self._cap = None
        self._thread = None
        self._frame_cond = threading.Condition()
        self._latest = None
        self._frame_seq = 0
        self._reader_exception = None
        self._stop_event = threading.Event()
        self._started = False
        self._stopped = False

    def start(self, first_frame_timeout=5.0):
        """Open the device and begin reading. Returns self so it can be chained."""
        if self._started:
            return self
        self._started = True

        device = self.device or default_device()
        if device is None:
            raise RuntimeError(
                "no webcam found matching %s (is it plugged in?)" % DEFAULT_DEVICE_GLOB
            )
        # An int index still works if someone passes one explicitly.
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        if isinstance(device, str) and not os.path.exists(device):
            raise RuntimeError("webcam device does not exist: %s" % device)

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("could not open webcam %s" % device)

        # Order matters: the fourcc has to be set before the frame size or V4L2
        # may refuse the mode and quietly leave the device in its default format.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify rather than trust. This camera offers 1280x720 at 30fps in MJPG
        # but only 10fps in YUYV, and OpenCV's V4L2 backend defaults to YUYV --
        # so a silent fallback would record a third of the frames with no error.
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        if fourcc_str != "MJPG":
            cap.release()
            raise RuntimeError(
                "webcam %s negotiated %r instead of MJPG; at %dx%d that caps the "
                "frame rate well below %.0ffps"
                % (device, fourcc_str, self.width, self.height, self.fps)
            )

        if self.raw_jpeg:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._cap = cap
        self._thread = threading.Thread(
            target=self._reader_loop, name="webcam-reader", daemon=True
        )
        self._thread.start()

        # Surface a dead-on-arrival camera here rather than at the first read.
        deadline = time.monotonic() + first_frame_timeout
        with self._frame_cond:
            while self._latest is None and self._reader_exception is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "no frame from webcam %s within %.1fs"
                        % (device, first_frame_timeout)
                    )
                self._frame_cond.wait(remaining)
            if self._reader_exception is not None:
                raise RuntimeError("webcam reader thread died") from self._reader_exception

        atexit.register(self.stop)
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("Webcam %s open: %dx%d MJPG @ %.0ffps" % (device, actual[0], actual[1], self.fps))
        return self

    def _reader_loop(self):
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    # A dropped frame is routine on USB; only give up if the
                    # camera stops producing entirely (unplugged mid-run).
                    consecutive_failures += 1
                    if consecutive_failures >= 100:
                        raise RuntimeError("webcam stopped delivering frames")
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0

                if self.raw_jpeg:
                    frame = frame.tobytes()     # the JPEG the camera produced
                with self._frame_cond:
                    self._frame_seq += 1
                    self._latest = _Frame(frame, self._frame_seq)
                    self._frame_cond.notify_all()
        except Exception as exc:
            self._reader_exception = exc
            print("Webcam reader thread died:\n" + traceback.format_exc())
        finally:
            # Wake every blocked consumer so they return instead of hanging.
            with self._frame_cond:
                self._frame_cond.notify_all()

    def wait_for_color_bgr(self, last_seq=0, timeout=1.0):
        """Latest frame as BGR plus its sequence number.

        Same contract as Camera.wait_for_color_bgr, so VideoRecorder can consume
        either. Returns (None, last_seq) if nothing newer arrived within `timeout`.
        """
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self._reader_exception is None and (
                self._latest is None or self._frame_seq <= last_seq
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq
                self._frame_cond.wait(remaining)
            if self._reader_exception is not None:
                raise RuntimeError("webcam reader thread died") from self._reader_exception
            frame = self._latest
        return frame.bgr, frame.seq

    def show_video_realtime(self, window_scale=0.8, show_stats=True):
        """Show the live third-person view, to frame the camera before a run.

        Mirrors Camera.show_video_realtime, minus depth. Starts the source if it
        is not already running.

        Args:
            window_scale (float): scale factor for the display window (0.1 to 1.0)
            show_stats (bool): overlay resolution and measured frame rate

        Controls:
            - Press 'q' to quit
            - Press 's' to save current frame
            - Press 'r' to reset window positions
        """
        if not self._started:
            self.start()

        print("Starting third-person video stream...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset window positions")

        frame_count = 0
        seq = 0
        measured_fps = 0.0
        window = "Webcam Third-Person View"
        fps_mark = time.monotonic()
        fps_count = 0

        try:
            while True:
                # Frames from the reader thread are already BGR straight off the
                # V4L2 device -- no colour conversion here. (Camera.get_data
                # returns RGB and does need one; copying that would invert this.)
                frame, seq = self.wait_for_color_bgr(seq, timeout=1.0)
                if frame is None:
                    # No new frame this cycle; keep the window responsive so 'q'
                    # still works if the camera has stalled.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # Measured over a rolling second: this is the cheapest way to see
                # that the device really negotiated 30fps MJPG and did not fall
                # back to a slower mode.
                fps_count += 1
                now = time.monotonic()
                if now - fps_mark >= 1.0:
                    measured_fps = fps_count / (now - fps_mark)
                    fps_count = 0
                    fps_mark = now

                display = frame
                if window_scale != 1.0:
                    height, width = display.shape[:2]
                    display = cv2.resize(
                        display, (int(width * window_scale), int(height * window_scale))
                    )

                if show_stats:
                    label = "%dx%d  %.1f fps" % (
                        frame.shape[1], frame.shape[0], measured_fps
                    )
                    cv2.putText(display, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(display, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quitting video stream...")
                    break
                elif key == ord("s"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = "pov_frame_%s.png" % timestamp
                    # The full-resolution frame, not the scaled preview.
                    cv2.imwrite(filename, frame)
                    print("Saved frame: %s" % filename)
                elif key == ord("r"):
                    cv2.destroyAllWindows()
                    print("Reset window positions")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print("Error during video streaming: %s" % e)
        finally:
            cv2.destroyAllWindows()
            print("Video stream ended. Total frames: %d" % frame_count)

    def stop(self):
        """Release the device. Safe to call more than once."""
        if self._stopped or not self._started:
            self._stopped = True
            return
        self._stopped = True
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("Webcam reader thread did not exit; releasing device anyway")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("Webcam released")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview the third-person webcam (q quit, s save frame, r reset windows)"
    )
    parser.add_argument("--device", default=None, help="device path (default: C615 by-id)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--scale", type=float, default=0.8, help="window scale (default: 0.8)")
    args = parser.parse_args()

    cam = WebcamSource(
        device=args.device, width=args.width, height=args.height, fps=args.fps
    )
    try:
        cam.show_video_realtime(window_scale=args.scale)
    finally:
        cam.stop()
