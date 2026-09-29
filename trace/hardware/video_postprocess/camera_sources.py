"""Resolve an optional third-person camera without inventing footage."""
import atexit
import json
from pathlib import Path
import subprocess
import tempfile


def third_person(trial, raw=True):
    trial = Path(trial)
    suffix = '_raw.mp4' if raw else '.mp4'
    for camera in ('webcam_scene', 'd415_scene'):
        video = trial / (camera + suffix)
        if video.is_file():
            for folder in (trial / 'metadata', trial):
                metadata = folder / (camera + '_rec.json')
                if metadata.is_file():
                    return video, json.loads(metadata.read_text())
            return video, {}
    overhead = trial / ('d455_topdown' + suffix)
    duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(overhead)]))
    temporary = tempfile.TemporaryDirectory(prefix='trace-missing-camera-')
    atexit.register(temporary.cleanup)
    video = Path(temporary.name) / 'not-connected.mp4'
    from compose_closed_loop import find_encoder
    subprocess.run([find_encoder(), '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=black:s=1280x720:r=15', '-t', str(duration), '-vf',
                    "drawtext=text='Third-person camera not connected':fontcolor=white:"
                    'fontsize=28:x=(w-tw)/2:y=(h-th)/2',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                    str(video)], check=True)
    return video, {'duration_s': duration}
