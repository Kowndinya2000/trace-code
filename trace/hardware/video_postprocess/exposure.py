"""Reproducible, spatially uniform exposure corrections for figure snapshots and videos.

The still-image half (TARGET_P90, HIGHLIGHT_KNEE, brightness_lut, luminance_p90,
adjust_brightness) is the figure-snapshot transfer, unchanged. The video half applies the
same curve inside ffmpeg: the gain is measured once per stream from sampled frames of the
exact crop that will be shown, so a whole clip gets one spatially uniform correction.

    transfer, meta = video_transfer(path, 'd455', crop='crop=iw*0.75:ih:iw*0.25:0,hflip,vflip')
    filter_chain = f"[1:v]{crop},{transfer}scale=..."
"""
import json
import math
import subprocess
from pathlib import Path

TARGET_P90 = 208.0
HIGHLIGHT_KNEE = 224.0
GAMMA = {'webcam': 1.16, 'd455': 1.5, 'd415': 1.16}


def brightness_lut(gamma, gain):
    """Lift midtones and roll off highlights smoothly; apply equally to RGB."""
    values = []
    for value in range(256):
        lifted = gain * 255 * (value / 255) ** (1 / gamma)
        if lifted > HIGHLIGHT_KNEE:
            room = 255 - HIGHLIGHT_KNEE
            lifted = HIGHLIGHT_KNEE + room * (
                1 - math.exp(-(lifted - HIGHLIGHT_KNEE) / room)
            )
        values.append(round(lifted))
    return values


def luminance_p90(image):
    """BT.709 luminance percentile, measured over the full, unannotated crop."""
    histogram = image.convert(
        'L', (0.2126, 0.7152, 0.0722, 0)
    ).histogram()
    threshold = image.width * image.height * 0.9
    total = 0
    for value, count in enumerate(histogram):
        total += count
        if total >= threshold:
            return value


def adjust_brightness(image, camera):
    gamma = 1.16 if camera == 'webcam' else 1.5
    gamma_lut = [round(255 * (value / 255) ** (1 / gamma))
                 for value in range(256)]
    reference_p90 = luminance_p90(image.point(gamma_lut * 3))
    gain = round(max(1.0, min(2.0, TARGET_P90 / max(reference_p90, 1))), 6)
    corrected = image.point(brightness_lut(gamma, gain) * 3)
    return corrected, {
        'gamma': gamma,
        'brightness_gain': gain,
        'brightness_transfer': 'gamma_gain_exponential_highlight_shoulder_v1',
        'highlight_knee_8bit': HIGHLIGHT_KNEE,
        'brightness_target_p90_8bit': TARGET_P90,
        'source_luminance_p90_8bit': luminance_p90(image),
        'corrected_luminance_p90_8bit': luminance_p90(corrected),
    }


# ------------------------------------------------------------------ video
def duration(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                   '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True)
    return float(out.strip())


def sample_frames(path, crop=None, samples=9, encoder='ffmpeg'):
    """PIL frames of the displayed crop, evenly spaced over the clip (never the first frame)."""
    from PIL import Image
    import io
    total = duration(path)
    times = [total * (i + 1) / (samples + 1) for i in range(samples)]
    frames = []
    for when in times:
        command = [encoder, '-v', 'error', '-ss', f'{when:.3f}', '-i', str(path), '-frames:v', '1']
        if crop:
            command += ['-vf', crop]
        command += ['-f', 'image2', '-c:v', 'png', '-']
        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0 and result.stdout:
            frames.append(Image.open(io.BytesIO(result.stdout)).convert('RGB'))
    if not frames:
        raise RuntimeError(f'No frames sampled from {path}')
    return frames


def measure_gain(frames, gamma):
    """Median of the per-frame gains the still-image rule would choose."""
    gamma_lut = [round(255 * (value / 255) ** (1 / gamma)) for value in range(256)]
    p90s, gains = [], []
    for frame in frames:
        reference = luminance_p90(frame.point(gamma_lut * 3))
        p90s.append(reference)
        gains.append(max(1.0, min(2.0, TARGET_P90 / max(reference, 1))))
    gains.sort()
    p90s.sort()
    middle = len(gains) // 2
    gain = gains[middle] if len(gains) % 2 else 0.5 * (gains[middle - 1] + gains[middle])
    return round(gain, 6), p90s[len(p90s) // 2]


def ffmpeg_expression(gamma, gain):
    """The brightness_lut curve as one ffmpeg lut expression (commas escaped for filtergraphs)."""
    room = 255 - HIGHLIGHT_KNEE
    lifted = f'{gain:.6f}*255*pow(val/255\\,{1 / gamma:.6f})'
    shoulder = (f'{HIGHLIGHT_KNEE:.0f}+{room:.0f}*(1-exp(-(LIFT-{HIGHLIGHT_KNEE:.0f})/{room:.0f}))'
                .replace('LIFT', lifted))
    # +0.5 so ffmpeg's truncation matches Python's round() in brightness_lut
    return f'0.5+if(gt({lifted}\\,{HIGHLIGHT_KNEE:.0f})\\,{shoulder}\\,{lifted})'


def video_transfer(path, camera, crop=None, saturation=1.0, samples=9):
    """(filter prefix, metadata) applying the snapshot exposure curve to a whole clip."""
    gamma = GAMMA[camera]
    gain, source_p90 = measure_gain(sample_frames(path, crop, samples), gamma)
    expression = ffmpeg_expression(gamma, gain)
    chain = (f'format=gbrp,lutrgb=r={expression}:g={expression}:b={expression},format=yuv420p,')
    if saturation != 1.0:
        chain += f'eq=saturation={saturation},'
    return chain, {
        'camera': camera,
        'gamma': gamma,
        'brightness_gain': gain,
        'brightness_transfer': 'gamma_gain_exponential_highlight_shoulder_v1',
        'highlight_knee_8bit': HIGHLIGHT_KNEE,
        'brightness_target_p90_8bit': TARGET_P90,
        'median_gamma_corrected_p90_8bit': source_p90,
        'frames_sampled': samples,
        'saturation': saturation,
        'source': str(Path(path)),
        'crop': crop,
    }


def write_metadata(path, entries):
    Path(path).write_text(json.dumps(entries, indent=2) + '\n')
