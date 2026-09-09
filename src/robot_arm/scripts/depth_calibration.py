"""
depth_calibration.py
====================
Linear correction for this Orbbec Astra Pro's depth readings.

This particular unit reports depth that is wrong by a large but *consistent*
amount: measured 41 mm at a true 500 mm, and 56 mm at a true 1000 mm. The
readings are stable and repeatable (spread of only 1-2 mm over 300 samples),
so the sensor is not noisy, it is mis-scaled. A two-parameter affine fit
recovers the real distance:

    reported = scale * true + offset          <- what we fit
    true     = (reported - offset) / scale    <- what we apply

Fit the constants with data_collection/depth_calibration_gui.py, which writes
config/depth_calibration.json. Consumers (inference_node.py) call load() once
and apply() per frame; with no saved calibration load() returns None and the
raw values pass through unchanged.
"""

import json
import os
from datetime import datetime

import numpy as np

CALIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config', 'depth_calibration.json')


def load(path: str = CALIB_PATH):
    """Return (scale, offset), or None if nothing physically valid is saved."""
    try:
        with open(path) as f:
            d = json.load(f)
        scale = float(d['scale'])
        offset = float(d['offset'])
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if scale < 1e-9:
        # A depth sensor has to report larger numbers for farther objects, so
        # a zero or negative slope is not a sensor that needs correcting, it
        # is evidence the readings are not distance at all. Applying such a
        # fit would silently inverting near and far, which is far worse than
        # having no calibration, so it is refused rather than trusted.
        # Measured on this unit: 300mm->55, 400mm->46.5, 500mm->38, 600mm->42,
        # which is non-monotonic and cannot be any affine map of distance.
        print(f'[depth_calibration] REFUSING saved calibration in {path}: '
              f'scale={scale:.5f} is not positive, so the fitted readings do '
              f'not increase with distance. Depth will pass through raw.')
        return None
    return scale, offset


def save(scale: float, offset: float, points, r2: float, path: str = CALIB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'scale': scale,
            'offset': offset,
            'r2': r2,
            'points_true_mm_reported_mm': [[float(a), float(b)] for a, b in points],
            'saved': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)


def fit(points):
    """
    points: list of (true_mm, reported_mm), at least 2.
    Returns (scale, offset, r2). Two points always give r2 == 1.0, which
    proves nothing -- three or more is what actually tests linearity.
    """
    t = np.asarray([p[0] for p in points], dtype=float)
    r = np.asarray([p[1] for p in points], dtype=float)
    scale, offset = np.polyfit(t, r, 1)
    pred = scale * t + offset
    ss_res = float(((r - pred) ** 2).sum())
    ss_tot = float(((r - r.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(scale), float(offset), float(r2)


def load_depth_stack(npz_path: str, path: str = CALIB_PATH):
    """
    Read a recorder-written *_depth.npz and return (stack_in_true_mm, note).

    Recordings differ in what their stored numbers mean: ones captured before
    a calibration existed hold the sensor's raw mis-scaled values, ones after
    hold true mm. The recorder marks which via a 'calibrated' flag; files
    without the flag predate it and are raw. Correcting an already-corrected
    stack would be far worse than leaving it alone, so the flag decides and
    the value ranges are never guessed at.

    A raw stack with no calibration saved yet is returned untouched, with a
    note saying so, so the caller can warn instead of silently fusing
    garbage.
    """
    z = np.load(npz_path)
    stack = z['depth_mm'].astype(np.float32)

    # Recordings made before the Y11 discovery hold Y12 frames, which are not
    # distance at all (~9 flat values, non-monotonic). No calibration applies
    # to them, and correcting them with the Y11 constants would produce
    # confident nonsense rather than an obvious failure, so they are refused.
    # Files written since then record which format they came from.
    if 'depth_format' not in z:
        return None, ('PRE-Y11 recording (Y12 frames, not distance) -- '
                      'unusable, re-record')
    fmt = str(z['depth_format'][()])
    if fmt != 'Y11':
        return None, f'depth format {fmt} is not the real depth stream -- unusable'

    already = bool(z['calibrated'][()]) if 'calibrated' in z else False
    if already:
        return stack, 'already calibrated at capture time'
    calib = load(path)
    if calib is None:
        return stack, 'RAW, uncorrected (no calibration saved -- depth is unusable)'
    return apply(stack, calib), 'raw values corrected on load'


def apply(depth_mm: np.ndarray, calib) -> np.ndarray:
    """
    Convert an array of raw reported mm into true mm.
    Zero means "no reading" both in and out, so it must survive the affine
    map untouched, and a correction that lands below zero is not a real
    distance either.
    """
    scale, offset = calib
    out = (depth_mm.astype(np.float32) - offset) / scale
    out[depth_mm <= 0] = 0.0
    out[out < 0] = 0.0
    return out
