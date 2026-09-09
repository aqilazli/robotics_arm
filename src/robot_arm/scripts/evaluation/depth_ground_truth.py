#!/usr/bin/env python3
"""
depth_ground_truth.py — depth ground truth that is actually paired to the video.

Why this replaces kinematics_ground_truth.sample_ground_truth()
----------------------------------------------------------------
That function built its "ground truth" by drawing RANDOM joint angles from
JOINT_LIMITS and running forward kinematics. It never looked at the recorded
data, so sample i of the ground truth had no relationship to frame i of the
video. Measured on the arm track, that made the metric meaningless:

    real predictions      RMSE-kin 1206.009
    shuffled predictions  RMSE-kin 1205.255   (0.06% different)
    all-zero predictions  RMSE-kin  122.861   (ten times BETTER)

A metric that rewards predicting nothing measures nothing, so both RMSE
columns were withdrawn from the results.

What makes the depth comparison answerable now is that the depth sensor works. The Orbbec was
streaming Y12, which is not distance at all; Y11 is, its units are
centimetres, and it is calibrated (see depth_calibration.py). Calibrated
depth gives a real metric 3-D position for the wrist IN THE SAME FRAME as the
landmarks -- genuinely paired, which is exactly what was missing.

The comparison then becomes answerable as literally worded: reconstruct the
wrist in 3-D two ways and compare both against the measured depth.

    ground truth : wrist pixel -> calibrated depth -> pinhole back-projection
    candidate A  : kinematics reconstruction from the predicted landmark
    candidate B  : MediaPipe's own relative z, the scale-ambiguous baseline

Both are scored with the same rmse_depth_mm() against the same ground truth.

Honest limits of this ground truth
-----------------------------------
  * It is a MEASUREMENT, not a perfect reference. The sensor quantises: raw
    Y11 is integral centimetres, so depth resolves to ~10mm steps. Differences
    below that are not meaningful.
  * A wrist landmark can land on a pixel with no valid depth (out of the
    sensor's ~60cm minimum, or shadowed). Those frames are dropped rather
    than guessed at, and the count is reported.
  * It is the camera's view of the operator, not the robot's true state. The
    question here is reconstructing position from vision, so that is the right
    reference -- but it is not a motion-capture rig.
"""

import numpy as np

from motion_mapping import DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY, IMG_W, IMG_H


# Sensor floor. Y11 reads ~60cm at the closest; anything below is invalid, and
# anything beyond a few metres is not the operator.
MIN_VALID_MM = 550.0
MAX_VALID_MM = 4000.0


def wrist_ground_truth(lm_arr: np.ndarray,
                       depth_mm: np.ndarray,
                       wrist_idx: int):
    """
    Measured 3-D position of one landmark, in metres, camera frame.

    lm_arr    : (K, 5) landmarks, columns [x, y, z, visibility, presence],
                x/y normalised to [0, 1]
    depth_mm  : (H, W) CALIBRATED depth in millimetres
    wrist_idx : which landmark to measure (0 for hands, 16 for pose)

    Returns (3,) XYZ in metres, or None when the pixel carries no usable
    depth -- the caller drops that frame instead of inventing a value.
    """
    u = int(lm_arr[wrist_idx, 0] * IMG_W)
    v = int(lm_arr[wrist_idx, 1] * IMG_H)
    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None

    z = float(depth_mm[v, u])
    if not (MIN_VALID_MM <= z <= MAX_VALID_MM):
        return None

    z_m = z / 1000.0
    return np.array([(u - DEPTH_CX) * z_m / DEPTH_FX,
                     (v - DEPTH_CY) * z_m / DEPTH_FY,
                     z_m], dtype=np.float32)


def mediapipe_native_3d(lm_arr: np.ndarray, wrist_idx: int, scale_m: float = 1.0):
    """
    The BASELINE: MediaPipe's own relative z, back-projected.

    MediaPipe's z is scale-ambiguous -- roughly "depth relative to the hips"
    in normalised units, with no metric meaning. Treating it as metres is
    precisely the naive approach the depth comparison sets out to test, so it is deliberately
    NOT rescaled here beyond an optional caller-supplied factor.
    """
    u = int(lm_arr[wrist_idx, 0] * IMG_W)
    v = int(lm_arr[wrist_idx, 1] * IMG_H)
    z_m = float(lm_arr[wrist_idx, 2]) * scale_m
    return np.array([(u - DEPTH_CX) * z_m / DEPTH_FX,
                     (v - DEPTH_CY) * z_m / DEPTH_FY,
                     z_m], dtype=np.float32)


def build_paired_set(frames, depth_stack, wrist_idx):
    """
    Walk a recording and collect frames where a real depth measurement exists.

    frames      : list of (K, 5) landmark arrays, one per detected frame
    depth_stack : (N, H, W) calibrated depth, index-aligned with `frames`
    wrist_idx   : landmark to measure

    Returns (gt, mp, kept_indices, stats):
        gt  (M, 1, 3) measured ground truth, metres
        mp  (M, 1, 3) MediaPipe-native baseline, metres
        kept_indices  which original frame each row came from, so a model's
                      predictions can be lined up against these rows
        stats         dict: total, kept, dropped_no_depth
    """
    gt, mp, kept = [], [], []
    dropped = 0

    for i, lm in enumerate(frames):
        if i >= len(depth_stack):
            break
        g = wrist_ground_truth(lm, depth_stack[i], wrist_idx)
        if g is None:
            dropped += 1
            continue
        gt.append(g)
        mp.append(mediapipe_native_3d(lm, wrist_idx))
        kept.append(i)

    stats = {'total': len(frames), 'kept': len(kept), 'dropped_no_depth': dropped}
    if not kept:
        return None, None, [], stats

    return (np.asarray(gt, dtype=np.float32)[:, None, :],
            np.asarray(mp, dtype=np.float32)[:, None, :],
            kept, stats)
