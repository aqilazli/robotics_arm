#!/usr/bin/env python3
"""
feature_extractor.py
====================
Pure NumPy — no ROS, no camera.

Input : lm_arr  np.ndarray (21, 5)   [x, y, z, visibility, presence]
        depth_mm np.ndarray (480,640) optional — Orbbec real depth in mm

Output: np.ndarray (63,)  flat feature vector per frame
        21 hand landmarks × 3 values (x, y, z)
        z = real depth_mm if depth available, else MediaPipe relative z

To swap the feature scheme:
  - edit extract() to return a different length vector
  - update FEATURE_DIM constant
  - retrain the model with the new features
"""

import numpy as np

FEATURE_DIM   = 63       # 21 hand landmarks × 3 (x, y, z)
NUM_LANDMARKS = 21
IMG_W, IMG_H  = 640, 480
MIN_VIS       = 0.5


class FeatureExtractor:
    """
    Extracts a flat feature vector from a single MediaPipe hand landmark frame.

    Usage
    -----
    fe  = FeatureExtractor()
    vec = fe.extract(lm_arr)              # shape (63,)
    vec = fe.extract(lm_arr, depth_mm)    # with real depth fusion
    """

    def extract(self, lm_arr: np.ndarray,
                depth_mm: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        lm_arr   : (21, 5)  MediaPipe hand landmark array
        depth_mm : (480, 640) float32  optional Orbbec depth in mm

        Returns
        -------
        np.ndarray (FEATURE_DIM,)
        """
        arr = lm_arr.astype(float, copy=True)

        # ── depth fusion: replace MediaPipe z with real mm ──────────────────
        if depth_mm is not None:
            arr = self._fuse_depth(arr, depth_mm)

        # ── flatten x, y, z for all 21 hand landmarks ───────────────────────
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        for i in range(NUM_LANDMARKS):
            features[i * 3    ] = float(arr[i, 0])
            features[i * 3 + 1] = float(arr[i, 1])
            features[i * 3 + 2] = float(arr[i, 2])

        return features

    # ── depth fusion ─────────────────────────────────────────────────────────

    @staticmethod
    def _fuse_depth(arr: np.ndarray,
                    depth_mm: np.ndarray) -> np.ndarray:
        """
        Replace column 2 (z) with real camera-space depth in metres for each
        visible landmark whose pixel falls within a valid depth reading.
        """
        # Orbbec Astra Pro 640×480 intrinsics
        fx, fy = 570.0, 570.0
        cx, cy = 320.0, 240.0

        for i in range(NUM_LANDMARKS):
            if arr[i, 3] < MIN_VIS:
                continue
            u = int(arr[i, 0] * IMG_W)
            v = int(arr[i, 1] * IMG_H)
            if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                continue
            z = float(depth_mm[v, u])
            if z <= 0 or z > 5000:
                continue
            z_m = z / 1000.0
            arr[i, 0] = (u - cx) * z_m / fx
            arr[i, 1] = (v - cy) * z_m / fy
            arr[i, 2] = z_m
        return arr
