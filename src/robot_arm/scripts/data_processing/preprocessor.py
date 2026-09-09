#!/usr/bin/env python3
"""
preprocessor.py
===============
Pure NumPy — no ROS, no camera.

Responsibilities
----------------
1. Normalize landmark coordinates to wrist centre + palm scale
   → position and hand-size invariant
2. Maintain a sliding window buffer (default 30 frames)
3. Return (window_size, FEATURE_DIM) sequence when buffer is full

Usage
-----
    pre = Preprocessor(window_size=30)
    pre.reset()

    for each frame:
        vec   = feature_extractor.extract(lm_arr)   # (63,)
        ready, seq = pre.update(vec, lm_arr)
        if ready:
            # seq shape: (30, 63) — feed to model
"""

import numpy as np
import collections
from .feature_extractor import FEATURE_DIM, NUM_LANDMARKS

# MediaPipe Hands landmark indices
WRIST           = 0
MIDDLE_FINGER_MCP = 9   # palm-size reference point


class Preprocessor:
    """
    Sliding-window sequence builder with wrist-centre normalisation.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._buffer     = collections.deque(maxlen=window_size)

    def reset(self):
        """Clear the sequence buffer (call between gestures if needed)."""
        self._buffer.clear()

    def update(self, features: np.ndarray,
               lm_arr: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        features : (FEATURE_DIM,)  raw feature vector from FeatureExtractor
        lm_arr   : (21, 5)         landmark array used for normalisation

        Returns
        -------
        (ready: bool, sequence: np.ndarray | None)
        sequence shape: (window_size, FEATURE_DIM) when ready, else None
        """
        norm = self._normalize(features, lm_arr)
        self._buffer.append(norm)

        if len(self._buffer) == self.window_size:
            seq = np.array(self._buffer, dtype=np.float32)   # (30, 63)
            return True, seq

        return False, None

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(features: np.ndarray,
                   lm_arr: np.ndarray) -> np.ndarray:
        """
        Translate all x,y coordinates so that the wrist is (0, 0).
        Scale by the wrist-to-middle-finger-MCP distance (palm size) so the
        vector is invariant to camera distance and hand size.
        """
        f = features.copy()

        # wrist centre in normalised image coords
        wr = lm_arr[WRIST, :2]
        wrist_cx, wrist_cy = wr[0], wr[1]

        # palm size = distance between wrist and middle-finger MCP
        mf = lm_arr[MIDDLE_FINGER_MCP, :2]
        palm_size = float(np.linalg.norm([mf[0] - wrist_cx,
                                          mf[1] - wrist_cy]))
        if palm_size < 1e-6:
            palm_size = 1.0

        # apply to every x, y (columns 0, 1 of each landmark triplet)
        for i in range(NUM_LANDMARKS):
            f[i * 3    ] = (f[i * 3    ] - wrist_cx) / palm_size
            f[i * 3 + 1] = (f[i * 3 + 1] - wrist_cy) / palm_size
            # z is kept as-is (depth is already relative)

        return f
