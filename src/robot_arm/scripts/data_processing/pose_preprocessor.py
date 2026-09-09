#!/usr/bin/env python3
"""
pose_preprocessor.py
=====================
Pure NumPy — no ROS, no camera.

Pose-landmark analogue of preprocessor.py. Same sliding-window job, but
normalises around the SHOULDER (translation reference) and scales by
upper-arm length, shoulder-to-elbow distance (scale reference) instead of
the hand's wrist-centre + palm-size — shoulder/elbow are the natural
"anchor" for an arm the same way wrist/palm are for a hand.

Responsibilities
----------------
1. Normalize landmark coordinates to shoulder centre + upper-arm scale
   -> position and body-size invariant, matching motion_mapping.py's own
   assumption that only inter-joint VECTORS matter (translation/scale
   invariant), never absolute position.
2. Maintain a sliding window buffer (default 30 frames)
3. Return (window_size, POSE_FEATURE_DIM) sequence when buffer is full

Usage
-----
    pre = PosePreprocessor(window_size=30)
    for each frame:
        vec   = pose_feature_extractor.extract(lm_arr)   # (99,)
        ready, seq = pre.update(vec, lm_arr)
        if ready:
            # seq shape: (30, 99) -- feed to model
"""

import numpy as np
import collections
from .pose_feature_extractor import POSE_FEATURE_DIM, POSE_NUM_LANDMARKS

# MediaPipe Pose landmark indices (right arm) -- same as motion_mapping.py
R_SHOULDER = 12
R_ELBOW    = 14


class PosePreprocessor:
    """Sliding-window sequence builder with shoulder-centre normalisation."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._buffer     = collections.deque(maxlen=window_size)

    def reset(self):
        """Clear the sequence buffer."""
        self._buffer.clear()

    def update(self, features: np.ndarray,
               lm_arr: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        features : (POSE_FEATURE_DIM,)  raw feature vector from PoseFeatureExtractor
        lm_arr   : (33, 5)              landmark array used for normalisation

        Returns
        -------
        (ready: bool, sequence: np.ndarray | None)
        sequence shape: (window_size, POSE_FEATURE_DIM) when ready, else None
        """
        norm = self._normalize(features, lm_arr)
        self._buffer.append(norm)

        if len(self._buffer) == self.window_size:
            seq = np.array(self._buffer, dtype=np.float32)
            return True, seq

        return False, None

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(features: np.ndarray,
                   lm_arr: np.ndarray) -> np.ndarray:
        """
        Translate all x,y coordinates so the shoulder is (0, 0).
        Scale by shoulder-to-elbow distance (upper-arm length) so the
        vector is invariant to camera distance and body size.
        """
        f = features.copy()

        sh = lm_arr[R_SHOULDER, :2]
        sh_cx, sh_cy = sh[0], sh[1]

        el = lm_arr[R_ELBOW, :2]
        arm_scale = float(np.linalg.norm([el[0] - sh_cx, el[1] - sh_cy]))
        if arm_scale < 1e-6:
            arm_scale = 1.0

        for i in range(POSE_NUM_LANDMARKS):
            f[i * 3    ] = (f[i * 3    ] - sh_cx) / arm_scale
            f[i * 3 + 1] = (f[i * 3 + 1] - sh_cy) / arm_scale
            # z kept as-is (already relative depth)

        return f
