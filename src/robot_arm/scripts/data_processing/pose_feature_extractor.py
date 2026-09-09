#!/usr/bin/env python3
"""
pose_feature_extractor.py
==========================
Pure NumPy — no ROS, no camera.

Pose-landmark analogue of feature_extractor.py — same flatten-to-vector
job, but for MediaPipe Pose's 33 body landmarks instead of Hands' 21
finger landmarks. Exists to build a temporal predictor for the ARM
(joint_L1-L4), the same way feature_extractor.py/predictor models build
one for the GRIPPER (joint_L6).

Input : lm_arr  np.ndarray (33, 5)   [x, y, z, visibility, presence]

Output: np.ndarray (99,)  flat feature vector per frame
        33 pose landmarks × 3 values (x, y, z)
"""

import numpy as np

POSE_FEATURE_DIM   = 99       # 33 pose landmarks × 3 (x, y, z)
POSE_NUM_LANDMARKS = 33


class PoseFeatureExtractor:
    """Extracts a flat feature vector from a single MediaPipe Pose landmark frame."""

    def extract(self, lm_arr: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        lm_arr : (33, 5)  MediaPipe Pose landmark array

        Returns
        -------
        np.ndarray (POSE_FEATURE_DIM,)
        """
        arr = lm_arr.astype(float, copy=True)

        features = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        for i in range(POSE_NUM_LANDMARKS):
            features[i * 3    ] = float(arr[i, 0])
            features[i * 3 + 1] = float(arr[i, 1])
            features[i * 3 + 2] = float(arr[i, 2])

        return features
