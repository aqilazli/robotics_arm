#!/usr/bin/env python3
"""
baseline_predictor.py  —  PyTorch not required, no training
==============================================================
Constant-velocity (linear extrapolation) predictor — the trivial baseline
LSTM/GRU/Transformer should be beating.

Why this exists
----------------
Without this, the thesis compares three neural nets against each other but
never against "don't bother training anything." A reviewer's first question
tends to be exactly that. This has zero trainable parameters and needs no
dataset.npz at all to "train" — it just estimates velocity from the last
few frames of whatever window it's given and extrapolates one step forward.

Architecture
------------
predicted = last_frame + mean(frame-to-frame deltas over the last K frames)

Usage
-----
    from models.baseline_predictor import BaselinePredictor
    m = BaselinePredictor(window_size=30)
    m.build()
    pred = m.predict(sequence)   # sequence: (30, feature_dim) -> (feature_dim,)
"""

import numpy as np
from .predictor_base import PredictorBase

DEFAULT_WINDOW = 30
DEFAULT_FDIM   = 63
DEFAULT_K      = 5   # number of recent frame-to-frame deltas to average


class BaselinePredictor(PredictorBase):
    """Linear extrapolation from recent velocity. No training, no weights file."""

    def __init__(self, window_size: int = DEFAULT_WINDOW,
                 feature_dim: int = DEFAULT_FDIM, k: int = DEFAULT_K):
        self._ws = window_size
        self._fd = feature_dim
        self._k  = k

    @property
    def name(self) -> str:
        return 'Baseline'

    def build(self, window_size: int = None, feature_dim: int = None) -> None:
        if window_size is not None:
            self._ws = window_size
        if feature_dim is not None:
            self._fd = feature_dim

    def load(self, path: str) -> None:
        # nothing to load — no trainable parameters
        pass

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        sequence : (window_size, feature_dim)
        returns  : (feature_dim,) — last frame + average recent velocity
        """
        k = min(self._k, sequence.shape[0] - 1)
        if k < 1:
            return sequence[-1].copy()

        deltas = sequence[-k:] - sequence[-k - 1:-1]   # (k, feature_dim)
        velocity = deltas.mean(axis=0)
        return (sequence[-1] + velocity).astype(np.float32)

    def summary(self):
        print(f'BaselinePredictor: constant-velocity extrapolation, '
              f'k={self._k}, 0 trainable parameters')


if __name__ == '__main__':
    # architecture preview / smoke test — no args needed, nothing to train
    m = BaselinePredictor()
    m.build()
    m.summary()
    dummy = np.random.randn(DEFAULT_WINDOW, DEFAULT_FDIM).astype(np.float32)
    out = m.predict(dummy)
    print('predict() output shape:', out.shape)
