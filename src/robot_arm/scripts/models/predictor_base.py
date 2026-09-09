#!/usr/bin/env python3
"""
predictor_base.py
=================
Abstract base class for temporal hand pose predictors.

These models perform sequence-to-next-frame regression to compensate
pipeline latency:

    Input:  (window_size, feature_dim)  — sequence of past poses
    Output: (feature_dim,)              — predicted next pose

Implement this to add LSTM / GRU / Transformer predictors.
"""

from abc import ABC, abstractmethod
import numpy as np


class PredictorBase(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model identifier, e.g. 'LSTM', 'GRU', 'Transformer'."""

    @abstractmethod
    def build(self, window_size: int, feature_dim: int) -> None:
        """Instantiate model architecture (no weights loaded)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained weights from `path` (.h5 or SavedModel folder)."""

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict the next pose from a sequence of past poses.

        Parameters
        ----------
        sequence : (window_size, feature_dim)

        Returns
        -------
        np.ndarray (feature_dim,)
        """

    def summary(self):
        """Print model summary (optional)."""
