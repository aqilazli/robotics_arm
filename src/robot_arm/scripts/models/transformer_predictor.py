#!/usr/bin/env python3
"""
transformer_predictor.py  —  PyTorch
=====================================
Transformer temporal pose predictor for latency compensation.

Architecture
------------
Input  (batch, window_size, feature_dim)
Linear(d_model=128)          ← input projection
+ sinusoidal positional encoding
2 × TransformerEncoderLayer  (4 heads, FFN=256, GELU, dropout=0.1)
GlobalAveragePool over time
Linear(feature_dim)          ← predict next pose

Save format: .pt  (torch.save)

Training
--------
    python3 transformer_predictor.py --train \
        --data  dataset.npz \
        --out   transformer_predictor.pt \
        --epochs 100
"""

import math
import numpy as np
import torch
import torch.nn as nn
from .predictor_base import PredictorBase

DEFAULT_WINDOW = 30
DEFAULT_FDIM   = 63
D_MODEL        = 128
NUM_HEADS      = 4
FFN_DIM        = 256
NUM_LAYERS     = 2
DROPOUT        = 0.1


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding (seq_len, d_model)."""
    pe  = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (T, D)


class _TransformerNet(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = D_MODEL,
                 nhead: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
                 ffn_dim: int = FFN_DIM, dropout: float = DROPOUT,
                 window_size: int = DEFAULT_WINDOW):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)

        # register sinusoidal PE as buffer (non-trainable, saved with model)
        self.register_buffer('pe', _sinusoidal_pe(window_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, feature_dim)

    def forward(self, x):
        x = self.input_proj(x) + self.pe   # (B, T, d_model)
        x = self.encoder(x)                # (B, T, d_model)
        x = x.mean(dim=1)                  # global avg pool → (B, d_model)
        return self.fc(x)                  # (B, feature_dim)


class TransformerPredictor(PredictorBase):
    """Transformer sequence-to-next-frame pose predictor (PyTorch)."""

    def __init__(self, window_size: int = DEFAULT_WINDOW,
                 feature_dim: int = DEFAULT_FDIM):
        self._ws     = window_size
        self._fd     = feature_dim
        self._model  = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def name(self) -> str:
        return 'Transformer'

    def build(self, window_size: int = None, feature_dim: int = None) -> None:
        ws = window_size or self._ws
        fd = feature_dim or self._fd
        self._model = _TransformerNet(fd, window_size=ws).to(self._device)
        self._model.eval()

    def load(self, path: str) -> None:
        # state_dict, not a whole pickled module -- see lstm_predictor.py's
        # load() for why (breaks loading from any script other than the one
        # that trained it, otherwise).
        self.build()
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
        print(f'[TransformerPredictor] loaded {path}')

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('Call build() or load() first')
        x = torch.from_numpy(sequence[np.newaxis]).float().to(self._device)
        with torch.no_grad():
            return self._model(x).cpu().numpy()[0]

    def summary(self):
        if self._model:
            print(self._model)
            total = sum(p.numel() for p in self._model.parameters())
            print(f'Total parameters: {total:,}')


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path, out_path, epochs=100, batch_size=32, val_split=0.15, lr=1e-3):
    from torch.utils.data import TensorDataset, DataLoader, random_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TransformerPredictor] training on {device}')

    data = np.load(data_path)
    X    = torch.from_numpy(data['X'].astype(np.float32))
    y    = torch.from_numpy(data['y'].astype(np.float32))
    print(f'Dataset: X={tuple(X.shape)}  y={tuple(y.shape)}')

    dataset = TensorDataset(X, y)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = _TransformerNet(X.shape[2], window_size=X.shape[1]).to(device)
    print(model)

    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience_cnt = 0
    PATIENCE = 15

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'Early stop at epoch {epoch}')
                break

    print(f'[TransformerPredictor] best val loss={best_val:.6f}  saved → {out_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train',  action='store_true')
    p.add_argument('--data',   default='dataset.npz')
    p.add_argument('--out',    default='transformer_predictor.pt')
    p.add_argument('--epochs', type=int,   default=100)
    p.add_argument('--batch',  type=int,   default=32)
    p.add_argument('--lr',     type=float, default=1e-3)
    args = p.parse_args()

    if args.train:
        train(args.data, args.out, args.epochs, args.batch, lr=args.lr)
    else:
        m = TransformerPredictor()
        m.build()
        m.summary()
