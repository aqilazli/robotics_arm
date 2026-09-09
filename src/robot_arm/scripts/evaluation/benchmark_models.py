#!/usr/bin/env python3
"""
benchmark_models.py
===================
Compare LSTM, GRU, and Transformer temporal predictors.

Measures all four evaluation metrics from the research proposal:
  • Inference time (ms)          — Objective 1
  • End-to-end latency (ms)      — Objective 1
  • RMSE (mm) for 3D depth       — Objective 2
  • MPJPE (mm) for pose accuracy — Objective 3

Usage
-----
# Architecture-only test (no trained weights, synthetic data):
    python3 evaluation/benchmark_models.py

# With trained model files and real test data:
    python3 evaluation/benchmark_models.py \
        --lstm        models/lstm_predictor.pt \
        --gru         models/gru_predictor.pt \
        --transformer models/transformer_predictor.pt \
        --data        test_dataset.npz

Dataset format (.npz)
---------------------
  X : (N, window_size, feature_dim)   input sequences
  y : (N, feature_dim)                ground-truth next poses
      feature_dim = 63  (21 hand landmarks × x,y,z)

Depth ground truth (optional, for RMSE — see kinematics_ground_truth.py)
  y_depth_gt  : (N, 1, 3)  wrist camera-space XYZ in metres, from kinematics simulation
  y_depth_mp  : (N, 1, 3)  MediaPipe native relative depth (baseline)
"""

import os
import sys
import time
import argparse
import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)

# ── MediaPipe Hands landmark indices used for MPJPE ───────────────────────────
# wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
ARM_JOINTS = [0, 4, 8, 12, 16, 20]

# Camera intrinsics for Orbbec Astra Pro (640×480) in metres
FX, FY = 570.0, 570.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480

N_WARMUP  = 10     # discard first N inferences (JIT compilation warm-up)
N_REPEAT  = 100    # number of timed runs per model


# ══════════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def mpjpe_mm(pred_sequences: np.ndarray,
             gt_sequences:   np.ndarray,
             feature_dim: int = 63,
             scale_factor: float = 1000.0) -> float:
    """
    Mean Per Joint Position Error on representative hand landmarks (mm).

    pred_sequences : (N, feature_dim)
    gt_sequences   : (N, feature_dim)
    feature_dim    : 63 = 21 hand landmarks × 3
    scale_factor   : 1000 to convert metres → mm (if coords are in metres)
                     1.0  if already in mm
    """
    errors = []
    for pred, gt in zip(pred_sequences, gt_sequences):
        # reshape to (21, 3)
        pred_lm = pred.reshape(feature_dim // 3, 3)
        gt_lm   = gt.reshape(feature_dim // 3, 3)

        for idx in ARM_JOINTS:
            diff = (pred_lm[idx] - gt_lm[idx]) * scale_factor
            errors.append(float(np.linalg.norm(diff)))

    return float(np.mean(errors))


def similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991) mapping src onto dst.

    Why this is needed at all: there is no camera-to-robot extrinsic
    calibration in this project. A kinematics reconstruction lives in the
    robot base frame, MediaPipe's native z lives in a scale-ambiguous camera
    frame, and the depth ground truth lives in the metric camera frame. A raw
    RMSE between any two of those is dominated by that unknown rigid offset
    and scale, so it reports frame misalignment rather than reconstruction
    quality -- it will happily rank a constant ahead of a real signal.

    Fitting one global rotation/scale/translation over the whole sequence
    removes exactly that nuisance and nothing else. It is fit once, not per
    frame, so it cannot absorb per-frame error: a reconstruction that does not
    track the wrist still scores badly afterwards. This is the standard
    Procrustes-aligned protocol used for 3-D pose error (PA-MPJPE).

    Always report alongside the controls in `reconstruction_controls`; the
    alignment makes the number meaningful but does not by itself prove the
    reconstruction carries any signal.

    Parameters
    ----------
    src, dst : (N, 3) point sets, correspondences row-matched.

    Returns
    -------
    np.ndarray (N, 3)  src mapped into dst's frame.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    dst = np.asarray(dst, dtype=float).reshape(-1, 3)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:       # keep it a rotation
        E[2, 2] = -1
    R = U @ E @ Vt
    var = (S ** 2).sum() / len(src)
    scale = np.trace(np.diag(d) @ E) / var if var > 1e-12 else 1.0
    return scale * (R @ src.T).T + (mu_d - scale * (R @ mu_s))


def reconstruction_controls(candidate: np.ndarray, gt: np.ndarray,
                            seed: int = 0) -> dict:
    """Sanity controls that say whether a reconstruction RMSE means anything.

    Returns the candidate's aligned RMSE next to two null models:

      shuffled : the same predictions in the wrong temporal order. Scores
                 close to the candidate mean the metric cannot tell correct
                 landmarks from scrambled ones, so it is not measuring
                 reconstruction.
      gt_mean  : predicting the constant average wrist position. A candidate
                 that does not beat this has learned nothing about motion.

    All values are millimetres. Report these; a bare RMSE pair is not
    interpretable on its own.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(candidate, dtype=float).reshape(-1, 3).copy()
    rng.shuffle(shuffled)
    gt3 = np.asarray(gt, dtype=float).reshape(-1, 3)
    const = np.broadcast_to(gt3.mean(0), gt3.shape).copy()
    return {
        'aligned':  rmse_depth_mm(similarity_align(candidate, gt3), gt3),
        'shuffled': rmse_depth_mm(similarity_align(shuffled, gt3), gt3),
        'gt_mean':  rmse_depth_mm(const, gt3),
    }


def rmse_depth_mm(pred_depth_3d: np.ndarray,
                  gt_depth_3d:   np.ndarray) -> float:
    """
    Root Mean Square Error between predicted and GT camera-space depth (mm).

    pred_depth_3d : (N, K, 3)  predicted XYZ in metres (K=1: wrist only, see kinematics_ground_truth.py)
    gt_depth_3d   : (N, K, 3)  ground-truth XYZ in metres
    """
    diff_mm = (pred_depth_3d - gt_depth_3d) * 1000.0   # metres → mm
    # RMSE over Z-axis depth only (most meaningful for depth accuracy)
    return float(np.sqrt(np.mean(diff_mm[..., 2] ** 2)))


def kinematics_depth_reconstruction(lm_2d_norm: np.ndarray,
                                    depth_mm: np.ndarray) -> np.ndarray:
    """
    Reconstruct 3D camera-space coords from 2D normalised landmarks + depth
    (the same pinhole back-projection feature_extractor._fuse_depth() uses).

    lm_2d_norm : (K, 2)  normalised image coords [0,1]
    depth_mm   : (K,)    depth for each point in mm

    Returns (K, 3) camera-space XYZ in metres.
    """
    n = lm_2d_norm.shape[0]
    xyz = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        z_m = depth_mm[i] / 1000.0
        u   = lm_2d_norm[i, 0] * IMG_W
        v   = lm_2d_norm[i, 1] * IMG_H
        xyz[i, 0] = (u - CX) * z_m / FX
        xyz[i, 1] = (v - CY) * z_m / FY
        xyz[i, 2] = z_m
    return xyz


# ══════════════════════════════════════════════════════════════════════════════
#  Timing
# ══════════════════════════════════════════════════════════════════════════════

def measure_inference_ms(model, sequence: np.ndarray,
                          n_warmup=N_WARMUP, n_repeat=N_REPEAT) -> dict:
    """
    Returns dict with keys: mean_ms, std_ms, min_ms, max_ms.
    """
    # warm-up (TF graph compilation, GPU warm-up)
    for _ in range(n_warmup):
        model.predict(sequence)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        model.predict(sequence)
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms':  float(np.std(times)),
        'min_ms':  float(np.min(times)),
        'max_ms':  float(np.max(times)),
    }


def measure_e2e_latency_ms(model, sequence: np.ndarray,
                            mediapipe_ms: float = 30.0,
                            n_warmup=N_WARMUP, n_repeat=N_REPEAT) -> float:
    """
    End-to-end latency = MediaPipe extraction time + model inference time.

    mediapipe_ms : typical MediaPipe Pose inference time on this machine.
                   Measured separately; default 30 ms is a reasonable estimate.
                   Override with --mp_latency argument.
    """
    inf = measure_inference_ms(model, sequence, n_warmup, n_repeat)
    return mediapipe_ms + inf['mean_ms']


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic data generator (used when no real dataset is provided)
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_data(n_samples=200, window_size=30, feature_dim=63,
                        seed=42) -> tuple:
    """
    Generate smooth sinusoidal pose sequences to simulate arm movement.
    Returns (X, y) where y[i] = X[i, -1] + small_delta  (one-step ahead).
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, n_samples + window_size)

    # base trajectory: arm swings in a smooth arc
    base = np.stack([
        0.05 * np.sin(t),           # x oscillates
        0.03 * np.cos(1.3 * t),     # y oscillates
        0.02 * np.sin(0.7 * t),     # z oscillates
    ], axis=-1)                      # (N+win, 3)

    # tile across all landmarks with small per-landmark variation
    seq_full = np.tile(base[:, np.newaxis, :], (1, feature_dim // 3, 1))
    seq_full += 0.005 * rng.standard_normal(seq_full.shape)
    seq_full  = seq_full.reshape(n_samples + window_size, feature_dim)

    X = np.stack([seq_full[i: i + window_size]
                  for i in range(n_samples)], axis=0).astype(np.float32)
    y = seq_full[window_size:].astype(np.float32)   # one step ahead

    # synthetic depth ground truth for the wrist (landmark 0) only, matching
    # kinematics_ground_truth.py's convention — scale to camera-like values (~0.5-1.5 m)
    wrist_y = y.reshape(n_samples, feature_dim // 3, 3)[:, 0:1, :]
    gt_3d = (wrist_y * 0.3 + 0.8).astype(np.float32)
    mp_3d = gt_3d + 0.05 * rng.standard_normal(gt_3d.shape).astype(np.float32)

    return X, y, gt_3d, mp_3d


# ══════════════════════════════════════════════════════════════════════════════
#  Main benchmark
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    from models.lstm_predictor        import LSTMPredictor
    from models.gru_predictor         import GRUPredictor
    from models.transformer_predictor import TransformerPredictor
    from models.baseline_predictor    import BaselinePredictor

    # ── data ─────────────────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        print(f'Loading test data from {args.data}')
        d   = np.load(args.data)
        X   = d['X'].astype(np.float32)
        y   = d['y'].astype(np.float32)
        # optional depth arrays
        gt_3d = d['y_depth_gt'].astype(np.float32) if 'y_depth_gt' in d else None
        mp_3d = d['y_depth_mp'].astype(np.float32) if 'y_depth_mp' in d else None
        window_size  = X.shape[1]
        feature_dim  = X.shape[2]
    else:
        print('No dataset provided — using synthetic data for benchmark.')
        window_size, feature_dim = 30, 63
        X, y, gt_3d, mp_3d = make_synthetic_data(
            n_samples=200, window_size=window_size, feature_dim=feature_dim)

    sample_seq = X[0]   # (window_size, feature_dim) — for latency timing

    # ── build/load models ─────────────────────────────────────────────────────
    models_cfg = [
        (LSTMPredictor,        args.lstm,        'LSTM'),
        (GRUPredictor,         args.gru,         'GRU'),
        (TransformerPredictor, args.transformer, 'Transformer'),
        (BaselinePredictor,    None,             'Baseline'),
    ]

    loaded_models = []
    for Cls, path, label in models_cfg:
        m = Cls(window_size=window_size, feature_dim=feature_dim)
        if label == 'Baseline':
            m.build()
            print('[Baseline] linear extrapolation, no weights needed — '
                  'the trivial baseline the other 3 models should be beating')
        elif path and os.path.exists(path):
            m.load(path)
            print(f'[{label}] weights loaded from {path}')
        else:
            m.build()
            if path:
                print(f'[{label}] WARNING: {path} not found — using random weights')
            else:
                print(f'[{label}] no weights path given — using random weights (latency test only)')
        loaded_models.append(m)

    # ── run inference on all test samples (for MPJPE) ─────────────────────────
    print('\nRunning inference on test set...')
    results = {}
    for m in loaded_models:
        preds = []
        for i in range(len(X)):
            preds.append(m.predict(X[i]))
        results[m.name] = np.array(preds, dtype=np.float32)

    # ── measure timing ────────────────────────────────────────────────────────
    print(f'Timing inference ({N_WARMUP} warm-up + {N_REPEAT} timed runs)...')
    timing = {}
    for m in loaded_models:
        timing[m.name] = measure_inference_ms(m, sample_seq)

    # ── compute metrics ───────────────────────────────────────────────────────
    print('\nComputing metrics...')
    metrics = {}
    for m in loaded_models:
        preds   = results[m.name]
        inf_ms  = timing[m.name]['mean_ms']
        e2e_ms  = args.mp_latency + inf_ms
        mpjpe   = mpjpe_mm(preds, y, feature_dim=feature_dim)

        # RMSE: compare kinematics-reconstructed depth vs MediaPipe native
        # Both baselines use the same depth source for fair comparison.
        # y_depth_gt/y_depth_mp are wrist-only (N, 1, 3) — see
        # kinematics_ground_truth.py — so reconstruction uses just the
        # predicted wrist landmark (index 0) too.
        if gt_3d is not None and mp_3d is not None:
            # RMSE of MediaPipe native depth (baseline)
            rmse_mp = rmse_depth_mm(mp_3d, gt_3d)
            # RMSE of kinematics reconstruction using the predicted wrist landmark
            pred_lm  = preds.reshape(len(preds), feature_dim // 3, 3)
            pred_2d  = pred_lm[:, 0:1, :2]
            pred_z_m = pred_lm[:, 0:1, 2]
            kin_3d   = np.zeros_like(gt_3d)
            for n in range(len(preds)):
                kin_3d[n] = kinematics_depth_reconstruction(
                    pred_2d[n], pred_z_m[n] * 1000.0)
            rmse_kin = rmse_depth_mm(kin_3d, gt_3d)
        else:
            rmse_mp  = float('nan')
            rmse_kin = float('nan')

        metrics[m.name] = {
            'inference_ms': inf_ms,
            'inf_std_ms':   timing[m.name]['std_ms'],
            'e2e_ms':       e2e_ms,
            'mpjpe_mm':     mpjpe,
            'rmse_kin_mm':  rmse_kin,
            'rmse_mp_mm':   rmse_mp,
        }

    # ── print comparison table ────────────────────────────────────────────────
    _print_table(metrics, args)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if args.out:
        _save_csv(metrics, args.out)
        print(f'\nResults saved to {args.out}')

    return metrics


def _print_table(metrics: dict, args):
    SEP = '─' * 78
    print(f'\n{SEP}')
    print(f'  BENCHMARK RESULTS  |  MediaPipe latency assumed: {args.mp_latency:.1f} ms')
    print(SEP)
    header = (f"{'Model':<14} {'Inf.(ms)':>10} {'±':>6} {'E2E(ms)':>10} "
              f"{'MPJPE(mm)':>11} {'RMSE-kin(mm)':>13} {'RMSE-MP(mm)':>12}")
    print(header)
    print('─' * 78)

    for name, m in metrics.items():
        rmse_kin = f"{m['rmse_kin_mm']:.2f}" if not np.isnan(m['rmse_kin_mm']) else 'N/A'
        rmse_mp  = f"{m['rmse_mp_mm']:.2f}"  if not np.isnan(m['rmse_mp_mm'])  else 'N/A'
        print(f"{name:<14} {m['inference_ms']:>10.2f} {m['inf_std_ms']:>6.2f} "
              f"{m['e2e_ms']:>10.2f} {m['mpjpe_mm']:>11.2f} "
              f"{rmse_kin:>13} {rmse_mp:>12}")

    print(SEP)
    print('  Inf.     = model inference time only')
    print('  E2E      = MediaPipe + inference (end-to-end latency)')
    print('  MPJPE    = Mean Per Joint Position Error on hand landmarks (wrist + 5 fingertips)')
    print('  RMSE-kin = kinematics-reconstructed wrist depth vs simulation ground truth')
    print('  RMSE-MP  = MediaPipe native wrist depth vs simulation ground truth (baseline)')
    print(f'{SEP}\n')


def _save_csv(metrics: dict, path: str):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Inference_ms', 'Inf_std_ms', 'E2E_ms',
                    'MPJPE_mm', 'RMSE_kin_mm', 'RMSE_MP_mm'])
        for name, m in metrics.items():
            w.writerow([name,
                        f"{m['inference_ms']:.4f}",
                        f"{m['inf_std_ms']:.4f}",
                        f"{m['e2e_ms']:.4f}",
                        f"{m['mpjpe_mm']:.4f}",
                        f"{m['rmse_kin_mm']:.4f}" if not np.isnan(m['rmse_kin_mm']) else '',
                        f"{m['rmse_mp_mm']:.4f}"  if not np.isnan(m['rmse_mp_mm'])  else ''])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Benchmark LSTM / GRU / Transformer pose predictors')
    p.add_argument('--lstm',        default=None,
                   help='Path to trained LSTM model (.h5)')
    p.add_argument('--gru',         default=None,
                   help='Path to trained GRU model (.h5)')
    p.add_argument('--transformer', default=None,
                   help='Path to trained Transformer model (.h5)')
    p.add_argument('--data',        default=None,
                   help='Test dataset .npz (X, y, optionally y_depth_gt, y_depth_mp)')
    p.add_argument('--mp_latency',  type=float, default=30.0,
                   help='MediaPipe inference time in ms to add to E2E latency (default 30)')
    p.add_argument('--out',         default='benchmark_results.csv',
                   help='Output CSV file path')
    args = p.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
