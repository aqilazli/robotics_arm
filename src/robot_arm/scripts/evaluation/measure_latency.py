#!/usr/bin/env python3
"""
measure_latency.py — does latency compensation actually work?

The claim in the project title is that temporal prediction compensates for
perception latency. Benchmarking the models does not test that: model
inference is ~0.3-0.6ms inside a loop running at a few Hz, so architecture
contributes a fraction of a percent of end-to-end delay. Comparing LSTM at
0.58ms against GRU at 0.29ms measures the one component that is not the
bottleneck.

What DOES test the claim is the lag between the operator moving and the robot
moving, measured with prediction on and with it off. If compensation works,
that lag falls.

Method
------
Two signals are recorded live:

    input   /arm/pose_landmarks -> operator's wrist x
    output  /joint_states       -> OUTPUT_JOINT angle (currently joint_L2)

Both are resampled onto a common time grid and cross-correlated. The shift
that maximises correlation is the system's tracking lag. Run once with the
predictor active and once with it OFF (publish 'none' to /ai_model), and the
difference is the compensation actually delivered.

Cross-correlation is used rather than a step-response test because it needs no
special gesture and averages over the whole recording, so a single mistimed
movement cannot dominate the result.

Usage
-----
    # with the pipeline already running:
    python3 evaluation/measure_latency.py --seconds 40 --label predictor-on
    python3 evaluation/measure_latency.py --seconds 40 --label predictor-off --model none

Move your arm continuously and smoothly while it records -- a varied signal is
what makes the correlation well conditioned. Holding still measures nothing.
"""

import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_mapping import (_L2_base_pitch,         # noqa: E402
                            ARM_CONTROL_MIN_VISIBILITY,
                            JOINT_LIMITS)

R_ELBOW = 14          # MediaPipe Pose right elbow
R_WRIST = 16          # MediaPipe Pose right wrist
# joint_L1 = atan2(dx, -dz) was tried first and abandoned: its second
# argument can be negative, so it has a genuine discontinuity, and with
# MediaPipe's noisy z estimate that discontinuity turned out to be very
# hard to stay away from in practice -- five recordings in a row came back
# with 40-63% of samples pinned to the joint's exact limit, deliberately
# angling the arm away from the camera made it WORSE (63%) not better, and
# the ambiguity/width checks correctly refused every one of them.
#
# joint_L2 = atan2(dy, sqrt(dx^2+dz^2)) cannot have this problem: its
# second argument is a magnitude, never negative, so the result is always
# confined to [-pi/2, pi/2] with no branch-cut nearby. Verified against the
# exact noise condition that broke L1 (dx, dz both noisy near zero): L1
# clamped 44% of samples, L2 clamped 0%. A simple raise/lower arm motion
# drives it directly.
OUTPUT_JOINT = 'joint_L2'
GRID_HZ = 100.0       # resample rate; finer than either source


class LatencyProbe(Node):
    def __init__(self):
        super().__init__('latency_probe')
        self.inp, self.out, self.e2e = [], [], []
        self.dropped_lowvis = 0
        self.create_subscription(Float32MultiArray, '/arm/pose_landmarks', self._pose, 50)
        self.create_subscription(JointState, '/joint_states', self._joints, 50)
        self.create_subscription(Float32, '/e2e_latency_ms', self._lat, 50)

        # TWO predictors exist and they are switched on DIFFERENT topics:
        #   /ai_model      -> inference_node, the HAND/gripper predictor
        #   /pose_ai_model -> robot_node,     the POSE/arm predictor
        # OUTPUT_JOINT -- the output signal measured here -- is driven by the POSE
        # track. Publishing only to /ai_model therefore toggled the gripper
        # while measuring the arm, leaving the arm configured identically in
        # both runs and making "predictor on vs off" a comparison of nothing.
        self.model_pub      = self.create_publisher(String, '/ai_model', 10)
        self.pose_model_pub = self.create_publisher(String, '/pose_ai_model', 10)
        self.active_pose    = None
        self.create_subscription(String, '/active_pose_ai_model',
                                 self._active_pose, 10)

    def _active_pose(self, msg):
        self.active_pose = msg.data.lower().strip()

    def _pose(self, msg):
        """Input signal = the joint_L2 angle these landmarks COMMAND.

        Not raw wrist position. joint_L2 is motion_mapping's own
        _L2_base_pitch over the elbow->wrist vector, so input and output are
        the same quantity, differing only by the pipeline delay -- what the
        cross-correlation is supposed to isolate. See the OUTPUT_JOINT
        comment for why L2 rather than L1.
        """
        n = 33 * 5
        if len(msg.data) < n:
            return
        a = np.asarray(msg.data[:n], dtype=float).reshape(33, 5)
        # Same gate robot_node applies, not a looser one. At 0.5 the probe
        # recorded commands the robot ignored (it gates at 0.78), so the
        # input series contained motion that the output never had a chance
        # to follow -- correlation gets diluted by frames the arm never saw.
        if any(a[i, 3] < ARM_CONTROL_MIN_VISIBILITY
               for i in (R_ELBOW, R_WRIST)):
            self.dropped_lowvis += 1
            return
        cmd = _L2_base_pitch(a[R_ELBOW, :3], a[R_WRIST, :3])
        self.inp.append((time.time(), float(cmd)))

    def _joints(self, msg):
        d = dict(zip(msg.name, msg.position))
        if OUTPUT_JOINT in d:
            self.out.append((time.time(), float(d[OUTPUT_JOINT])))

    def _lat(self, msg):
        self.e2e.append(float(msg.data))


def _resample(series, t0, t1):
    """Raw signal on a uniform grid. Standardisation happens per-lag instead."""
    t  = np.arange(t0, t1, 1.0 / GRID_HZ)
    ts = np.array([p[0] for p in series])
    vs = np.array([p[1] for p in series])
    return np.interp(t, ts, vs)


MIN_LAG_MS, MAX_LAG_MS = -100.0, 1500.0

# Below this the signals are not tracking each other and any argmax is noise.
MIN_PEAK_R = 0.5

# How much better the true peak must be than any rival lag at least
# AMBIGUITY_LAG_MS away, or the result is reported as ambiguous rather
# than as a number.
AMBIGUITY_MARGIN  = 0.05
AMBIGUITY_LAG_MS  = 150.0

# How far correlation must drop before the peak counts as resolved.
WIDTH_MARGIN = 0.03

# The near-peak plateau is allowed to be this many multiples of the INPUT
# signal's own median sample interval, not a fixed number of ms. A fixed
# 300ms cutoff seemed reasonable until two same-day recordings at the same
# ~350ms median pose interval landed at width/interval ~= 1.13 with
# completely different peak heights (0.726 -- a real, usable result -- and
# 0.208 -- noise): width alone, at a fixed rate, cannot tell those apart.
# Interpolating a sparse signal onto a fine grid smears the correlation
# peak over roughly one sample interval as a matter of course, so
# demanding a peak narrower than the interval itself rejects good data for
# being exactly as sharp as the sampling allows. Peak HEIGHT (MIN_PEAK_R,
# checked by the caller) is what actually separates signal from noise here
# -- width still catches a plateau far wider than one interval, which is a
# real multi-candidate ambiguity rather than an artefact of interpolation.
WIDTH_RESOLUTION_FACTOR = 2.0


def cross_correlation_lag_ms(inp, out):
    """Shift (ms) of `out` behind `inp` that maximises correlation.

    Uses a properly normalised cross-correlation: at every candidate lag the
    Pearson coefficient is computed over ONLY the overlapping region, with
    both slices re-standardised there.

    This is not a detail. The obvious implementation --
    np.correlate(b, a, 'full') on globally standardised signals -- sums fewer
    products as |lag| grows, so its value decays with lag purely from having
    less data to add up. argmax then sits near zero almost regardless of the
    true delay. Tested against injected delays it returned 80ms for a real
    400ms shift and 180ms for 250ms, i.e. it would have reported large
    compensation gains that were entirely artefacts of the bias.

    Returns (lag_ms, note, peak_r); lag_ms and peak_r are None on failure.
    """
    t0 = max(inp[0][0], out[0][0])
    t1 = min(inp[-1][0], out[-1][0])
    if t1 - t0 < 5.0:
        return None, 'overlapping window shorter than 5s', None

    a, b = _resample(inp, t0, t1), _resample(out, t0, t1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None, 'a signal is flat -- was the arm actually moving?', None

    lo = int(round(MIN_LAG_MS / 1000.0 * GRID_HZ))
    hi = int(round(MAX_LAG_MS / 1000.0 * GRID_HZ))
    # Need enough overlap for the coefficient to mean anything.
    min_overlap = max(int(2.0 * GRID_HZ), n // 4)

    best_lag, best_r = None, -2.0
    curve = {}
    for k in range(lo, hi + 1):
        if k >= 0:
            x, y = a[:n - k], b[k:]          # out delayed by k -> compare later out
        else:
            x, y = a[-k:], b[:n + k]
        if len(x) < min_overlap:
            continue
        xs, ys = x - x.mean(), y - y.mean()
        d = xs.std() * ys.std()
        if d < 1e-12:
            continue
        r = float((xs * ys).mean() / d)
        curve[k] = r
        if r > best_r:
            best_r, best_lag = r, k

    if best_lag is None:
        return None, 'no lag had enough overlapping samples', None

    # A high peak isn't enough by itself -- a genuine single delay produces
    # ONE lobe (correlation rises to a maximum, then falls away on both
    # sides). A recording with no resolvable delay instead produces a broad,
    # nearly flat plateau: several lags fit about equally well, and picking
    # the single highest among them is close to arbitrary. Seen in practice:
    # a "650ms" result whose correlation from 550-900ms sat entirely within
    # 0.006 of the reported peak.
    #
    # The two look different only in SHAPE, not in peak height, so what is
    # tested is whether a second local maximum exists far from the first --
    # a real second lobe, or a plateau (which is a wide band of near-tied
    # local maxima) -- rather than just checking distant correlation values,
    # which flags a broad-but-genuinely-single-peaked curve for the crime of
    # decaying gradually.
    best_ms = best_lag / GRID_HZ * 1000.0
    ks = sorted(curve)
    i0 = ks.index(best_lag)

    # Two distinct failure modes, both invisible from peak height alone:
    #
    # 1. A genuine second lobe far from the first -- a real competing delay
    #    hypothesis (e.g. a periodic signal, where a lag one period away
    #    fits almost as well). Tested via local maxima.
    #
    # 2. The peak's own lobe is too WIDE to pin down a lag at all -- seen in
    #    practice: a "650ms" result where correlation stayed within 0.006 of
    #    the peak across 550-900ms, a single connected plateau with no
    #    second lobe anywhere, but 350ms of near-tied lags around the
    #    reported number. This is the failure that actually occurred and
    #    the local-maxima check alone does not catch it, because there is
    #    only one lobe -- it is just too flat to resolve. Tested by walking
    #    outward from the peak while it stays within WIDTH_MARGIN.
    local_maxima = [
        k for i, k in enumerate(ks)
        if 0 < i < len(ks) - 1
        and curve[k] >= curve[ks[i - 1]] and curve[k] >= curve[ks[i + 1]]
    ]
    rival = max(
        (curve[k] for k in local_maxima
         if abs(k - best_lag) / GRID_HZ * 1000.0 >= AMBIGUITY_LAG_MS),
        default=-2.0)

    lo_i = i0
    while lo_i > 0 and curve[ks[lo_i - 1]] >= best_r - WIDTH_MARGIN:
        lo_i -= 1
    hi_i = i0
    while hi_i < len(ks) - 1 and curve[ks[hi_i + 1]] >= best_r - WIDTH_MARGIN:
        hi_i += 1
    width_ms = (ks[hi_i] - ks[lo_i]) / GRID_HZ * 1000.0

    in_ts = sorted(p[0] for p in inp)
    in_dt = np.diff(in_ts)
    median_interval_ms = (float(np.median(in_dt)) * 1000.0
                          if len(in_dt) else float('inf'))
    max_resolvable_width_ms = median_interval_ms * WIDTH_RESOLUTION_FACTOR

    if best_r - rival < AMBIGUITY_MARGIN:
        return (best_ms,
                f'AMBIGUOUS -- peak {best_r:.3f} at {best_ms:.0f}ms, but a '
                f'second local maximum {rival:.3f} exists >= '
                f'{AMBIGUITY_LAG_MS:.0f}ms away. Not a resolvable single '
                f'delay from this recording.',
                None)          # None peak -> caller's MIN_PEAK_R gate fires
    if width_ms > max_resolvable_width_ms:
        return (best_ms,
                f'AMBIGUOUS -- peak {best_r:.3f} at {best_ms:.0f}ms sits on '
                f'a {width_ms:.0f}ms-wide plateau, {width_ms/median_interval_ms:.1f}x '
                f'the {median_interval_ms:.0f}ms input sample interval '
                f'(limit {WIDTH_RESOLUTION_FACTOR:.1f}x). Not a resolvable '
                f'single delay -- try slower/steadier motion or better '
                f'visibility so more frames are kept.',
                None)
    return (best_ms, f'peak correlation {best_r:.3f} (resolution +/-'
            f'{width_ms/2:.0f}ms)', best_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=40.0)
    ap.add_argument('--label', default='run')
    ap.add_argument('--dump', default=None,
                    help='save raw input/output signals to this .npz for '
                         'offline diagnosis when the correlation is poor')
    ap.add_argument('--model', default=None,
                    help="switch predictor first: lstm|gru|transformer|none")
    args = ap.parse_args()

    rclpy.init()
    node = LatencyProbe()

    if args.model:
        want = args.model.lower().strip()
        deadline = time.time() + 8.0
        while time.time() < deadline:
            m = String(); m.data = want
            node.model_pub.publish(m)
            node.pose_model_pub.publish(m)
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.active_pose == want:
                break
        if node.active_pose == want:
            print(f'predictor set to: {want}  (arm predictor CONFIRMED)')
        else:
            print(f'  ABORT -- asked for arm predictor "{want}" but '
                  f'/active_pose_ai_model reports "{node.active_pose}".')
            print('  Refusing to measure: the run would not be the configuration')
            print('  you asked for. Is robot_node running?')
            rclpy.shutdown()
            return 1
        time.sleep(2.0)

    print(f'[{args.label}] recording {args.seconds:.0f}s -- MOVE YOUR ARM CONTINUOUSLY')
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.02)

    print(f'  pose samples  : {len(node.inp)}  '
          f'(dropped {node.dropped_lowvis} for low wrist visibility)')
    print(f'  joint samples : {len(node.out)}')

    if node.inp:
        # joint_L2's atan2 has a non-negative second argument, so it should
        # not clamp the way L1 did -- kept as a cheap sanity check in case
        # motion is extreme enough to hit the joint's actual physical limit.
        lo2, hi2 = JOINT_LIMITS['joint_L2']
        cmd = np.array([v for _, v in node.inp])
        clamped = np.mean((np.abs(cmd - hi2) < 1e-6) | (np.abs(cmd - lo2) < 1e-6))
        if clamped > 0.35:
            print(f'  WARNING: {clamped*100:.0f}% of input samples are pinned '
                  f'to the joint_L2 limit ({lo2:+.2f}/{hi2:+.2f} rad). Your '
                  f'arm reached its full raise/lower range very often -- try')
            print('  a smaller, steadier range of motion.')
    if node.out:
        jr = np.array([v for _, v in node.out])
        span = jr.max() - jr.min()
        print(f'  {OUTPUT_JOINT} range: {span:.4f} rad  '
              f'({"MOVING" if span > 0.02 else "BARELY MOVED -- see below"})')
        if span <= 0.02:
            print('    The arm did not follow you. Check, in this order:')
            print('      1. Gazebo: is the arm moving at all?')
            print('      2. Arm GUI: is control mode set to GESTURE, not MANUAL?')
            print('      3. Were you in frame with your right arm visible?')

    if len(node.inp) < 20 or len(node.out) < 20:
        print('  NOT ENOUGH DATA -- is the pipeline running and were you in frame?')
        rclpy.shutdown()
        return 1

    if args.dump:
        np.savez(args.dump,
                 inp_t=np.array([p[0] for p in node.inp]),
                 inp_v=np.array([p[1] for p in node.inp]),
                 out_t=np.array([p[0] for p in node.out]),
                 out_v=np.array([p[1] for p in node.out]),
                 label=args.label)
        print(f'  raw signals written to {args.dump}')

    lag, note, peak = cross_correlation_lag_ms(node.inp, node.out)
    dur  = node.inp[-1][0] - node.inp[0][0]
    rate = len(node.inp) / max(dur, 1e-9)
    print()
    print(f'=== {args.label} ===')
    if lag is None:
        print(f'  lag: could not measure -- {note}')
    elif peak is None:
        # cross_correlation_lag_ms found A number but flagged it AMBIGUOUS
        # (second lobe, or the peak's own lobe too wide to pin a lag down).
        # It still returns that number so it can be shown here, but it must
        # not land in the same branch as a trusted result: this used to
        # print "TRACKING LAG : 650.0 ms (AMBIGUOUS ...)", the number sitting
        # right next to its own disqualification where a skim reads only the
        # number.
        print(f'  LAG NOT MEASURABLE -- {note}')
        print(f'  (best single guess would be {lag:.0f}ms, but do not use it)')
        print('  Usually means: too few pose samples for how fast the arm')
        print(f'  moved -- perception ran at {rate:.1f} Hz. Move more slowly,')
        print('  or improve visibility so more frames are kept.')
    elif peak < MIN_PEAK_R:
        # A weak peak means the two signals are not tracking each other, so
        # the argmax is just the best of a set of bad fits. Reporting it as a
        # latency would be inventing a number.
        print(f'  LAG NOT MEASURABLE -- peak correlation {peak:.3f} '
              f'(need >= {MIN_PEAK_R})')
        print(f'  The commanded and actual {OUTPUT_JOINT} are not tracking '
              f'each other.')
        print('  Usual causes, in order:')
        print('    - the arm was not actually following you (check Gazebo)')
        print('    - you held still, or moved too little, during the run')
        print('    - control mode is set to MANUAL, so gestures are ignored')
        print(f'    - perception at {rate:.1f} Hz is too slow to resolve the lag')
    else:
        print(f'  TRACKING LAG        : {lag:8.1f} ms   ({note})')
        if rate < 10.0:
            print(f'  WARNING: perception {rate:.1f} Hz = {1000/rate:.0f} ms per '
                  f'frame, so this lag is only good to about that.')
    if node.e2e:
        e = np.array(node.e2e)
        print(f'  reported e2e latency: {e.mean():8.1f} ms  '
              f'(median {np.median(e):.1f}, p95 {np.percentile(e, 95):.1f}, n={len(e)})')
    print(f'  perception rate     : {rate:8.1f} Hz  '
          f'({len(node.inp)} samples over {dur:.0f}s)')
    print()
    print('  Run again with --model none to get the uncompensated baseline.')
    print('  Compensation delivered = (lag with predictor off) - (lag with it on).')

    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
