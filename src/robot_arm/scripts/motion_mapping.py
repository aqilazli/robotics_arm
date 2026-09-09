#!/usr/bin/env python3
"""
motion_mapping.py  —  Pure math library.  No ROS, no camera.
=============================================================

Maps MediaPipe POSE body landmarks (shoulder/elbow/wrist) directly to arm
joint angles L1-L4 — the robot's elbow bends because the operator's elbow
bends, no inverse kinematics involved. This is the ARM half of the hybrid
control scheme; the GRIPPER (L7) is driven separately from MediaPipe HANDS
finger landmarks via robot_control/ik_solver.gripper_from_hand() — hand
landmarks have no shoulder/elbow, so they were never suited to driving the
arm itself, only what a hand actually does: open and close.

landmarks_to_joint_angles()'s own L7 value (from pose fingertip landmarks,
a rough approximation) is overridden downstream in robot_node.py by the
real hand-landmark gripper mapping.

Input:  lm_arr  np.ndarray shape [33, 5]  (MediaPipe Pose landmarks)
        Columns: [x, y, z, visibility, presence]
        x, y are MediaPipe normalized image coords [0,1].
        z   is either MediaPipe's relative hip-depth (default) or
            real camera-space depth in metres after fuse_depth().

Output: dict  {joint_name: angle_radians}

Joint-to-human mapping (shoulder = robot base, wrist = robot end)
  joint_L1  Base yaw       ← upper-arm horizontal angle (shoulder→elbow in XZ)
  joint_L2  Base pitch     ← upper-arm elevation angle  (shoulder→elbow vertical)
  joint_L3  Elbow flexion  ← angle at human elbow (shoulder/elbow/wrist)
  joint_L4  Forearm roll   ← pronation/supination estimated from elbow plane
  joint_L5  Wrist pitch    ← angle between forearm and hand direction, world-vertical
  joint_L6  Wrist roll     ← knuckle-line angle around the forearm axis, world-vertical
  joint_L7  Gripper        ← rough pose-fingertip estimate; overridden by
                              ik_solver.gripper_from_hand() in robot_node.py

  L1/L2 used to be driven by elbow->wrist (the FOREARM segment), on the
  reasoning "elbow is near the robot's base link". That reasoning was
  wrong: the robot's base is physically fixed, but a human elbow moves
  constantly -- it swings through space every time the shoulder rotates
  AND every time the elbow itself bends. Proven with the shoulder and
  elbow held at literally the same 3-D position throughout (i.e. isolating
  pure elbow flexion, zero shoulder movement): the old elbow->wrist L1
  swung through its ENTIRE 195-degree range from elbow bending alone.
  L1/L2 now read shoulder->elbow (the UPPER-ARM segment) instead, which a
  human keeps fixed in exactly the case that broke the old version:
  bending the elbow does not move the shoulder or the upper arm, so it
  does not move shoulder->elbow at all (0.000000 rad in the same test).
  This is also the geometrically correct correspondence for a 6-DOF arm
  (base yaw + shoulder pitch + elbow, then a 3-DOF wrist): the human
  shoulder is what the robot's base and first pitch joint should track,
  not the elbow.

Landmark indices used  (right arm — arm-to-fist, no fingers)
  12 R_SHOULDER  14 R_ELBOW  16 R_WRIST
"""

import math
import numpy as np

# ── Landmark indices ─────────────────────────────────────────────────────────
NOSE        = 0
L_SHOULDER  = 11
R_SHOULDER  = 12
R_ELBOW     = 14
R_WRIST     = 16
R_PINKY_TIP = 18   # right pinky fingertip
R_INDEX_TIP = 20   # right index fingertip
R_THUMB_TIP = 22   # right thumb tip

# Gripper thresholds (fingertip-to-wrist distance / forearm length)
_GRIPPER_CLOSED_NORM = 0.15   # below this → fist (gripper closed)
_GRIPPER_OPEN_NORM   = 0.40   # above this → flat hand (gripper open)

# ── Joint limits (radians) — from arm_gui.py JOINTS list ────────────────────
JOINT_LIMITS = {
    'joint_L1': (-1.70,  1.70),
    'joint_L2': (-0.98,  1.00),
    'joint_L3': (-2.00,  1.30),
    'joint_L4': (-2.00,  2.00),
    'joint_L5': (-2.10,  2.10),
    'joint_L6': (-3.10,  3.10),
    # joint_L7 is PRISMATIC: metres of jaw travel, not radians. It drives the
    # L7_R jaw; the opposing L7_L jaw is a URDF <mimic> of it, so both move as
    # one degree of freedom and cannot drift apart.
    # Values below are metres, not radians:
    # 0.0 = closed, 0.011 = fully open (the jaw's modelled position in
    # L6.STL, which sat 11mm off centre). Every other entry here is an angle,
    # so anything treating this dict as uniformly angular is wrong.
    'joint_L7_R': ( 0.00,  0.011),
    'joint_L7_L': ( 0.00,  0.011),
}
JOINT_NAMES = ['joint_L1', 'joint_L2', 'joint_L3',
               'joint_L4', 'joint_L5', 'joint_L6',
               'joint_L7_R', 'joint_L7_L']

# ── Depth camera intrinsics (Orbbec Astra Pro, 640×480) ─────────────────────
DEPTH_FX = 570.0
DEPTH_FY = 570.0
DEPTH_CX = 320.0
DEPTH_CY = 240.0
IMG_W    = 640
IMG_H    = 480

MIN_VISIBILITY = 0.3

# Separate, much stricter bar for landmarks that actually DRIVE the arm.
# MediaPipe's `visibility` is optimistic: with an operator sitting at the desk
# but their arm out of shot, measured medians were shoulder 0.992 (real) but
# elbow 0.356 and wrist 0.236 -- invented. Those guesses jitter every frame,
# and at a 0.3 bar 13 of 74 such frames slipped through, so the robot chased
# limbs the camera could not see.
#
# 0.78 sits in the measured gap between invented and real landmarks, using
# ELBOW's own separation:
#
#   operator ABSENT   elbow median 0.356, max 0.746 | wrist median 0.236, max 0.649
#   operator PRESENT  elbow median 0.829            | wrist median 0.805
#
# 0.75 was inside the noise (13 of 74 absent frames passed, the arm twitched).
# 0.85 overshot the other way -- only 10 of 80 REAL frames passed, so the arm
# stopped following at all. 0.78 clears the absent-frame ceiling of 0.746 and
# still admits the bulk of genuine tracking.
#
# The gap is narrow, so this is a trade-off rather than a clean separation:
# raising it costs responsiveness, lowering it lets phantom motion back in.
#
# MIN_VISIBILITY stays 0.3 for depth fusion, where a marginal landmark is
# still worth a depth sample and a bad one is discarded downstream anyway.
ARM_CONTROL_MIN_VISIBILITY = 0.78

# Wrist gets its OWN, lower bar instead of reusing 0.78. That number was
# derived from ELBOW's absent-ceiling (0.746) alone; wrist's absent-ceiling
# is meaningfully lower (0.649, same calibration run above), so 0.78 was
# never actually validated for wrist -- it just inherited elbow's number.
# The claim this replaced ("a genuinely raised arm reads well above 0.9")
# does not hold for every gesture: watched live, raising/straightening the
# arm made wrist visibility crash to 0.33-0.46 for most of the motion
# (self-occlusion/framing during that specific reach, not an absent arm --
# elbow and shoulder stayed confidently high throughout), so 0.78 was
# silently rejecting nearly every frame of a real, intended gesture.
# Reported directly: "i raise my hand but robots arm not stretch".
#
# Same margin-above-ceiling method as 0.78 itself (elbow: 0.746 + 0.034):
# wrist 0.649 + 0.034 = 0.683. This still only recovers PART of the
# reported motion -- checked against the actual captured session, most
# frames sat at 0.33-0.46, well under even this relaxed bar. The rest is
# a genuine tracking limit for that pose (self-occlusion/framing), not
# something a threshold number can fix without letting phantom-limb
# noise back in -- see the ABSENT wrist ceiling (0.649) directly above.
ARM_CONTROL_MIN_VISIBILITY_WRIST = 0.683


# ══════════════════════════════════════════════════════════════════════════════
#  Depth fusion
# ══════════════════════════════════════════════════════════════════════════════

def fuse_depth(lm_arr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """
    Replace MediaPipe's relative Z with real camera-space 3-D coords.

    For each visible landmark the normalized (x, y) is projected to a pixel,
    the depth is looked up, and columns 0-2 are overwritten with metric
    camera-space (X, Y, Z) in metres:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth_mm[v, u] / 1000

    Returns a new array (original is not modified).
    Landmarks with no valid depth keep their MediaPipe coords -- EXCEPT the
    arm-control triplet (shoulder/elbow/wrist), which is all-or-nothing (see
    below).

    Found live, straight from real captured data, chasing "when i move right
    it doesn't rotate the full path" and "thrust forward is crazy, all
    misalign": shoulder fuses successfully almost every frame (large, slow,
    easy depth target) while elbow/wrist routinely fail their own depth
    lookup (small, fast-moving, frequently at a depth-image edge or briefly
    below MIN_VISIBILITY) and silently keep raw MediaPipe coordinates
    instead -- normalized [0,1] image-plane x/y with MediaPipe's own
    hip-relative z, a completely different frame from the shoulder's fused
    metres. _L1_base_yaw and friends then compute dx/dz between a
    metres-scale shoulder and a normalized-scale elbow, which is not a
    small numeric error, it's two incompatible coordinate systems averaged
    together -- confirmed directly: shoulder.z stayed a plausible, stable
    0.58-0.70 m across a real capture while elbow.z swung from -2.5 to
    +1.0 in the SAME window (physically impossible for a shoulder-anchored
    forearm), matching exactly the elbow/wrist reverting to raw coords
    almost every other frame. That explains both symptoms: the yaw angle
    computed from a mismatched pair is dominated by whichever axis's raw
    numbers happen to be larger, not by the real physical swing, so real
    motion barely moves the joint (never reaches full range); and every
    frame where the fusion outcome flips (fused one moment, raw the next)
    injects a discontinuous jump into L1/L2/L3/L5 simultaneously, which is
    exactly what "crazy, all misalign" looks like.

    Fix: shoulder/elbow/wrist fuse as a unit. If any one of them fails,
    all three fall back to raw MediaPipe coordinates together, so whichever
    frame the downstream geometry runs in, both ends of every vector it
    builds (shoulder->elbow, shoulder->wrist) are always in the SAME frame.
    Costs some depth accuracy on frames where only one of the three
    genuinely failed; that is a far smaller error than mixing frames
    outright, and MIN_VISIBILITY stays at 0.3 here (not
    ARM_CONTROL_MIN_VISIBILITY) -- this isn't about raising the bar for
    when to trust a landmark, it's about not blending two landmarks that
    each individually cleared whatever bar was used.
    """
    arr = lm_arr.astype(float, copy=True)
    fused = np.zeros(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        if arr[i, 3] < MIN_VISIBILITY:
            continue
        u = int(arr[i, 0] * IMG_W)
        v = int(arr[i, 1] * IMG_H)
        if not (0 <= u < IMG_W and 0 <= v < IMG_H):
            continue
        z = float(depth_mm[v, u])
        if z <= 0 or z > 5000:          # invalid / out-of-range
            continue
        z_m = z / 1000.0
        arr[i, 0] = (u - DEPTH_CX) * z_m / DEPTH_FX
        arr[i, 1] = (v - DEPTH_CY) * z_m / DEPTH_FY
        arr[i, 2] = z_m
        fused[i] = True

    if arr.shape[0] > max(R_SHOULDER, R_ELBOW, R_WRIST):
        triplet = (R_SHOULDER, R_ELBOW, R_WRIST)
        if not all(fused[i] for i in triplet):
            for i in triplet:
                arr[i, :3] = lm_arr[i, :3]

    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _v(arr: np.ndarray, idx: int) -> np.ndarray:
    """Extract xyz of landmark idx as a 3-vector."""
    return arr[idx, :3].copy()


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _angle_at(proximal, joint, distal) -> float:
    """Angle at `joint` between joint→proximal and joint→distal."""
    return _angle_between(proximal - joint, distal - joint)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _remap(val, in_lo, in_hi, out_lo, out_hi) -> float:
    if abs(in_hi - in_lo) < 1e-9:
        return out_lo
    t = (val - in_lo) / (in_hi - in_lo)
    return _clamp(out_lo + t * (out_hi - out_lo), out_lo, out_hi)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-joint mappings
# ══════════════════════════════════════════════════════════════════════════════

# Below this horizontal magnitude (metres), shoulder->elbow's XZ direction
# is dominated by landmark noise rather than real arm orientation -- an
# arm hanging near-vertically at rest ("robot home" stance) has dx and dz
# BOTH close to zero, and atan2 near the origin is extremely noise-
# sensitive: measured (2cm of synthetic landmark noise, matching typical
# depth-fused jitter) std of 73 degrees at 1cm magnitude, falling to 9
# degrees by 10cm. landmarks_to_joint_angles() reports this magnitude
# alongside joint_L1 so the caller can hold the last value near the
# singularity instead of chasing noise -- this module is pure math with
# no state of its own to hold a "last value" in.
#
# 0.10 was not nearly enough margin. Reported directly: "whn i dropped
# my arm fully, the l1 robot will turn to right" -- a live capture
# through an actual raise-then-drop motion showed magnitude collapsing
# to 0.02-0.14 while the arm was down (straddling the old 0.10 line),
# and in that band the angle was not scattered noise, it consistently
# read -140 to -150 degrees for several straight seconds -- close
# enough to L1_YAW_RIGHT_RANGE_DEG's -135 bound to clamp to a full
# "turn right" almost every time, for this operator's specific
# drop-arm geometry. Genuine intentional swings measured separately
# stayed at 0.13 and above even during transitions. 0.18 sits above the
# entire observed degenerate band with margin, comfortably below real
# swing magnitudes.
L1_MIN_MAG = 0.18

# L2/L3 share L1's underlying shoulder->elbow magnitude (see
# robot_control/robot_node.py's _apply_smoothing skip-dict) but do NOT
# share its threshold: gating them at the full 0.18 held all three
# joints frozen together for 10+ second stretches during completely
# ordinary movement, reported directly as "the robot arm is not follow
# my hand. rotation l1 also broken, and the l3 is not following". A live
# capture of the actual degenerate-geometry bug this gate exists for
# (motion_mapping's own _L2_base_pitch going to 65-85 degrees of pure
# noise) showed that behaviour concentrated below ~0.10 -- the 0.10-0.18
# band still ran somewhat elevated (55-65 degrees) but nowhere near the
# +/-90 degree edge that clamps L2 to a JOINT_LIMITS extreme, so it does
# not need to be held, just smoothed same as any other noisy frame
# (median-5 + EMA + step-cap already do that). Lower than L1_MIN_MAG on
# purpose -- L1's 0.18 is tuned for A DIFFERENT failure (a dropped-arm
# angle landing on L1's own yaw limit almost every time, see that
# constant's own declaration), not this one.
L2_MIN_MAG = 0.10

# Same failure mode, different geometry: L4's "elbow plane" vector
# (elbow_perp below) vanishes as the arm approaches full extension --
# upper arm and forearm nearly colinear, so which way the elbow points
# relative to the forearm axis is ill-defined. Measured (1.5cm synthetic
# landmark noise): 49.5deg of noise-driven roll std at elbow_perp=0.017
# (arm almost straight), still 18.4deg at 0.26, down to ~8deg by 0.42.
L4_MIN_MAG = 0.35

# L6 reads a plane-normal magnitude now (cross product of two edge
# vectors -- see _l6_raw_angle), not the simple 2-point difference this
# threshold was originally set for. A cross product's magnitude is on an
# entirely different scale (roughly length^2, not length): real footage
# put it at 0.001-0.007, so the OLD threshold (0.03) sat above every
# frame ever recorded and would have held L6 permanently. Re-swept the
# same way (1.5cm synthetic landmark noise): std fell from 98deg at
# 0.0008 to 18deg by 0.0134, no sharp elbow, so this signal stays
# noisier at achievable magnitudes than L1/L4's. 0.0012 picked from real
# footage, not the sweep alone: held only 1.5-8.7% of frames across 3
# recordings, versus the old signal's 36-43% -- a looser gate is
# reasonable here since the 3-point plane fit already reduces noise
# before this check ever runs (see _l6_raw_angle's docstring).
L6_MIN_MAG = 0.0012

# Human forearm pronation/supination -- the physical motion joint_L6
# represents -- is commonly cited around 75-90 degrees each way from
# neutral, nowhere near the +/-180 degrees atan2 can mathematically
# output. Clipping the raw angle straight into JOINT_LIMITS['joint_L6']
# (+/-3.10 rad, ~+/-177.7 deg) meant a real, comfortable wrist twist
# only ever reached about half the joint's actual range, while the
# raw math's own not-quite-linear behaviour near its own extremes (see
# _L6_wrist_roll's docstring) showed up as a modest rotation appearing
# to run disproportionately far. WRIST_ROLL_HUMAN_RANGE is remapped
# onto the full joint range instead -- see _L6_wrist_roll.
WRIST_ROLL_HUMAN_RANGE = math.radians(90)

# Same idea for L5 (wrist pitch, front/back bend): JOINT_LIMITS allows
# +/-2.10 rad (~+/-120 deg, 240 deg total), far past what a human wrist's
# flexion/extension actually does (~+/-90 deg, 180 deg total is the
# commonly cited figure -- and what was asked for directly: "this dof is
# only front and back with 180 degree"). _L5_wrist_pitch used to clip
# straight into the full joint range like L6 originally did.
WRIST_PITCH_HUMAN_RANGE = math.radians(90)

# L1 (base yaw / shoulder horizontal swing) runs the OPPOSITE direction
# from L5/L6 above: JOINT_LIMITS['joint_L1'] is +/-1.70 rad (~+/-97 deg,
# ~195 deg total) -- NARROWER than a human shoulder's real horizontal
# sweep, given directly as "around 270 degree" (~+/-135 deg). L5/L6 clip
# a NARROW human range into a WIDE joint range (remapping there caused
# ~2x amplification and was rejected, see _L6_wrist_roll); L1 has to fit
# a WIDE human range into a NARROW joint range instead, so the fix runs
# the other way: without remapping, a full natural shoulder swing hits
# JOINT_LIMITS long before the operator reaches their own limit, and
# every degree past that point is silently thrown away -- 1:1 gain up to
# the robot's limit, then completely flat. Reported directly: "angle is
# not accurate enough". BASE_YAW_HUMAN_RANGE is instead the _remap()
# input span (see _L1_base_yaw), same mechanism L2/L3 already use for
# their own shoulder-pitch/elbow-flex ranges -- proportional compression
# across the whole gesture, not a hard clip partway through it.
#
# Split into two independent ranges, right and left, not one symmetric
# span -- reported directly: "l1 cannot rotate to the left fully", and a
# clean, stable live capture (steady for many consecutive frames, not a
# noise spike) measured the operator's genuine full-left swing at only
# ~25 degrees of raw angle, against ~135 degrees comfortably reached
# swinging right. A single symmetric range can't give both sides full
# output -- whichever value fits the smaller side leaves the larger side
# clipped early, and whichever fits the larger side leaves the smaller
# side unable to ever reach the joint's true extreme, exactly the
# symptom reported. Same piecewise-anchor technique already used for
# L3's home point: each side gets its own slope, angle=0 (arm pointing
# straight ahead) is the shared anchor.
# 135 was never actually measured for this operator -- it was the
# original guess-based value, carried over unchanged when the left side
# got its own real measurement. Reported directly: "i need to strech my
# body to right hard. thats the problem" -- right DID reach its true
# extreme (confirmed live, held at -1.70), just only under a hard,
# straining stretch, because 135 degrees of raw angle is nowhere near
# what a COMFORTABLE swing produces. A clean live capture of a
# comfortable (not straining) right hold measured the raw angle settling
# around -25 to -35 degrees -- close to the same order of magnitude as
# the left side's own comfortable range, not the wildly different number
# 135 implied.
L1_YAW_RIGHT_RANGE_DEG = 25.0
# Lowered from 20 to 12 -- 20 still required straining to reach, reported
# directly: "it too hard idiot. i need to strehc my body too ratation
# left". Several live captures the same session measured this operator's
# actual raw angle during real attempts: quick swing-throughs peaked
# around 15-30 degrees only at the very top of an effortful reach and
# briefly (not held), while anything closer to a light, non-straining
# motion stayed under ~15 degrees, several times under 10. 12 sits below
# even the lighter end of that range, so a comfortable swing -- not a
# maximal one -- reaches the true left limit. Not narrower still: L3's
# own history this session showed an overly tight span turns ordinary
# noise into hard-clamped extremes.
#
# Briefly raised to 25 chasing a hand-near-face gesture's incidental
# elbow sway reading as a false left rotation, then reverted back to this
# exact JSON-reference value along with the L1 home-blend attempt in
# landmarks_to_joint_angles (also reverted) -- neither held up, and this
# is the confirmed-working number from arm_config_backup.json.
L1_YAW_LEFT_RANGE_DEG = 12.0

# L3 (elbow flexion) tried the same "theoretical range isn't the
# practical range" fix as L1 -- narrowing the _remap() input span to
# roughly p2/p98 of real footage (75deg/170deg) -- and it was WRONG for
# this joint, unlike L1. The measured percentiles looked reasonable in
# isolation (p1=70.3 p5=90.8 p50=131.3 p95=164.5 p99=173.5 deg), and the
# MEDIAN case did land close to home under the narrow mapping (-2.5deg).
# The actual failure was the TAILS: _remap() hard-clamps anything outside
# its input span straight to the joint's mechanical limit, and a narrow
# span is much easier to get pushed outside of by ordinary landmark
# noise or an unusual-but-real pose (raising the arm changes the
# self-occlusion/depth geometry MediaPipe sees, which can transiently
# swing the computed flex angle) -- one noisy frame outside 75-170deg and
# the whole arm SLAMS to the joint's hard limit. Reported directly with a
# screenshot of the robot folded into an extreme, unnatural pose from
# what should have been an ordinary arm-raise, not a deliberate full
# elbow bend. Reverted to the FULL theoretical [0, pi] range -- L3's own
# geometry (acos-based, singular right at 0/pi -- see the median-3 filter
# in robot_node.py) already means those true extremes are rarely and
# only briefly touched, so nothing extra needs pulling the window in from
# them the way L1's genuinely-narrower joint limit required.


def _L1_base_yaw(shoulder, elbow, wrist=None) -> float:
    """
    Horizontal angle of the UPPER ARM (shoulder→elbow) in the XZ plane.
    Robot base rotates to point toward where the elbow is from the shoulder.
    atan2(dx, -dz): raw MediaPipe x, not mirrored.

    Falls back to shoulder->WRIST only when shoulder->elbow's own
    magnitude is below L1_MIN_MAG -- i.e. only in the exact degenerate
    case the skip-gate in robot_node.py already exists to catch, where
    shoulder->elbow has too little signal to trust at all and the
    alternative is simply holding the last value forever. Live capture
    (this operator, multiple attempts at a full left swing) showed the
    elbow itself often stays close to the shoulder's own position -- the
    actual swinging motion happens further out, at the forearm/wrist --
    so shoulder->elbow alone reads as "nothing happened" even during a
    genuine, deliberate swing. Reported directly, repeatedly: "rotation
    to left is not good... doesn't reach far enough left".

    Shoulder->wrist is NOT free of elbow-bend contamination either (see
    the module docstring and this function's own history below for why
    an elbow->wrist forearm vector was already tried as the PRIMARY
    signal and rejected for exactly that reason) -- bending the elbow
    alone still moves the wrist and so still moves this vector some.
    Used only as a fallback, not the primary signal, because the
    trade-off is different here: shoulder->elbow being near-degenerate
    means it was already contributing ~nothing but noise, so accepting
    some bend-contamination from the wrist is a straight improvement
    over holding the last value forever, not a regression from a signal
    that was working.

    History: a prior version negated dx here, reasoned from "the camera
    faces the operator, so raw x is mirrored like looking at another
    person face-on" -- that assumption was never actually verified against
    the real rig (camera placement, which way the operator watches the
    robot/screen while gesturing), only inferred. It was shipped after
    "when i swing my arm to left robot swing to right" and confirmed
    correct once ("rotation is good") on what was evidently a small test
    swing. A later, unrelated change (remapping the human range onto the
    joint's full range, see L1_YAW_RIGHT_RANGE_DEG) was verified three
    independent ways -- isolated math, a fresh process, and the actual
    live running node driven over real ROS topics -- to be perfectly
    sign-preserving; all three still showed the negated version's sign
    convention afterward. Despite that, the operator then reported the
    exact same symptom again: "rotation is reverse again". Since the code
    was proven unchanged in sign, the earlier "rotation is good" call was
    most likely a false confirmation from a swing too small to reveal it,
    not a real regression from the range fix -- so dx goes back to raw,
    undoing the earlier negation, on the operator's direct and repeated
    report rather than on an unverified camera-facing assumption.

    Deliberately NOT elbow->wrist (see the module docstring): that vector
    is the forearm, which swings on its own whenever the elbow bends, with
    the shoulder not moving at all. shoulder->elbow is what a human keeps
    still while bending the elbow, matching what this robot joint should
    physically represent.

    Callers wanting a stable result should check L1_MIN_MAG against
    math.hypot(elbow[0]-shoulder[0], elbow[2]-shoulder[2]) themselves (see
    landmarks_to_joint_angles' '_L1_mag' key) -- this function does not
    gate its own output, since it has no state to hold a fallback value.

    Remapped from the operator's own measured range onto the full joint
    range, not clamped straight into JOINT_LIMITS -- see
    L1_YAW_RIGHT_RANGE_DEG's declaration for why this joint needs the
    opposite treatment from L5/L6's clamp: a raw human shoulder swing is
    WIDER than this joint's own +/-97 deg mechanical range on at least
    one side, so without remapping, the last part of every real swing
    did nothing -- the robot pinned at its limit while the operator kept
    moving. Reported directly: "angle is not accurate enough".

    Piecewise, right and left independently (see L1_YAW_RIGHT_RANGE_DEG /
    L1_YAW_LEFT_RANGE_DEG) -- angle=0 deg (arm pointing straight ahead)
    is the shared anchor, negative -> right (JOINT_LIMITS lower bound),
    positive -> left (upper bound), matching this operator's own
    confirmed sign convention. Same piecewise mechanism as L3's home
    anchor, needed here because the two sides' comfortable raw ranges
    are not close to symmetric for this operator (~135 deg right,
    ~25 deg left) -- a single shared range clips whichever side is
    smaller. _remap's own _clamp needs out_lo <= out_hi (see the L3
    direction-flip bug already found once, 224d3d2), so each branch below
    keeps its output ascending and lets the INPUT pairing carry the sign.
    """
    dx = elbow[0] - shoulder[0]
    dz = elbow[2] - shoulder[2]
    if wrist is not None and math.hypot(dx, dz) < L1_MIN_MAG:
        wdx = wrist[0] - shoulder[0]
        wdz = wrist[2] - shoulder[2]
        if math.hypot(wdx, wdz) >= L1_MIN_MAG:
            dx, dz = wdx, wdz
    # dx is raw, NOT negated -- a negation was tried TWICE now (see this
    # function's own earlier history in this docstring: shipped once,
    # "rotation is good" then "rotation is reverse again"; tried again
    # this same session after five straight captures showed LEFT reading
    # backward, then immediately reported the opposite way: "l1 already
    # reversible! fuck fix it" -- right went backward instead). Reverted
    # back to raw both times. A single global sign flip cannot be the
    # right fix for a symptom that flips WHICH side is wrong depending on
    # when it's tested -- something about how this operator's setup maps
    # to raw dx is not a fixed left/right convention at all (camera
    # angle, which arm, how they're facing it moment to moment), and no
    # amount of negating this one line will make it one. Do not flip this
    # again without first capturing BOTH directions in the SAME session,
    # back to back, showing a clean and CONSISTENT reversal -- one-sided
    # evidence has been wrong both times so far.
    angle_deg = math.degrees(math.atan2(dx, -dz))
    lo, hi = JOINT_LIMITS['joint_L1']
    if angle_deg >= 0:
        return _remap(angle_deg, 0.0, L1_YAW_LEFT_RANGE_DEG, 0.0, hi)
    return _remap(angle_deg, -L1_YAW_RIGHT_RANGE_DEG, 0.0, lo, 0.0)


# Raise-arm side only, same asymmetric-range problem L1 had (see
# L1_YAW_LEFT_RANGE_DEG). Reported directly: "i raise my arm already, but
# the robot arm not raise" -- a clean, HELD live capture (stable for
# 25+ seconds, not a transient spike) measured this operator's genuine
# raised-arm pitch at only ~27 degrees, against the +/-90 degrees the old
# single symmetric remap assumed. 22, not the measured 27 -- same margin
# reasoning as L1_YAW_LEFT_RANGE_DEG (a little headroom so the true
# limit doesn't need an uncomfortable maximum stretch every time).
# Reach-down side (elbow below shoulder) is UNCHANGED at 90 degrees --
# no live evidence yet that it's also too wide, and L3 is the operator's
# primary "reach down" joint now anyway (see its own docstring), so
# leaving this side alone rather than guessing.
L2_PITCH_UP_RANGE_DEG = 22.0
L2_PITCH_DOWN_RANGE_DEG = 90.0


def _L2_base_pitch(shoulder, elbow) -> float:
    """
    Elevation of the UPPER ARM (shoulder→elbow) above/below horizontal.
    MediaPipe y increases downward, so -dy = upward movement.

    Deliberately NOT elbow->wrist -- see _L1_base_yaw and the module
    docstring for why: that vector moves on elbow bend alone, this one
    does not.

    Piecewise, same mechanism as L1/L3's own anchors -- pitch=0 (upper
    arm level) is the shared anchor, negative -> up/raise (JOINT_LIMITS
    lower bound, confirmed by a direct kinematic TF test: L2 near its
    lower bound puts the gripper highest), positive -> down/forward
    (upper bound). See L2_PITCH_UP_RANGE_DEG for why the two sides use
    different spans.

    atan2(dy, abs(dx)), NOT atan2(dy, hypot(dx,dz)) -- dz is MediaPipe's
    own depth estimate, by far the least reliable of the three axes on a
    single RGB camera, and its error is not symmetric: it is biased
    differently depending on which way the arm is rotated (L1), so
    hypot(dx,dz) shrinks or grows with rotation even at the EXACT same
    real pitch. Live capture holding one fixed pose (same measured
    elbow-flexion angle throughout) while sweeping L1 fully left then
    fully right showed L2 reading ~0.35-0.40 at full left but only
    ~0.15-0.18 at full right -- roughly double, purely from which way
    the arm happened to be facing, not a real difference in how raised
    the arm was. Reported directly: "why the l3 dont want to bow down
    when to the left? like to the right position... i want to make sure
    that right possition rotation is same with left postion turining!".
    dx is a direct 2D image-plane coordinate, not a depth estimate, and
    carries far less rotation-dependent bias -- dropping dz from this
    calculation removes the actual source of the asymmetry instead of
    just re-deriving the same biased number a different way (an asin
    reformulation was tried first and is mathematically IDENTICAL to the
    the original atan2 for the same dx/dy/dz, so it changed nothing).
    First attempt at this specific fix -- report back if L2 still reads
    asymmetric between left and right holds of the same pose.
    """
    dx = elbow[0] - shoulder[0]
    dy = elbow[1] - shoulder[1]
    pitch_deg = math.degrees(math.atan2(dy, abs(dx)))
    lo, hi = JOINT_LIMITS['joint_L2']
    if pitch_deg <= 0:
        return _remap(pitch_deg, -L2_PITCH_UP_RANGE_DEG, 0.0, lo, 0.0)
    return _remap(pitch_deg, 0.0, L2_PITCH_DOWN_RANGE_DEG, 0.0, hi)


# The operator's own reference elbow bend for this sign, MEASURED live
# (not a visual estimate off a screenshot, which is what the original
# 90.0 was and turned out wrong): a direct capture of the actual
# reference pose (hand raised, this specific sign) averaged flex_deg =
# 129.1 degrees over 74 samples with the arm actively rotating (L1
# swinging -1.68 to -0.24) the whole time, confirming this is a stable
# per-pose measurement, not a fluke single frame. 90.0 (a visual guess:
# "roughly a right-angle bend") put this same pose's L3 at ~+0.35/+0.40
# instead of the requested -0.35 -- reported directly, twice, with the
# reference image each time: "position l3 also shud be at -0.35!!!...
# withthis hand sign". See _L3_elbow_flexion's docstring for why this
# needs a piecewise mapping rather than just re-narrowing the remap's
# input range (already tried, already reverted -- that made the joint
# fragile to ordinary landmark noise).
L3_FLEX_HOME_DEG = 129.0

# What the home-anchor pose above maps TO -- see _L3_elbow_flexion's own
# comment at its usage site for why this is -0.35 (joint_L3's own
# midpoint) and not 0.0.
L3_HOME_ANCHOR_OUT = -0.35


def _L3_elbow_flexion(shoulder, elbow, wrist) -> float:
    """
    Angle at elbow.  π = straight arm → L3_max.  0 = fully bent → L3_min.

    Flipped a third time, this time on a deliberate BEHAVIOUR change, not
    a corrected guess: with L2 (shoulder pitch) now frozen out of live
    control ("l2 is fix... the link that i want to use is l1, l3, l5 and
    l7"), L3 is the only joint left that can move the gripper toward the
    ground at all -- and the operator explicitly wants STRAIGHTENING the
    arm toward the camera to be the "reach down and pick something up"
    gesture, not bending it ("i dont want bending elbow to lower the
    robot arm... now im straigthen my arms towards the camera"). Verified
    which JOINT_LIMITS extreme is physically "down" with a direct
    kinematic test (isolated robot_state_publisher + TF, L1/L2/L4/L6 held
    at their live values, L3 swept end to end): L3_max (+1.3) put the
    gripper lowest (Z=0.11), L3_min (-2.0) put it highest (Z=0.45) -- not
    a visual guess. So straight (flex=pi) now maps to L3_max, bent
    (flex=0) to L3_min, the opposite of the previous evidence-backed
    convention (which was solving a different problem: matching the
    robot's own folded-vs-extended visual pose to a bent-vs-straight
    human elbow, before L2 existed as an independent down-reaching DOF).

    Range: FULL theoretical [0, pi] geometric span on each side of the
    home anchor below, not narrowed to a "practical" range the way L1
    was. That narrowing was tried here too and reported broken with a
    screenshot -- the robot slammed into an extreme, folded pose from an
    ordinary arm raise, not a deliberate full elbow bend. Root cause:
    _remap() hard-clamps anything outside its input span straight to the
    joint's mechanical limit, and a narrow span (was ~75-170deg, from
    real-footage percentiles) is easy for ordinary landmark noise -- or a
    genuinely different but valid pose, like raising the arm changing
    MediaPipe's depth/occlusion geometry -- to land outside of. The full
    range does not have this fragility -- ordinary noise has much more
    room before it can reach either true extreme.

    Home anchor: a single straight-line remap over that full range has
    home (output 0) at a fixed point determined entirely by
    JOINT_LIMITS -- ~109deg here, close to a straight arm. The operator
    sent a reference screenshot: a roughly right-angle elbow bend they
    want to BE home, not ~109deg. A single _remap() call cannot put an
    arbitrary point at output 0 while ALSO covering the full [0,pi] input
    range on both sides (its zero-crossing is fixed once the four
    endpoints are chosen) without re-narrowing one side and reintroducing
    exactly the hard-clamp fragility above. Piecewise instead: one
    _remap() from [0, L3_FLEX_HOME_DEG] onto [L3_min, 0], another from
    [L3_FLEX_HOME_DEG, 180] onto [0, L3_max] -- home lands exactly on the
    reference pose, and the full theoretical range is still covered
    (each half just has a different slope, since JOINT_LIMITS itself is
    asymmetric: -2.0 to +1.3).

    L3_FLEX_HOME_DEG=90 is a visual estimate from that screenshot (a
    right-angle bend, forearm up), not a landmark measurement -- ask the
    operator to confirm or refine it if the robot's resting pose doesn't
    quite match a relaxed right-angle bend once tested live.

    _angle_between (which this calls into) uses acos, whose derivative
    diverges as the dot product approaches +/-1 -- i.e. right at flex=0
    or flex=pi, the true extremes of this range. Jitter from that is
    handled with a median-3 pre-filter instead of narrowing the range
    away from the singularity, see MEDIAN_WINDOW['joint_L3'] in
    robot_node.py -- measured -30% to -39% wiggle across all 7 recordings,
    the same validation L1 and L6 already got.
    """
    flex_deg = math.degrees(_angle_at(shoulder, elbow, wrist))  # [0,180], 0=bent, 180=straight
    lo, hi = JOINT_LIMITS['joint_L3']
    # Home anchor's OUTPUT is L3_HOME_ANCHOR_OUT (-0.35, the middle of
    # joint_L3's own range), not 0.0 -- requested directly with two
    # screenshots (the reference elbow-bend pose, and the GUI slider
    # sitting at its exact middle): "when position gesture is like this
    # image, i want position of the robot arm is at the angle l3 in the
    # middle of the slider". Matches HOME_POSE['joint_L3'] in
    # robot_control/robot_node.py so the go-home paths and this live-
    # tracked reference pose land on the same value, not two different
    # "home"s.
    #
    # _remap's _clamp needs out_lo <= out_hi (see the L3 direction-flip bug
    # already found once, 224d3d2) -- so direction comes from swapping which
    # INPUT endpoint is passed first, never from swapping the output bounds.
    if flex_deg >= L3_FLEX_HOME_DEG:
        # straighter than home -> toward L3_max (down, see docstring).
        # in_lo=HOME pairs with out_lo=anchor; in_hi=180(straight) pairs
        # with out_hi=hi(max).
        return _remap(flex_deg, L3_FLEX_HOME_DEG, 180.0, L3_HOME_ANCHOR_OUT, hi)
    # more bent than home -> toward L3_min (up). in_lo=0(fully bent)
    # pairs with out_lo=lo(min); in_hi=HOME pairs with out_hi=anchor.
    return _remap(flex_deg, 0.0, L3_FLEX_HOME_DEG, lo, L3_HOME_ANCHOR_OUT)


def _L4_forearm_roll(shoulder, elbow, wrist):
    """
    Disabled: held at 0.0 (home). Was an attempt at axial forearm rotation
    (pronation/supination) from shoulder/elbow/wrist alone, using the
    upper-arm vector's component perpendicular to the forearm, compared
    to world-up.

    Proven structurally incapable of measuring what it claimed, not just
    noisy: rotating the forearm around its OWN axis does not move the
    wrist relative to the elbow at all (a rigid forearm's wrist stays put
    under pure pronation/supination) -- so `fore` (wrist-elbow), the only
    forearm-orientation information this formula had, carries zero real
    twist information by construction. The signal it computed was
    actually reading which way the CURRENT flexion plane happens to be
    oriented against world-up, not any twist.

    Reported directly: "why l3 move the l4" -- confirmed with a synthetic
    test, zero actual forearm twist, pure elbow flexion sweep only: the
    raw roll value jumped from 0deg to a stable 180deg partway through an
    ordinary bend and stayed there for the rest of the range, a real,
    reproducible ~180deg swing from flexion ALONE, not sensor noise. This
    is the same limitation that made L6 switch to hand-fingertip
    landmarks (points genuinely off the forearm's own axis, which DO move
    under real twist) -- L4 never had an equivalent signal available from
    pose-only shoulder/elbow/wrist.

    Held at exactly 0.0 rather than left producing this proven-wrong
    number: a fixed home reads as "no signal for this DOF" honestly,
    where the old formula read as "confident-looking motion" that was
    actually an artifact of whatever the rest of the arm was doing.
    Reworking this to use hand landmarks the way L6 does (the only fix
    that would give it a REAL signal) would make it a near-duplicate of
    L6's own rotation -- a bigger design decision, not made here without
    asking first.
    """
    return 0.0, 1.0   # magnitude=1.0: never let L4_MIN_MAG hold this at
                       # a stale value instead -- it's meant to be 0 always.


def _L5_wrist_pitch(lm_arr: np.ndarray) -> float:
    """
    Wrist pitch (up/down bend) — was a stub that always returned 0.0
    regardless of the operator's actual wrist, the only joint with no real
    mapping at all. Implemented the same way L3 measures elbow bend: the
    signed angle between the forearm (elbow->wrist) and the hand direction
    (wrist->knuckle line midpoint, the same reference L6 already uses),
    projected onto the vertical plane so it reads "up" and "down" the way
    a human means it (world-relative), independent of forearm roll.

    Sign convention: bending the hand DOWN is positive, bending it UP is
    negative, a straight wrist is 0 -- negated from the synthetic bend
    test's own result (straight=0.0deg, bent up=+63.4deg, bent
    down=-63.4deg) because that convention drove the real joint backwards.
    See the negation at the end of this function for why.
    """
    # Wrist/pinky/index dropped from this check -- same fix as
    # _l6_raw_angle's and l6_calibrate's now-removed gates: these
    # landmarks read 0.11-0.23 for this operator, consistently below
    # MIN_VISIBILITY=0.3, so this joint returned 0.0 (its own
    # "unreadable" fallback) almost every frame instead of tracking real
    # wrist bend. Only shoulder ever proved reliably separated between
    # present/absent this session -- see ARM_CONTROL_MIN_VISIBILITY's own
    # declaration -- so it stays the one hard visibility floor.
    if lm_arr[R_ELBOW, 3] < MIN_VISIBILITY:
        return 0.0

    elbow = _v(lm_arr, R_ELBOW)
    wrist = _v(lm_arr, R_WRIST)
    hand_point = (_v(lm_arr, R_PINKY_TIP) + _v(lm_arr, R_INDEX_TIP)) / 2.0

    forearm = wrist - elbow
    nf = np.linalg.norm(forearm)
    hand = hand_point - wrist
    nh = np.linalg.norm(hand)
    if nf < 1e-9 or nh < 1e-9:
        return 0.0
    forearm = forearm / nf
    hand = hand / nh

    world_up = np.array([0.0, -1.0, 0.0])      # MediaPipe y increases downward
    right = np.cross(forearm, world_up)
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        # Forearm itself is nearly vertical -- world_up is degenerate as a
        # cross-product reference here, fall back to a fixed horizontal axis.
        right = np.cross(forearm, np.array([1.0, 0.0, 0.0]))
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            return 0.0
    right = right / nr
    up_ish = np.cross(right, forearm)
    up_ish = up_ish / np.linalg.norm(up_ish)

    pitch = math.atan2(float(np.dot(hand, up_ish)), float(np.dot(hand, forearm)))
    # Negated: the sign convention documented in this function's own
    # docstring (bend up = MediaPipe -Y = positive) is backwards relative to
    # the actual robot joint -- bending the real wrist up moved L5 the wrong
    # way on the arm. Reported directly, twice: "it is reversible, thats
    # why... the l5 behaive likeshit" then "the l5 angle is reversible
    # idiot!!!". Unlike L1's sign saga earlier this session (flipped twice,
    # reverted twice, because each flip broke the OTHER direction), this is
    # a first-time report on a joint that had never been sign-tested live
    # before now, not a repeated flip-flop -- so this stands until live
    # data says otherwise, rather than being provisional.
    pitch = -pitch
    # Clamped to the human range (+/-90deg), not JOINT_LIMITS (+/-120deg) --
    # same fix as L6's WRIST_ROLL_HUMAN_RANGE, see its declaration.
    return float(np.clip(pitch, -WRIST_PITCH_HUMAN_RANGE, WRIST_PITCH_HUMAN_RANGE))


def hand_openness(lm_arr: np.ndarray):
    """
    Detect how open the right hand is from MediaPipe Pose fingertip landmarks.

    Measures average distance from wrist to the 3 available fingertips
    (pinky=18, index=20, thumb=22), normalised by forearm length so the
    result is scale-invariant (works at any camera distance).

    Returns
    -------
    float [0.0, 1.0] : 0.0 = closed fist, 1.0 = flat open hand
    None             : key landmarks not visible
    """
    tips     = [R_PINKY_TIP, R_INDEX_TIP, R_THUMB_TIP]
    # Wrist/tips dropped from this check -- same reasoning as the other
    # wrist/fingertip gates fixed alongside this one this session. This
    # function's own OUTPUT is overridden downstream anyway (see the
    # module docstring: the real gripper comes from HAND landmarks via
    # ik_solver.gripper_from_hand()), so the practical impact is smaller,
    # but it is the identical bug and cheap to fix consistently.
    if lm_arr[R_ELBOW, 3] < MIN_VISIBILITY:
        return None

    wrist   = _v(lm_arr, R_WRIST)
    elbow   = _v(lm_arr, R_ELBOW)
    forearm = float(np.linalg.norm(wrist - elbow))
    if forearm < 1e-6:
        return None

    avg_dist = float(np.mean([np.linalg.norm(_v(lm_arr, t) - wrist) for t in tips]))
    norm = (avg_dist / forearm - _GRIPPER_CLOSED_NORM) / (
           _GRIPPER_OPEN_NORM - _GRIPPER_CLOSED_NORM)
    return float(np.clip(norm, 0.0, 1.0))


def _L6_wrist_roll(elbow, wrist, lm_arr: np.ndarray, baseline: float = 0.0):
    """
    Wrist roll from the hand's own orientation, RELATIVE to the forearm.

    Was computed as the knuckle line's raw angle in the camera's image
    plane -- an ABSOLUTE-frame angle, the exact same category of bug L1
    originally had (elbow->wrist instead of shoulder->elbow). Proven with
    a synthetic test that twisted the wrist NOT AT ALL and instead just
    tilted the whole forearm+hand sideways in front of the camera: the old
    L6 swung 1:1 with the tilt (90 degrees of arm tilt -> 90 degrees of L6,
    despite zero actual wrist rotation). Any time the operator moved their
    arm at all, that motion leaked into L6 on top of whatever the wrist
    itself was doing.

    Fixed using the same technique that already works correctly in
    _L4_forearm_roll: project the signal vector (here, the knuckle line)
    perpendicular to the forearm axis, and measure its angle against
    world-up (also projected perpendicular to the forearm) rather than
    against the raw image axes. This measures rotation AROUND the forearm,
    which is what a wrist roll physically is, independent of how the
    forearm itself is currently oriented.

    Returns (angle, magnitude). magnitude is |across_perp| -- near zero
    when the knuckle line happens to point along the forearm axis itself
    (fingers curled straight forward/back), the degenerate case for this
    formula, analogous to L4's elbow_perp. See L6_MIN_MAG.

    The raw atan2 here spans the full +/-180 degrees; a real wrist does
    not (pronation/supination is commonly cited around +/-75-90 degrees).
    Clamped into +/-WRIST_ROLL_HUMAN_RANGE (not JOINT_LIMITS, which is
    wider): 1:1 gain, the robot shows approximately the angle actually
    rotated. An earlier version remapped onto the full joint range
    instead, which amplified every rotation by ~2x (rotate 45deg for
    real, robot shows ~89deg) -- reported directly: "still not translate
    well to the angle that i want".

    Fixed now, not just documented: "0" used to be wherever the knuckle
    line happened to align with world-up projected perpendicular to the
    forearm -- an external reference with no reason to match any given
    operator's own relaxed wrist orientation. Confirmed as the actual
    cause of a real complaint, not a theoretical one: a synthetic sweep
    of genuine 0-90 degree wrist rotation came back PINNED at the exact
    joint limit for the entire range, because that test's "neutral" raw
    angle already sat right at the edge of the +/-90 degree window being
    remapped -- any further real rotation only pushed it outside that
    window into the clamp. Reported directly: "if i rotate my wrist...
    it will follow 360... not good".

    `baseline` is the operator's own raw angle at rest (the caller
    captures this once, e.g. on the first live frame, and passes it back
    in every call after) -- subtracted before remapping, so the +/-90
    degree window is centred on THIS operator's actual neutral instead of
    an arbitrary world direction. Wrapped through atan2(sin, cos) rather
    than a plain subtraction so a baseline near +/-180 degrees does not
    produce a bogus multi-hundred-degree difference.
    """
    angle, mag = _l6_raw_angle(elbow, wrist, lm_arr)
    if angle is None:
        return 0.0, 0.0

    # Centre on the operator's own neutral before remapping. Plain
    # subtraction would break near the +/-180 wrap (e.g. angle=+179,
    # baseline=-179 is really 2 degrees apart, not 358); atan2(sin,cos)
    # of the difference wraps it back into (-180, +180] correctly.
    centered = math.atan2(math.sin(angle - baseline), math.cos(angle - baseline))
    # 1:1 gain: remapping onto JOINT_LIMITS (+/-177.6deg) from a +/-90deg
    # human range meant a real 45deg rotation showed up as ~89deg on the
    # robot -- reported directly: "still not translate well to the angle
    # that i want". Clamping into +/-WRIST_ROLL_HUMAN_RANGE itself instead
    # means the robot shows approximately the angle actually rotated,
    # clipped only beyond a realistic wrist's own range, not amplified
    # within it. Costs some of the joint's mechanical range going unused
    # (it can reach +/-177.6deg, a human wrist can't drive it past ~90),
    # which is the right side of this trade-off to be on.
    mapped = _clamp(centered, -WRIST_ROLL_HUMAN_RANGE, WRIST_ROLL_HUMAN_RANGE)
    return mapped, mag


def _l6_raw_angle(elbow, wrist, lm_arr: np.ndarray):
    """
    The uncalibrated, uncentred angle _L6_wrist_roll builds on -- shared so
    a caller can capture it once as a per-operator baseline (see
    l6_calibrate below) without duplicating this geometry. Returns
    (angle_or_None, magnitude); None means degenerate (see L6_MIN_MAG).

    Uses the PLANE through all 3 of Pose's hand points (pinky, index,
    thumb tips), not just pinky-index -- averaging in a 3rd point this
    way is spatial noise reduction (each point's independent MediaPipe
    jitter gets partly cancelled by the other two), unlike a temporal
    filter, so it costs no extra lag. Measured directly on real footage
    against the old 2-point (index-pinky) signal, same calibration and
    smoothing on both sides: wiggle score (how much the signal visibly
    dances back and forth) dropped 48-81% across 3 recordings (3261->1691,
    4085->1781, 3195->609) -- a real reduction in the jitter-vs-lag
    trade-off documented on MEDIAN_WINDOW in robot_node.py, not another
    turn of that same dial.
    """
    # No wrist/fingertip visibility gate here -- this was the THIRD
    # independent visibility check found this session (after the ones in
    # robot_node.py's outer gate and landmarks_to_joint_angles()'s inner
    # gate, both already fixed the same way). Confirmed directly, live:
    # wrist/pinky/index/thumb read 0.11-0.23 for this operator, genuinely
    # present and gesturing -- consistently below this check's own
    # MIN_VISIBILITY=0.3, so L6 silently held its last value (frozen)
    # regardless of any freeze switch. Reported directly with a
    # screenshot: "why the fuck gripper behave like this". The magnitude
    # check just below (mag < 1e-9, backed by L6_MIN_MAG at the call
    # site) already rejects genuinely degenerate geometry; visibility
    # scores on this setup don't separate real tracking from absence any
    # better here than they did for the other two gates.
    fore = wrist - elbow
    norm_f = np.linalg.norm(fore)
    if norm_f < 1e-9:
        return None, 0.0
    fore = fore / norm_f

    pinky = _v(lm_arr, R_PINKY_TIP)
    across = np.cross(_v(lm_arr, R_INDEX_TIP) - pinky,
                      _v(lm_arr, R_THUMB_TIP) - pinky)
    across_perp = across - np.dot(across, fore) * fore
    mag = float(np.linalg.norm(across_perp))
    if mag < 1e-9:
        return None, 0.0

    # MediaPipe/camera-space: y-down, so world-up = (0, -1, 0). Same
    # projection L4 uses for its own world-up reference.
    up = np.array([0.0, -1.0, 0.0])
    up = up - np.dot(up, fore) * fore
    if np.linalg.norm(up) < 1e-9:
        return None, 0.0

    angle = math.atan2(
        float(np.dot(np.cross(up, across_perp), fore)),
        float(np.dot(up, across_perp)),
    )
    return angle, mag


def l6_calibrate(lm_arr: np.ndarray):
    """
    Capture the operator's current raw L6 angle, to use as l6_baseline in
    landmarks_to_joint_angles from then on. Call this once while the
    operator's wrist is relaxed/neutral -- robot_node.py does this on the
    first live frame after startup. Returns None if the required
    landmarks aren't visible yet (caller should keep the previous
    baseline, or 0.0, and retry on a later frame).
    """
    # Wrist dropped from this check for the same reason as _l6_raw_angle's
    # own now-removed gate just above: consistently reads 0.11-0.23 for
    # this operator, so requiring it here meant l6_calibrate() returned
    # None on essentially every frame, l6_cal_frames never advanced, and
    # the operator's own neutral-wrist baseline never actually locked in.
    if lm_arr[R_ELBOW, 3] < MIN_VISIBILITY:
        return None
    elbow = _v(lm_arr, R_ELBOW)
    wrist = _v(lm_arr, R_WRIST)
    angle, _mag = _l6_raw_angle(elbow, wrist, lm_arr)
    return angle


def _L7_gripper(lm_arr: np.ndarray) -> float:
    """
    Map hand openness to gripper finger travel (L7, prismatic, metres).
    Flat open hand → 0.03 (open)
    Closed fist    → 0.00 (closed)

    hand_openness() already returns 0..1, and L7's limits are 0..0.03, so
    this is a straight scale with no offset -- unlike the old L6 mapping,
    which had to straddle a +/-3.1 rad range whose midpoint was "half open".
    """
    openness = hand_openness(lm_arr)
    if openness is None:
        return 0.0
    lo, hi = JOINT_LIMITS['joint_L7_R']
    return lo + openness * (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def landmarks_to_joint_angles(lm_arr: np.ndarray, l6_baseline: float = 0.0) -> dict:
    """
    Map a [33, 5] landmark array to robot joint angles.

    Parameters
    ----------
    lm_arr : np.ndarray, shape (33, 5)
        Columns: [x, y, z, visibility, presence]
        Coordinates may be MediaPipe-normalized or depth-fused camera-space
        (output of fuse_depth) — both work because only inter-joint vectors
        are used.
    l6_baseline : float, radians
        The operator's own raw L6 angle at rest, so wrist roll is measured
        relative to THEIR neutral rather than an arbitrary world direction.
        This module stays stateless (see the module docstring) -- the
        caller captures this once (e.g. robot_node.py, on the first live
        frame) and passes it back in on every call. Default 0.0 reproduces
        the old, uncalibrated behaviour.

    Returns
    -------
    dict {joint_name: float}  or  {} if key landmarks are not visible.
    """
    # Shoulder only, matching robot_node.py's own outer gate exactly --
    # see that gate's long comment in _on_pose_landmarks for the full
    # story. This inner check was still requiring elbow AND wrist here,
    # completely independently of that outer gate, at the SAME
    # MIN_VISIBILITY=0.3 that was already proven too strict for elbow/
    # wrist on this operator's setup (real captures showed wrist reading
    # 0.08-0.25 routinely, genuinely present). Fixing the outer gate and
    # leaving this one untouched meant every frame that cleared the outer
    # check could still be silently discarded HERE instead -- confirmed
    # directly: fed a realistic frame (shoulder=0.99, elbow=0.55,
    # wrist=0.18) through this function 15 times in a row and it returned
    # {} every single time, so _apply_smoothing never ran and L3 (or any
    # other joint) never updated -- reported as "l3 still broken" after
    # the outer gate was already fixed and verified correct.
    if lm_arr[R_SHOULDER, 3] < MIN_VISIBILITY:
        return {}

    shoulder = _v(lm_arr, R_SHOULDER)
    elbow    = _v(lm_arr, R_ELBOW)
    wrist    = _v(lm_arr, R_WRIST)

    _g = _L7_gripper(lm_arr)
    l1_mag = math.hypot(elbow[0] - shoulder[0], elbow[2] - shoulder[2])
    # Separate from l1_mag above on purpose -- l1_mag alone also gates
    # L2/L3 (see robot_node.py's skip-dict and L2_MIN_MAG's own notes),
    # which is about THEIR OWN degenerate pitch/flexion geometry and has
    # nothing to do with L1's wrist fallback below. Reporting this one
    # separately means L1 can be un-skipped by a usable wrist reading
    # without also silently un-skipping L2/L3 on a shoulder-elbow vector
    # that is still just as degenerate for THEM as it was before.
    l1_wrist_mag = math.hypot(wrist[0] - shoulder[0], wrist[2] - shoulder[2])
    l4_angle, l4_mag = _L4_forearm_roll(shoulder, elbow, wrist)
    l6_angle, l6_mag = _L6_wrist_roll(elbow, wrist, lm_arr, baseline=l6_baseline)
    l2_pitch = _L2_base_pitch(shoulder, elbow)
    l3_flex  = _L3_elbow_flexion(shoulder, elbow, wrist)
    # L2 (raise) and L3 (straighten -> down, see its own docstring)
    # directly fight each other whenever the arm is raised AND straight
    # at once -- a completely natural combination (this operator's own
    # elbow stays close to straight while raising, confirmed live).
    # Reported directly with a screenshot: "joint l3 shud be this
    # position [-2.00] when i rased my arm up to the sky" -- verified
    # with a direct kinematic test that -2.0 is genuinely the better
    # choice there too, not just requested blindly: with L2 held at its
    # raised extreme, sweeping L3 showed pushing it toward +1.3
    # (straight, its normal "down" reading) pulls the gripper BACK DOWN
    # (Z 0.381 at L3=0 down to Z=0.197 at L3=+1.3) -- the elbow's own
    # geometry works against a raised reach once L2 is doing the actual
    # lifting, so this is not overriding a correct reading with an
    # arbitrary one, it's replacing a reading that was actively
    # counter-productive at this range of L2. Blended, not a hard
    # cutover at some L2 threshold, so raising the arm partway doesn't
    # produce a visible snap in L3.
    _L2_raised = JOINT_LIMITS['joint_L2'][0]   # -0.98, L2's raised extreme
    _L3_raised = JOINT_LIMITS['joint_L3'][0]   # -2.00, requested directly
    raise_frac = _clamp(l2_pitch / _L2_raised, 0.0, 1.0) if _L2_raised else 0.0
    l3_flex = (1.0 - raise_frac) * l3_flex + raise_frac * _L3_raised

    # An L1 home-pull keyed off L3's proximity to L3_HOME_ANCHOR_OUT was
    # tried here and reverted -- disproven by an offline test against all
    # 7 of this operator's own recorded videos (not just one live capture):
    # during "hand near face" frames, L3 actually ranges from -0.27 to
    # -2.00 depending on the exact take, nowhere near consistently close
    # to -0.35, so the blend would almost never have triggered. No other
    # joint (L2 checked too) reliably predicts this gesture's L1 instability
    # either -- the same gesture's raw shoulder->elbow geometry genuinely
    # varies too much take-to-take on this camera/operator for any fixed
    # formula to separate "hold this gesture" from "genuine L1 rotation".
    #
    # DISCRETE detection instead of a geometric correction: rather than
    # trying to compute a centred L1 from unstable elbow geometry, detect
    # the hand-near-face POSE ITSELF (wrist close to the nose) and snap L1
    # straight to 0 while it's held, the same architecture as the existing
    # fingers-spread go-home gesture. Wrist-to-nose distance, normalised by
    # shoulder width for scale invariance (so it doesn't depend on distance
    # from the camera). Checked against all 7 of this operator's own
    # recordings: "near face" frames measured 0.17-0.53 (normalised),
    # everything else measured 0.91+ -- a wide, clean gap, unlike L3's
    # signal above. L1_FACE_OVERRIDE_DIST=0.6 sits in that gap with margin
    # on both sides. Reported directly, twice, with screenshots: "why it
    # rotate more to the right?" / "just mapping it as the my elobow
    # postion like that as home postion? cant?". Only L1's own output is
    # touched, per direct instruction: "dont disturb other link fucker.
    # iti is only link 1".
    L1_FACE_OVERRIDE_DIST = 0.6
    nose = _v(lm_arr, NOSE)
    l_shoulder = _v(lm_arr, L_SHOULDER)
    _shoulder_w = float(np.linalg.norm(shoulder - l_shoulder))
    _wrist_to_nose = float(np.linalg.norm(wrist - nose))
    _norm_wrist_to_nose = (_wrist_to_nose / _shoulder_w) if _shoulder_w > 1e-6 else 999.0
    _l1_face_override = _norm_wrist_to_nose < L1_FACE_OVERRIDE_DIST
    if _l1_face_override:
        l1_yaw = 0.0
    else:
        l1_yaw = _L1_base_yaw(shoulder, elbow, wrist)

    return {
        'joint_L1': l1_yaw,
        'joint_L2': l2_pitch,
        'joint_L3': l3_flex,
        'joint_L4': l4_angle,
        'joint_L5': _L5_wrist_pitch    (lm_arr),
        'joint_L6': l6_angle,
        # One value, both jaws: computed once so they cannot diverge.
        'joint_L7_R': _g,
        'joint_L7_L': _g,
        # Not joints -- see L1_MIN_MAG / L4_MIN_MAG / L6_MIN_MAG. Below
        # its threshold, the matching joint's angle above is dominated by
        # noise from a near-degenerate vector, not a real reading.
        # JOINT_NAMES-based consumers ignore these extra keys.
        '_L1_mag': l1_mag,
        '_L1_wrist_mag': l1_wrist_mag,
        '_L4_mag': l4_mag,
        '_L6_mag': l6_mag,
        # True while L1_FACE_OVERRIDE_DIST's hand-near-face override is
        # forcing joint_L1 to 0.0 above -- read by robot_node.py's
        # _apply_smoothing to hard-snap L1 instead of routing it through
        # the median/step-cap pipeline, which would otherwise take several
        # frames to visibly reach 0.
        '_l1_face_override': _l1_face_override,
    }
