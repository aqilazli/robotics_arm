#!/usr/bin/env python3
"""
ik_solver.py  —  Pure math library. No ROS, no camera.
=========================================================
Inverse/forward kinematics over the robot's own URDF (urdf/robot_arm.urdf),
used to drive all 6 joints from a single wrist Cartesian target.

Why this exists
----------------
motion_mapping.landmarks_to_joint_angles() computed joint angles directly
from shoulder+elbow+wrist body-pose geometry. MediaPipe Hands (21 landmarks)
never gives a shoulder or elbow, so that approach doesn't work once the
perception layer is hand-only. This module solves the same problem the
other way round: treat the (depth-fused) wrist landmark as a single
end-effector target position and let IK figure out all 6 joint angles.

Coordinate frame note
----------------------
The wrist target is the Orbbec camera-space (X, Y, Z) position in metres
(feature_extractor._fuse_depth / motion_mapping.fuse_depth output). This is
used directly as the IK target position, i.e. the camera frame is treated as
coincident with the robot base frame. There is no extrinsic camera↔robot
calibration in this project (motion_mapping.py made the same simplifying
assumption for its shoulder/elbow/wrist vectors) — WRIST_OFFSET below is a
single tunable translation to align the operator's reachable hand volume
with the arm's actual workspace; adjust it for your physical camera/arm
placement instead of doing full extrinsic calibration.

Gripper (L7)
------------
The gripper is joint_L7, a prismatic jaw, not L6 — L6 is the wrist roll, and
rotating it doesn't move the chain's end position anyway (its axis passes
through the tip), so IK's solved L6 value is meaningless. Gripper open/close
is derived separately from whole-hand openness via gripper_from_hand(): close
your palm and the gripper closes, open your hand and it opens.
"""

import os
import re
import tempfile

import numpy as np
from ikpy.chain import Chain

from motion_mapping import JOINT_NAMES, JOINT_LIMITS

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_URDF_PATH = os.path.join(_ROOT, 'urdf', 'robot_arm.urdf')

# Translate the wrist's camera-space position into the arm's workspace.
# Tune to your physical camera/arm placement.
WRIST_OFFSET = np.array([0.0, 0.0, 0.3])

# Hand landmark indices (MediaPipe Hands, 21 points)
WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20

# The four finger tips, thumb excluded: the thumb folds across the palm rather
# than curling toward the wrist, so its distance barely changes between a fist
# and a flat hand and including it only flattens the signal.
FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# Mean fingertip-to-wrist distance divided by palm length (wrist to middle
# knuckle). Measured on this metric, not the old thumb-to-index pinch, which
# these constants used to describe.
#
# These are MEASURED from the operator, not estimated: 293 samples holding a
# fist and 308 holding a flat hand gave medians of 0.85 and 1.65. The previous
# estimates of 1.10 / 1.95 were both too high, so a flat hand only reached
# about 65% open and the gripper never fully opened. Medians, not extremes:
# one fist frame read 7.258 where the hand was clearly mis-detected.
#
# Re-measure if the camera distance or operator changes.
_GRIPPER_CLOSED_NORM = 0.85
_GRIPPER_OPEN_NORM   = 1.65


def _build_chain() -> Chain:
    """
    ikpy's URDF parser doesn't support joint_L6's `continuous` type, so we
    parse a patched in-memory copy (continuous -> revolute, same limits) —
    the real urdf/robot_arm.urdf on disk (used by Gazebo/ros2_control) is
    never touched.
    """
    src = open(_URDF_PATH).read()
    patched = re.sub(r'type="continuous"', 'type="revolute"', src)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    try:
        tmp.write(patched)
        tmp.close()
        chain = Chain.from_urdf_file(
            tmp.name,
            base_elements=['base_link', 'joint_L1', 'L1', 'joint_L2', 'L2',
                           'joint_L3', 'L3', 'joint_L4', 'L4', 'joint_L5', 'L5'],
        )
    finally:
        os.unlink(tmp.name)
    return chain


_chain = None


def _get_chain() -> Chain:
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def _to_full_vector(angles: dict) -> list:
    """One value per link of the ikpy chain, in the chain's own order.

    Keyed by the chain's link names rather than built positionally from
    JOINT_NAMES. The chain is a single serial branch
    (base -> L1..L6 -> L7_R), but JOINT_NAMES also carries joint_L7_L, the
    second gripper jaw, which hangs off the same parent as L7_R and so is
    not on this branch at all. Building the vector positionally therefore
    produced a 9-long vector for an 8-link chain the moment the gripper
    joints were added, and every forward()/solve() call raised ValueError.
    Reading the names off the chain keeps this correct if the URDF changes
    again.
    """
    chain = _get_chain()
    return [float(angles.get(link.name, 0.0)) for link in chain.links]


def forward(angles: dict) -> np.ndarray:
    """
    Parameters
    ----------
    angles : dict {joint_name: radians}, e.g. output of solve()

    Returns
    -------
    np.ndarray (3,)  end-effector xyz in the chain's base frame (metres)
    """
    chain = _get_chain()
    fk = chain.forward_kinematics(_to_full_vector(angles))
    return fk[:3, 3]


def solve(target_xyz: np.ndarray, initial_angles: dict = None) -> dict:
    """
    Parameters
    ----------
    target_xyz     : (3,) desired wrist position, robot base frame, metres
    initial_angles : dict, optional warm-start (helps IK converge to a
                     nearby solution instead of jumping between poses)

    Returns
    -------
    dict {joint_name: radians}, clamped to JOINT_LIMITS. joint_L6 is left at
    0.0 here — set it separately with gripper_from_hand().
    """
    chain = _get_chain()
    seed  = _to_full_vector(initial_angles) if initial_angles else None
    sol   = chain.inverse_kinematics(np.asarray(target_xyz, dtype=float),
                                     initial_position=seed)

    # Same name-keyed walk as _to_full_vector: the solution vector is per
    # chain link, so a positional zip against JOINT_NAMES ran off the end
    # once JOINT_NAMES grew past the chain length.
    out = {}
    for link, value in zip(chain.links, sol):
        if link.name not in JOINT_LIMITS:
            continue                                  # dummy base link
        lo, hi = JOINT_LIMITS[link.name]
        out[link.name] = float(np.clip(value, lo, hi))
    return out


def gripper_from_hand(lm_arr: np.ndarray):
    """
    Map whole-hand openness to the gripper joint (L7).

    Closing your palm into a fist closes the gripper; opening your hand flat
    opens it. Measured as the mean distance from the wrist to the four finger
    tips, divided by palm length so the result is scale invariant and works at
    any distance from the camera.

    This deliberately does NOT use a thumb-to-index pinch (which an earlier
    version did): a pinch is a different gesture from closing your hand, and
    making a fist barely changes it, so a fist did not close the gripper.

    Parameters
    ----------
    lm_arr : (21, 5)  [x, y, z, visibility, presence]

    Returns
    -------
    float  jaw travel in METRES (not radians), clamped to the L7 limits, or
           None if the hand could not be measured this frame
    """
    wrist = lm_arr[WRIST,      :3]
    palm  = lm_arr[MIDDLE_MCP, :3]

    palm_size = float(np.linalg.norm(palm - wrist))
    if palm_size < 1e-6:
        # None, not 0.0: the caller holds the previous opening instead of
        # snapping the gripper shut on an unreadable frame.
        return None

    spread = float(np.mean([
        np.linalg.norm(lm_arr[t, :3] - wrist) for t in FINGER_TIPS
    ])) / palm_size

    openness = (spread - _GRIPPER_CLOSED_NORM) / (_GRIPPER_OPEN_NORM - _GRIPPER_CLOSED_NORM)
    openness = float(np.clip(openness, 0.0, 1.0))

    lo, hi = JOINT_LIMITS['joint_L7_R']
    return lo + openness * (hi - lo)


_ADJACENT_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def hand_fingers_spread(lm_arr: np.ndarray):
    """
    How far apart the fingers are held from each other -- a distinct shape
    from gripper_from_hand's wrist-to-fingertip OPENNESS, which stays high
    for any extended-fingers hand regardless of whether they're held
    together or splayed apart. Fingers held together (this operator's
    normal resting/tracking hand shape) is a LOW value here even though
    openness reads fully open; deliberately spreading fingers apart (a
    distinct, rare shape during ordinary use) is a HIGH value. Built as a
    second, independent go-home trigger signal after openness alone was
    found to fire on ordinary tracking -- see robot_node.py's
    _home_gesture_raw for that history.

    Measured as the summed gap between each pair of ADJACENT fingertips
    (thumb-index, index-middle, middle-ring, ring-pinky), divided by palm
    length for the same scale-invariance reason gripper_from_hand divides
    by it.

    Parameters
    ----------
    lm_arr : (21, 5)  [x, y, z, visibility, presence]

    Returns
    -------
    float  scale-invariant spread ratio, or None if the hand could not be
           measured this frame
    """
    wrist = lm_arr[WRIST,      :3]
    palm  = lm_arr[MIDDLE_MCP, :3]

    palm_size = float(np.linalg.norm(palm - wrist))
    if palm_size < 1e-6:
        return None

    tips = [lm_arr[t, :3] for t in _ADJACENT_TIPS]
    gaps = sum(float(np.linalg.norm(tips[i + 1] - tips[i])) for i in range(len(tips) - 1))
    return gaps / palm_size


def wrist_target_from_landmarks(lm_arr: np.ndarray) -> np.ndarray:
    """Depth-fused wrist position (camera frame) -> IK target (robot frame)."""
    return lm_arr[WRIST, :3] + WRIST_OFFSET
