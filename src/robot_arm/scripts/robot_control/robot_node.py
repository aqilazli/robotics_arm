#!/usr/bin/env python3
"""
robot_node.py  —  ROS 2 node
==============================
Drives the robot arm from a hybrid of two MediaPipe tracks published by
inference_node.py, each used for what it's actually suited to:

  /arm/pose_landmarks (33x5, MediaPipe Pose: shoulder/elbow/wrist)
      -> motion_mapping.landmarks_to_joint_angles() -> joint_L1-L4 directly.
      The robot's elbow bends because the operator's elbow bends — no
      inverse kinematics, no invented target position.

  /arm/landmarks (21x5, MediaPipe Hands: fingers)
      -> ik_solver.gripper_from_hand() -> joint_L7 only.
      Hand landmarks have no shoulder/elbow, so they were never suited to
      driving the arm itself — only to what a hand actually does: open
      and close a gripper.

Both topics are cached independently and combined on whichever arrives
most recently, so the arm keeps moving even if one track briefly drops a
frame. joint_L5 (wrist pitch) is not modelled — stays neutral.

Subscribed topics
-----------------
  /arm/pose_landmarks  std_msgs/Float32MultiArray  (33x5, arm control)
  /arm/landmarks       std_msgs/Float32MultiArray  (21x5, gripper control)
  /pose_ai_model       std_msgs/String  'lstm' | 'gru' | 'transformer' | 'none'
                        -- live-switch which trained pose predictor drives
                        L1-L4 latency compensation, no restart needed.

Published topics
----------------
  /robot_arm_controller/joint_trajectory   trajectory_msgs/JointTrajectory
  /active_pose_ai_model  std_msgs/String  current active pose predictor name
                          (published on startup and on every switch, so a
                          GUI elsewhere -- e.g. inference_node.py's camera
                          window -- can display what's actually driving the
                          arm right now)

Parameters
----------
  trajectory_dt_ms     int    time_from_start per trajectory point (default 150)
  smoothing            float  EMA smoothing factor 0=none, 0.9=heavy (default 0.2)
  pose_predictor_path  str    path to an extra trained pose predictor .pt to
                               load in addition to the auto-discovered ones
                               below (default '' = none)
  pose_predictor_type  str    which predictor is active at startup:
                               'lstm' | 'gru' | 'transformer' | 'none' (default 'none')

Optional arm latency compensation (live-switchable model toggle)
-------------------------------------------------------------------
Same idea as the gripper's hand-landmark predictor (perception/inference_node.py),
applied to the arm: a PosePreprocessor buffers the last 30 live pose frames,
and once full, a trained predictor guesses the NEXT pose frame instead of
using the (already slightly stale) live one, before it goes into
landmarks_to_joint_angles(). Both landmarks_to_joint_angles() and
gripper_from_hand() only use relative vector differences/ratios between
landmarks, never absolute position, so a normalised predicted pose vector
can be fed in directly -- no denormalisation step needed, matching how the
gripper's predicted landmarks are already used.

At startup, every trained model found at
models/pose_{lstm,gru,transformer}_predictor.pt is loaded into a cache (like
inference_node.py's hand-side _predictor_cache) -- not just the one named by
pose_predictor_type. That parameter only picks which one starts active.
Publish 'lstm' / 'gru' / 'transformer' / 'none' to /pose_ai_model at any
time afterward to switch live, for real-time side-by-side comparison
without restarting the node. Starts at 'none' (direct live pose, unchanged
behaviour) unless pose_predictor_type says otherwise.
"""

import math
import os
import sys
import time

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from motion_mapping import (JOINT_NAMES, JOINT_LIMITS, ARM_CONTROL_MIN_VISIBILITY,
                            R_SHOULDER, R_ELBOW, R_WRIST,
                            L1_MIN_MAG, L2_MIN_MAG, L4_MIN_MAG, L6_MIN_MAG,
                            landmarks_to_joint_angles, l6_calibrate)
from data_processing.pose_feature_extractor import PoseFeatureExtractor
from data_processing.pose_preprocessor import PosePreprocessor

# Requested directly: "i want this whenever no motion it will become
# position like this!", with a picture of the arm folded down close to the
# base. Read straight off /joint_states while the operator had it posed
# exactly right via the Arm GUI's MANUAL sliders -- these are the actual
# live values that produced the picture, not a visual estimate off the
# screenshot (that kind of guess got the L3 direction wrong twice earlier
# this session). Gripper deliberately excluded: joint_L7's own "open until
# a hand is seen" default (see self._gripper_angle) already covers it and
# wasn't part of this request.
# All-zeros / fully straight arm. Was a folded reference pose (L1=0.2565,
# L2=0.2923, L3=-0.35, L5=0.2188 -- itself the result of several earlier
# corrections chasing a specific reference hand-sign) until reconsidered
# directly: "why initail postion of gazebo robort startup like this? is
# houdl be straight" -- confirmed via AskUserQuestion that home should be
# a straight arm instead. Kept numerically in sync BY HAND with arm_gui.py's
# own HOME_POSE copy, load_controllers.py's HOME copy, and the URDF's
# ros2_control initial_value params -- none of these four are imported from
# a shared source (see home_pose.no_shared_import in arm_config_backup.json).
HOME_POSE = {
    'joint_L1': 0.0,
    'joint_L2': 0.0,
    'joint_L3': 0.0,
    'joint_L4': 0.0,
    'joint_L5': 0.0,
    'joint_L6': 0.0,
}

_POSE_PREDICTOR_CLASSES = {
    'lstm':        ('models.lstm_predictor',        'LSTMPredictor'),
    'gru':         ('models.gru_predictor',         'GRUPredictor'),
    'transformer': ('models.transformer_predictor', 'TransformerPredictor'),
}

# L6 calibration stability gate -- see _l6_baseline's declaration.
L6_CAL_STABLE_WINDOW  = 5
L6_CAL_STABLE_TOL_RAD = math.radians(15)
L6_CAL_MAX_FRAMES     = 30


def _circular_spread(vals):
    """Max pairwise angular distance within `vals` (radians), wrap-aware."""
    spread = 0.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = abs(math.atan2(math.sin(vals[i] - vals[j]),
                               math.cos(vals[i] - vals[j])))
            spread = max(spread, d)
    return spread


def _circular_mean(vals):
    return math.atan2(sum(math.sin(v) for v in vals),
                      sum(math.cos(v) for v in vals))


class RobotNode(Node):

    # How long the operator must be undetected (shoulder visibility gate
    # failing continuously) before the arm returns to home -- see its use
    # in _on_pose_landmarks. Long enough that a brief occlusion or a head
    # turn doesn't yank the arm home mid-gesture; short enough that
    # stepping away actually reads as "gone" promptly.
    _GO_HOME_AFTER_SEC = 1.5

    # Wrist visibility floor, checked ONLY as a sustained condition (see
    # WRIST_LOW_VIS_SUSTAIN_SEC), not a single-frame cutoff -- a single-frame
    # threshold was tried before for elbow/wrist (see the shoulder-only-gate
    # comment in _on_pose_landmarks) and rejected because this operator's
    # genuine, actively-tracked wrist regularly reads 0.3-0.6, overlapping
    # with "actually absent". 0.15 sits comfortably below that whole genuine-
    # tracking band, so it only catches wrist readings that are unambiguously
    # low. Added after a live capture of the exact failure the shoulder-only
    # gate's own comment predicted ("worth revisiting if the arm starts
    # moving on its own between gestures"): shoulder_vis pinned at 0.999 the
    # whole time (operator genuinely in frame) while wrist_vis decayed
    # smoothly from 0.33 down to 0.03 over ~14s because the arm itself was
    # mostly out of camera view -- MediaPipe kept extrapolating a guessed
    # wrist position with decaying confidence instead of ever reporting
    # "gone", and robot_node kept publishing that guess as real motion.
    # Reported directly: "sometimes it s dancing when no miotion... wtf".
    WRIST_MIN_VISIBILITY_SUSTAINED = 0.15
    WRIST_LOW_VIS_SUSTAIN_SEC = 1.5

    # Separate idle-timeout for "detected but not moving" -- see
    # _maybe_go_home_idle. Longer than _GO_HOME_AFTER_SEC on purpose: this
    # fires off a much softer signal (comparing joint values, not "is
    # anyone even there"), so it gets more time to be sure a real pause
    # between gestures isn't about to become a new gesture.
    _GO_HOME_IDLE_AFTER_SEC = 3.0
    # Compare against a snapshot from this far back, not frame-to-frame --
    # see self._idle_ref_values' declaration for why (filter settling looks
    # like motion frame-to-frame).
    IDLE_CHECK_WINDOW_SEC = 2.0
    # 0.15 (from an earlier, shorter capture) was not enough margin.
    # Reported directly: "stilll. wtf" -- sitting still, L1 visibly rock
    # steady, but it never went home. A longer (20s) live capture showed
    # why: L2/L3 have periodic noise SPIKES up to ~0.245 rad, recurring
    # roughly every 8s, distinct from the ~0.05-0.09 rad continuous jitter
    # the original threshold was measured against -- every spike reset the
    # idle clock before it could ever hold 3 uninterrupted seconds, even
    # though the arm looked still the whole time (L1, the most visually
    # obvious joint, genuinely was). Confirmed these are real landmark
    # noise, not a resurgence of the old coordinate-scale-jump bug (SH/EL/
    # WR stayed in one consistent, plausible scale across every spike, no
    # sudden jump to a different coordinate frame). 0.35 gives real margin
    # above the largest observed spike -- same value already used as
    # MAX_STEP_RAD's "clearly beyond normal noise" line elsewhere in this
    # file, not a new number invented for this.
    IDLE_MOTION_RAD = 0.35

    # Explicit "go home" hand gesture -- see _maybe_go_home_gesture. Went
    # through two designs before this one:
    #   v1: hand openness (gripper fraction >= 0.95). Reported broken --
    #       an open hand is ALSO this operator's normal resting hand shape
    #       during ordinary tracking, not a distinct sign, so it fired
    #       constantly and blocked normal movement ("icant move to other
    #       psotion when show sgin of hoem! dont block it fucker").
    #   v2: openness + wrist-to-nose "near face" position, to make it
    #       distinct from ordinary tracking. Fixed the block, but was
    #       explicitly asked to be replaced with a different hand shape
    #       entirely rather than a position qualifier on the same shape
    #       ("i mean he go to home is still the same sign, i wan tdiffrent
    #       hand land mark sign").
    #   v3 (current): FINGERS SPREAD APART (HAND_HOME_SPREAD_MIN below),
    #       not openness at all -- a hand with fingers extended but held
    #       together (this operator's normal tracking shape) and a hand
    #       with fingers deliberately splayed apart are different shapes
    #       even though both read as "open" on the old openness metric,
    #       so this is a genuinely distinct signal, not a variant of the
    #       one that kept causing problems.
    HAND_HOME_SPREAD_MIN = 2.8

    # Wall-clock seconds the fingers-spread condition must hold
    # CONTINUOUSLY before the sign counts as engaged -- explicit request,
    # a full reversal of the original "immediately, not waiting secs"
    # requirement from when this gesture was still openness-based: "go to
    # home sign need 7sec before it go to home, so that it will not
    # disutvrb any movent!". 7 corrected down to 5 immediately after,
    # directly: "go to home time taken after sign is 5 secs, not 7 secs".
    # Time-based rather than a tick count (the first debounce this
    # gesture used) because /arm/hand_spread's actual rate isn't
    # constant -- it depends on hand detection, which drops out entirely
    # sometimes -- so a fixed tick count would correspond to different
    # real durations depending on conditions; 5 seconds means 5 seconds
    # regardless. RELEASING is still NOT debounced -- one reading back
    # under the threshold drops it immediately, so relaxing the hand
    # resumes tracking with no lag; only ENGAGING waits.
    HOME_GESTURE_CONFIRM_SEC = 5.0

    # How many frames of raw (pre-smoothing) angle to average when latching
    # the teleop-engage reference -- see _on_control_mode's gesture-entry
    # branch and _on_pose_landmarks' engaging block. A single frame was
    # avoided deliberately: MediaPipe's per-frame noise would get baked into
    # the reference permanently for the rest of that engage, the same
    # single-frame-noise trap L6's own baseline calibration was built to
    # avoid (see l6_calibrate/_l6_baseline).
    TELEOP_ENGAGE_FRAMES = 5

    # L1/L2/L3 are ALL excluded from the engage-relative rescale now -- back
    # to the pure absolute mapping documented in arm_config_backup.json's
    # "98% confirmed" reference snapshot (motion_mapping_constants_reference:
    # L1_base_yaw/L2_base_pitch/L3_elbow_flexion, none of which mention any
    # rescale -- that snapshot predates this engage feature entirely).
    # History: L3 was excluded first, after rescaling it broke its precise
    # separately-calibrated -0.35 reference reading ("the l3 quite not
    # aacurete from previous pos, previous is better with -0.35 angle!").
    # L1/L2 were KEPT on the rescale at that point ("dont touch l1 and l2.
    # taht already prefect") and seemed fine in isolation -- but the very
    # next live test (a sustained L1 rotation) exposed the rescale's
    # interaction with L2's own rotation-damping cap (L2_ROTATION_DAMPED_
    # STEP_RAD): a live capture showed L2 stuck creeping for 13+ seconds
    # while the operator's real L2 barely moved. Loosening that cap
    # (0.08->0.20) measurably shortened the lag but still wasn't good
    # enough ("not good, please refer the json beofre for l1 l2 l3") -- so
    # rather than keep tuning a rescale none of these three joints actually
    # need, all three go back to the exact absolute-mapping behaviour the
    # JSON reference already proved correct. Only L4/L5/L6 keep the
    # engage-relative rescale now.
    OFFSET_JOINTS = tuple(
        j for j in JOINT_NAMES
        if j not in ('joint_L7_R', 'joint_L7_L', 'joint_L1', 'joint_L2', 'joint_L3'))

    def __init__(self):
        super().__init__('robot_node')

        # ── parameters ───────────────────────────────────────────────────────
        # trajectory_dt_ms must track the rate poses actually arrive at.
        # MediaPipe runs at mediapipe_fps (30 -> a pose every ~33ms), and
        # joint_trajectory_controller DISCARDS the active trajectory whenever a
        # new one arrives. With dt=100ms against 67ms arrivals the arm was
        # re-targeted at 67% of every motion and never settled, which read as
        # constant jitter. Keep this at or just above the arrival period.
        self.declare_parameter('trajectory_dt_ms',    40)
        # EMA is smoothed = alpha*old + (1-alpha)*new, so 0.3 passed 70% of
        # each frame's raw landmark noise straight through to the joints.
        # 0.5 damps that without adding the lag a heavier value would.
        self.declare_parameter('smoothing',           0.5)
        self.declare_parameter('pose_predictor_path', '')
        self.declare_parameter('pose_predictor_type', 'none')
        # Debugging aid: hold specific joints at whatever they were the
        # moment this was enabled, so only the REST respond live. For
        # isolating one joint at a time when troubleshooting -- e.g.
        # "does L5 alone track wrist bend, with nothing else in the chain
        # moving to confuse the picture." Not meant to stay on for normal
        # operation. Comma-separated joint names, e.g.
        #   --ros-args -p freeze_joints:=joint_L1,joint_L2,joint_L3,joint_L4
        # Started as a fixed "freeze L1-L5" boolean, generalised once
        # testing moved from "isolate L6" to "isolate L5" and a hardcoded
        # set stopped fitting -- same debugging need, different joints,
        # no reason this should take a code change each time.
        self.declare_parameter('freeze_joints', '')

        self._dt_ms      = self.get_parameter('trajectory_dt_ms').value
        self._alpha      = self.get_parameter('smoothing').value
        self._frozen_set = {j.strip() for j in
                            self.get_parameter('freeze_joints').value.split(',')
                            if j.strip()}
        # Loud and unmissable on purpose: run_arm.sh builds its own
        # hardcoded command line for this node and does NOT pass
        # freeze_joints, so launching the normal way silently leaves this
        # off even if you meant to test with it set. Check this line in
        # the terminal before troubleshooting a joint -- if it says NONE,
        # the flag never reached this process, full stop.
        self.get_logger().info(
            '='*60 +
            f'\nFREEZE_JOINTS = {", ".join(sorted(self._frozen_set)) if self._frozen_set else "NONE -- all joints live, normal operation"}\n'
            + '='*60)

        # Default 'gesture': run_arm.sh exists to drive the arm from the
        # camera, so the vision path owns it until the GUI says otherwise.
        # transient_local so the mode set before this node started still
        # arrives rather than being missed.
        # Live joint positions, used as the explicit start point of every
        # trajectory (see _publish).
        # Gripper opening in metres, computed once in inference_node from the
        # RAW landmarks. Subscribing to the number rather than recomputing it
        # here: this node receives /arm/landmarks, which is the PREDICTOR's
        # output in normalised feature space when a predictor is active and
        # depth-fused metric coords when it is not. Running gripper_from_hand
        # on whichever arrived gave two different answers, and could inverted
        # the mapping outright.
        self._joint_now = {}
        self.create_subscription(
            JointState, '/joint_states',
            lambda m: self._joint_now.update(dict(zip(m.name, m.position))), 10)
        self.create_subscription(Float32, '/arm/gripper_opening',
                                 self._on_gripper_opening, 10)
        # Drives the go-home gesture -- see HAND_HOME_SPREAD_MIN's own
        # declaration for why this is a separate signal from gripper
        # openness above, and ik_solver.hand_fingers_spread /
        # inference_node.py for where it's computed (from the same raw,
        # pre-depth-fusion landmarks gripper_opening already uses).
        self._hand_spread = 0.0
        self.create_subscription(Float32, '/arm/hand_spread',
                                 self._on_hand_spread, 10)

        # Safety default: teleop starts OFF. The GUI publishes the real,
        # latched startup value on /control_mode (see arm_gui.py's
        # _build_mode_panel); this local default only matters if robot_node
        # somehow starts before that latched message is ever received, and
        # must never be 'gesture' -- requested directly as a hardware safety
        # feature: "i ned to push a swtihc to on the teloperation... it is
        # safety features for thehardwarde later in real developemtn."
        self._control_mode = 'manual'

        # Teleop-engage relative offset -- see OFFSET_JOINTS and
        # _on_control_mode's gesture-entry branch. Without this, flipping
        # CAMERA TELEOPERATION on jumped the arm straight to the operator's
        # actual real-world pose at that instant (an absolute mapping),
        # instead of starting at home and only following movement FROM
        # there. Reported directly: "i click the button camera teloperatinn,
        # suddenly it moving to the right folowing my shoulder. it shoud
        # started from the init position! no go there by itslef".
        self._teleop_engaging   = False
        self._teleop_engage_buf = []
        # Per-joint RAW angle at the moment teleop engaged (not an offset --
        # see the piecewise rescale in _on_pose_landmarks for why a flat
        # additive offset was wrong).
        self._teleop_engage_ref = {}


        # Assert a known-good pose shortly after startup. load_controllers.py
        # sends home the instant it activates the controller, and if the
        # controller is not ready yet that message is silently dropped -- which
        # left the two gripper jaws in DIFFERENT states (one open, one closed)
        # with nothing to correct them, because this node only publishes when
        # landmarks arrive and stays silent until someone steps in front of the
        # camera. Firing once, late enough to be safe, removes that whole class
        # of startup mismatch.
        # Fires repeatedly, not once: see _assert_startup_pose.
        self._startup_timer = self.create_timer(2.0, self._assert_startup_pose)
        self.create_subscription(
            String, '/control_mode', self._on_control_mode,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # So the Gesture Monitor (a separate process) can show freeze
        # state directly instead of it only being visible in this
        # terminal's startup log -- reported directly: "fix the gui
        # control monitor too so that i can undesrtand what happening",
        # after repeated confusion this session over whether a freeze
        # flag had actually reached this process. transient_local: the
        # monitor may start before or after this node: either order
        # still gets the current value.
        self._pub_freeze_status = self.create_publisher(
            String, '/freeze_status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._publish_freeze_status()
        # Live per-joint freeze toggle from the Arm GUI's switch panel.
        # freeze_joints (the parameter above) is read once at startup only,
        # which meant isolating a different joint mid-session needed a full
        # relaunch -- asked for directly: "make a swtich at gui so that i
        # can off the joint that i want during teleopreation".
        # VOLATILE, not TRANSIENT_LOCAL -- deliberately does NOT match
        # /control_mode's latching here. Proved directly that a
        # TRANSIENT_LOCAL subscriber replays retained history from every
        # writer it has ever matched, in an order that isn't guaranteed, so
        # a late-started robot_node could inherit a stale freeze from an
        # orphaned old GUI process (gnome-terminal-server has left those
        # behind more than once this session) instead of the current GUI's
        # real state. arm_gui.py's _reassert_freeze republishes its true
        # checkbox state every 2s specifically so a VOLATILE subscriber
        # here still ends up correct within that window either way.
        self.create_subscription(
            String, '/freeze_joints_cmd', self._on_freeze_cmd,
            QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE))
        # Republished every 2s, not just once at startup. transient_local
        # already covers a late-joining monitor, and both orderings were
        # verified to deliver -- but the whole point of this banner is to
        # END uncertainty about whether a freeze is active, and a banner
        # that is silently absent looks identical to "no freeze". If it
        # ever fails to appear, that must not be because a single publish
        # went missing. Cheap: one tiny string every 2 seconds.
        self.create_timer(2.0, self._publish_freeze_status)
        self._smoothed = {}
        # Rolling raw-value history per joint, for MEDIAN_WINDOW below.
        self._raw_history = {}
        # Large-jump values awaiting confirmation, per joint. See
        # JUMP_CONFIRM_RAD.
        self._jump_pending = {}
        # L6's own "zero" -- the operator's raw wrist angle at rest. See
        # l6_calibrate() and motion_mapping._L6_wrist_roll's docstring:
        # without this, "0" was an arbitrary world direction with no
        # relation to any specific operator's actual neutral wrist, and
        # real rotation could pin straight to the joint limit instead of
        # tracking proportionally. Reported directly: "if i rotate my
        # wrist... it will follow 360... not good".
        #
        # Locking onto whatever the very first readable frame showed
        # risked calibrating to a mid-motion pose if the operator was not
        # actually relaxed yet the instant it fired -- there was no way
        # to tell "neutral" from "still moving" from one frame alone.
        # Instead this buffers recent readings and waits for
        # L6_CAL_STABLE_WINDOW of them to agree within
        # L6_CAL_STABLE_TOL_RAD before locking in their circular mean, so
        # calibration happens when the wrist is actually still. Gives up
        # waiting and locks in with whatever it has after
        # L6_CAL_MAX_FRAMES, so a fidgety startup can't block L6 forever.
        self._l6_baseline = None
        self._l6_cal_buf = []
        self._l6_cal_frames = 0

        # Go-home-when-undetected state -- see _maybe_go_home. Wall-clock
        # time the operator was LAST confirmed present (shoulder cleared
        # the gate); starts at "now" so a slow startup before the first
        # frame ever arrives doesn't read as an instant timeout.
        self._last_detected_time = time.time()
        self._went_home = False   # avoid re-publishing home every tick
                                   # once already there

        # Wall-clock time wrist visibility was LAST above WRIST_MIN_VISIBILITY_
        # SUSTAINED, or None if it currently is above it -- see the check in
        # _on_pose_landmarks, right after the shoulder gate.
        self._wrist_low_vis_since = None
        # Checked on a TIMER, not only inside the pose-landmarks callback:
        # inference_node does not publish ANYTHING at all when it detects
        # neither a hand nor a pose (a bare `continue`, no message sent) --
        # confirmed directly, live, with zero /arm/pose_landmarks messages
        # arriving for a full 10s window while genuinely undetected. A
        # check that only runs in response to an incoming message can
        # never fire if no message is ever going to arrive; this timer
        # fires regardless of whether anything is being published.
        # Reported directly, after the first (reactive-only) version
        # shipped and was found to never actually trigger this way: "why
        # still got motion when there is no traceking handpose and arm"
        # -- followed by confirming genuinely nobody was in frame and the
        # arm was stuck at a non-home position with nothing moving it.
        self.create_timer(0.5, self._maybe_go_home)

        # Go-home-when-MOTIONLESS state -- separate from the detection-based
        # one above. Reported directly: the operator sitting still, clearly
        # visible and detected the whole time ("Are you sitting still in
        # front of the camer im her") -- detection alone (shoulder visible)
        # says nothing about whether the arm is actually MOVING, and the old
        # behaviour only went home when the operator left the frame
        # entirely. This tracks a snapshot of self._smoothed from
        # IDLE_CHECK_WINDOW_SEC ago and compares against it, not
        # frame-to-frame -- comparing consecutive frames would mistake this
        # joint's own filter settling (EMA/median/step-cap all still
        # catching up after a real move) for continued motion. Measured
        # live, genuinely sitting still: resting jitter tops out around
        # 0.05-0.09 rad per joint over a couple of seconds -- IDLE_MOTION_RAD
        # sits with real margin above that.
        self._idle_ref_time = time.time()
        self._idle_ref_values: dict = {}
        self._idle_still_since = None
        self._went_home_idle = False
        self.create_timer(0.5, self._maybe_go_home_idle)

        self._have_pose = False
        # OPEN until a hand is seen, not 0.0. For a prismatic gripper 0.0 is
        # fully CLOSED, not neutral -- and pose landmarks arrive even when no
        # hand is detected, so this published a shut gripper on the first
        # frame and slammed it closed right after home had opened it.
        self._gripper_angle = JOINT_LIMITS['joint_L7_R'][1]

        # Explicit "go home" hand gesture -- separate from BOTH other
        # go-home paths (not-detected, and the general hold-any-pose-still
        # idle timeout). Reported directly with a screenshot of a raised,
        # fully open, spread hand: "i want it home postion when i make
        # this sign" -- a deliberate confirm gesture, not something that
        # should need the whole arm to also go quiet for
        # _GO_HOME_IDLE_AFTER_SEC first. Driven off /arm/hand_spread (see
        # HAND_HOME_SPREAD_MIN) -- see _maybe_go_home_gesture. Only state
        # needed is whether this is a NEW sign episode, for the log line.
        self._went_home_gesture = False
        # Debounce on the ENGAGE side only -- see _home_gesture_condition's
        # own declaration. None means "not currently holding the sign";
        # otherwise the time.time() the CURRENT continuous hold started.
        self._home_gesture_hold_since = None

        # ── optional arm latency compensation (live-switchable) ─────────────
        self._pose_extractor       = PoseFeatureExtractor()
        self._pose_preprocessor    = PosePreprocessor(window_size=30)
        self._pose_predictor_cache = {}     # name -> PredictorBase instance
        self._active_pose_name     = 'none'
        self._active_pose_predictor = None
        self._load_pose_predictors()

        # ── publishers ───────────────────────────────────────────────────────
        self._pub = self.create_publisher(
            JointTrajectory,
            '/robot_arm_controller/joint_trajectory',
            10,
        )
        self._pub_active_pose_model = self.create_publisher(String, '/active_pose_ai_model', 10)

        # Raw, per-frame human-tracked angle for every joint, BEFORE the
        # teleop engage rescale, median filter, EMA or step caps touch it --
        # i.e. exactly what the camera says your body is doing right now, in
        # the same units/convention as /joint_states so the two can be read
        # side by side. Added directly so the operator can see and verify
        # numbers themselves instead of me guessing from a description:
        # "make me other table gui for human arm position and robot arm
        # position. so taht i can undersstnd the value and can adjust
        # accrodingly myself ratrehr than you guess like shit".
        self._pub_human_angles = self.create_publisher(JointState, '/arm/human_angles', 10)

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/arm/pose_landmarks',
            self._on_pose_landmarks, 10,
        )
        # No /arm/landmarks subscription any more. That topic carries the
        # PREDICTOR's output in normalised feature space when a predictor is
        # active and depth-fused metric coords when it is not, so computing
        # the gripper from it gave two different answers and could invert the
        # mapping. inference_node now computes the opening once from the raw
        # landmarks and publishes it on /arm/gripper_opening, subscribed above.
        self.create_subscription(
            String, '/pose_ai_model',
            self._on_pose_model_switch, 10,
        )
        # Live re-trigger for L6's own "zero" -- see _l6_baseline's own
        # declaration. That calibration only ever ran once, automatically,
        # at startup; if the wrist wasn't actually neutral during that
        # window (or just needs redoing), there was no way to fix it
        # short of relaunching the whole node. Reported directly: "l6?
        # why not center?" ... "no if not frozen also like that".
        self.create_subscription(
            String, '/recalibrate_l6',
            self._on_recalibrate_l6, 10,
        )

        self._publish_active_pose_model()
        self.get_logger().info('RobotNode ready.')

    # ── optional arm latency compensation ───────────────────────────────────

    def _load_pose_predictors(self):
        """Load every trained pose predictor found into a cache, so any of
        them can be switched to live via /pose_ai_model -- not just the one
        active at startup."""
        import importlib

        for kind, (module_name, class_name) in _POSE_PREDICTOR_CLASSES.items():
            path = os.path.join(_SCRIPTS, 'models', f'pose_{kind}_predictor.pt')
            if not os.path.exists(path):
                continue
            try:
                Cls = getattr(importlib.import_module(module_name), class_name)
                predictor = Cls(window_size=30, feature_dim=99)
                predictor.load(path)
            except Exception as e:
                self.get_logger().warn(f'Failed to load {kind} pose predictor ({e}) — skipping.')
                continue
            self._pose_predictor_cache[kind] = predictor
            self.get_logger().info(f'Pose predictor loaded: {kind.upper()} ← {path}')

        # optional extra model at an explicit path, in addition to the auto-discovered ones
        extra_path = self.get_parameter('pose_predictor_path').value
        if extra_path:
            extra_kind = self.get_parameter('pose_predictor_type').value
            if extra_kind in _POSE_PREDICTOR_CLASSES and os.path.exists(extra_path):
                try:
                    import importlib as _il
                    module_name, class_name = _POSE_PREDICTOR_CLASSES[extra_kind]
                    Cls = getattr(_il.import_module(module_name), class_name)
                    predictor = Cls(window_size=30, feature_dim=99)
                    predictor.load(extra_path)
                    self._pose_predictor_cache[extra_kind] = predictor
                    self.get_logger().info(f'Pose predictor loaded: {extra_kind.upper()} ← {extra_path}')
                except Exception as e:
                    self.get_logger().warn(f'Failed to load pose_predictor_path ({e}).')
            elif not os.path.exists(extra_path):
                self.get_logger().warn(f'pose_predictor_path not found: {extra_path}')

        if not self._pose_predictor_cache:
            self.get_logger().info(
                'No pose predictors found — L1-L4 uses live pose directly (no prediction).')
            return

        # pick the startup-active model
        wanted = self.get_parameter('pose_predictor_type').value
        if wanted in self._pose_predictor_cache:
            self._set_active_pose_model(wanted)
        else:
            self.get_logger().info(
                f'pose_predictor_type "{wanted}" not loaded — starting in direct mode. '
                f'Available: {list(self._pose_predictor_cache.keys())}')

    def _set_active_pose_model(self, name: str):
        if name in ('none', 'off', ''):
            self._active_pose_predictor = None
            self._active_pose_name      = 'none'
            self._pose_preprocessor.reset()   # avoid mixing pre/post-switch frames in one window
            return True
        if name in self._pose_predictor_cache:
            self._active_pose_predictor = self._pose_predictor_cache[name]
            self._active_pose_name      = name
            self._pose_preprocessor.reset()
            return True
        return False

    def _on_pose_model_switch(self, msg: String):
        """Handle /pose_ai_model -- live-switch the active pose predictor."""
        name = msg.data.lower().strip()
        if self._set_active_pose_model(name):
            self.get_logger().info(f'Pose AI switched to: {self._active_pose_name.upper()}')
        else:
            self.get_logger().warn(
                f'Unknown pose_ai_model "{name}" — available: '
                f'{list(self._pose_predictor_cache.keys()) + ["none"]}')
        self._publish_active_pose_model()

    def _on_recalibrate_l6(self, msg: String):
        """
        Any message on /recalibrate_l6 resets calibration state so the
        SAME startup logic (see _l6_baseline's declaration, and the
        stability-gated loop in _on_pose_landmarks) runs again live: hold
        the wrist still for L6_CAL_STABLE_WINDOW frames within
        L6_CAL_STABLE_TOL_RAD and the new neutral locks in. Message
        content is ignored -- this is a trigger, not a parameter.
        """
        self._l6_baseline = None
        self._l6_cal_buf = []
        self._l6_cal_frames = 0
        self.get_logger().info(
            'L6 recalibration requested -- hold wrist still to set new neutral.')

    def _publish_freeze_status(self):
        fs = String()
        fs.data = ','.join(sorted(self._frozen_set))   # '' when nothing frozen
        self._pub_freeze_status.publish(fs)

    def _on_freeze_cmd(self, msg: String):
        """
        Replace the live frozen-joint set from the GUI switch panel.
        Comma-separated joint names, '' clears every freeze. A joint newly
        added here is not snapped anywhere -- _apply_smoothing simply stops
        updating self._smoothed[j] for it below, so it holds at exactly
        wherever it already was, mid-motion or not. Only the very first
        frame ever received (cold start, see _apply_smoothing) snaps a
        frozen joint to 0; that case does not apply here.
        """
        new_set = {j.strip() for j in msg.data.split(',') if j.strip()}
        if new_set == self._frozen_set:
            return
        self._frozen_set = new_set
        self.get_logger().info(
            'FREEZE_JOINTS (live, from GUI) = '
            f'{", ".join(sorted(self._frozen_set)) if self._frozen_set else "NONE -- all joints live"}')
        self._publish_freeze_status()

    def _publish_active_pose_model(self):
        status = String()
        status.data = self._active_pose_name
        self._pub_active_pose_model.publish(status)

    def _maybe_go_home(self):
        """
        Timer-driven, not message-driven -- see self._last_detected_time's
        declaration for why the message-driven version of this could
        never fire when inference_node stops publishing entirely. Fires
        at most once per undetected episode (self._went_home), and resets
        the moment the operator is confirmed present again (in
        _on_pose_landmarks' success path).
        """
        if self._went_home:
            return
        if time.time() - self._last_detected_time < self._GO_HOME_AFTER_SEC:
            return
        # Build a COMPLETE pose from scratch rather than mutating
        # self._smoothed in place -- if the operator has never been
        # detected since startup, self._smoothed may still be {} entirely,
        # and _publish() indexes every one of JOINT_NAMES unconditionally;
        # a partial dict would crash it. Non-frozen joints go to HOME_POSE
        # (0.0 for any joint not in it, i.e. the gripper); frozen joints
        # keep whatever self._smoothed already had for them, or 0.0 too if
        # there's nothing there yet -- same "freeze holds until real data
        # arrives" contract documented on _frozen_set elsewhere in this file.
        home = {j: (self._smoothed.get(j, 0.0) if j in self._frozen_set
                     else HOME_POSE.get(j, 0.0))
                for j in JOINT_NAMES}
        self._smoothed = home
        self._publish(home)
        self.get_logger().info(
            f'No operator detected for {self._GO_HOME_AFTER_SEC:.0f}s -- '
            'arm returned to home.')
        self._went_home = True

    def _maybe_go_home_idle(self):
        """
        Motion-based idle timeout, separate from the detection-based one
        above -- see self._idle_ref_values' declaration for why detection
        alone isn't enough (an operator can be clearly visible and
        perfectly still, holding any pose, and this needs to send the arm
        home anyway). Compares the CURRENT self._smoothed against a
        snapshot taken IDLE_CHECK_WINDOW_SEC ago, not frame-to-frame (see
        the same declaration for why); if every joint moved less than
        IDLE_MOTION_RAD across that window, this ticks the "been still"
        clock instead of resetting it. _GO_HOME_IDLE_AFTER_SEC of that
        publishes the same home pose _maybe_go_home does.

        Skips entirely while _maybe_go_home has already handled it
        (operator not even detected) -- no need for both to fire, and
        _went_home already means the arm is at home.
        """
        if self._went_home:
            return
        if not self._smoothed:
            return
        now = time.time()
        if not self._idle_ref_values:
            self._idle_ref_values = dict(self._smoothed)
            self._idle_ref_time = now
            return
        if now - self._idle_ref_time < self.IDLE_CHECK_WINDOW_SEC:
            return
        moved = max(
            abs(self._smoothed.get(j, 0.0) - self._idle_ref_values.get(j, 0.0))
            for j in JOINT_NAMES
        )
        # Refresh the reference every window regardless of outcome, so the
        # comparison always covers the MOST RECENT window rather than one
        # that keeps growing from a single stale starting point.
        self._idle_ref_values = dict(self._smoothed)
        self._idle_ref_time = now
        if moved >= self.IDLE_MOTION_RAD:
            # Real motion resumed -- allow this to fire again next time
            # things go still, same as _went_home resetting on detection.
            self._idle_still_since = None
            self._went_home_idle = False
            return
        if self._idle_still_since is None:
            self._idle_still_since = now
            return
        if self._went_home_idle:
            return
        if now - self._idle_still_since < self._GO_HOME_IDLE_AFTER_SEC:
            return
        home = {j: (self._smoothed.get(j, 0.0) if j in self._frozen_set
                     else HOME_POSE.get(j, 0.0))
                for j in JOINT_NAMES}
        # HOME_POSE has no L7 entry (see its own declaration for why), so
        # the dict comp above would default it to 0.0 -- CLOSED -- every
        # time this fires, unlike every other go-home path in this file
        # (_maybe_go_home, _maybe_go_home_gesture, _assert_startup_pose,
        # _on_control_mode's snap), which all explicitly override L7 with
        # the live gripper reading for exactly this reason. Missed here
        # until now -- found by inspection while fixing the idle-parked
        # publish race just above, not from a live report of the gripper
        # itself snapping shut, so keep an eye out for that if it recurs.
        if 'joint_L7_R' not in self._frozen_set:
            home['joint_L7_R'] = self._gripper_angle
        if 'joint_L7_L' not in self._frozen_set:
            home['joint_L7_L'] = self._gripper_angle
        self._smoothed = home
        self._publish(home)
        self.get_logger().info(
            f'No motion for {self._GO_HOME_IDLE_AFTER_SEC:.0f}s (operator '
            'still detected, just not moving) -- arm returned to home.')
        self._went_home_idle = True

    # ── arm: pose landmarks -> L1-L4 (direct geometry, or predicted pose) ─────

    def _on_pose_landmarks(self, msg: Float32MultiArray):
        n = 33 * 5
        if len(msg.data) < n:
            return

        pose_arr = np.array(msg.data[:n], dtype=float).reshape(33, 5)

        # Gate on the REAL landmarks before anything else. The predictor
        # stamps its output vis/presence = 1.0, so once it is active the
        # visibility check inside landmarks_to_joint_angles() can never fail
        # and the arm keeps chasing predictions made from nothing -- it moved
        # on its own with no operator in frame, because MediaPipe still emits
        # low-confidence guesses and the predictor launders them into a
        # confident-looking pose.
        #
        # This used to also require elbow/wrist visibility (with wrist given
        # its own lower bar, then a position/direction/angle "stability"
        # override on top of that -- four separate redesigns, all measured
        # against live data, documented in git history rather than kept
        # here). All of them were trying to solve the same impossible
        # problem: for THIS operator's actual camera/lighting, elbow and
        # wrist visibility during genuine, present, active tracking
        # regularly reads 0.3-0.6 -- BELOW elbow's own measured absent-
        # ceiling (0.746) and wrist's (0.649, both in
        # ARM_CONTROL_MIN_VISIBILITY's declaration). Present and absent
        # produce overlapping scores for these two landmarks on this setup;
        # no threshold, and no amount of smoothing the comparison, can
        # separate them, because the raw information to separate them
        # isn't there.
        #
        # Shoulder is the one landmark that stayed reliably separated in
        # every capture across this entire investigation: ~0.99-1.00
        # whenever the operator was actually present and testing, in every
        # single live session. Gating on shoulder alone, and trusting the
        # existing downstream smoothing (median pre-filter + EMA, see
        # MEDIAN_WINDOW) to absorb elbow/wrist noise instead of blocking it
        # at the door, prioritises the arm actually tracking real motion
        # over rejecting uncertain frames. Reported directly, after the
        # gate kept freezing the arm through several rounds of tightening:
        # "l3 not show well. please fix it follow my hand".
        #
        # Known, accepted cost: this reopens (for elbow/wrist specifically)
        # the original failure this whole gate was built to prevent --
        # "operator present, arm out of shot" phantom-chasing, since
        # shoulder alone does not distinguish that case (the original
        # calibration measured shoulder ~0.99 read as real EVEN THEN). Not
        # a risk during active testing; worth revisiting if the arm starts
        # moving on its own between gestures.
        if pose_arr[R_SHOULDER, 3] < ARM_CONTROL_MIN_VISIBILITY:
            # Say so, throttled. This gate returning silently is the single
            # most confusing failure this node has: every other indicator
            # looks healthy (controller active, gripper homing fine,
            # landmarks drawn on the monitor) while the arm simply never
            # moves, with nothing anywhere saying why. Reported as "still
            # not moving!!!!" with a fully healthy-looking startup log.
            self.get_logger().warning(
                f'ARM NOT MOVING: shoulder visibility below '
                f'{ARM_CONTROL_MIN_VISIBILITY} '
                f'(shoulder={pose_arr[R_SHOULDER,3]:.2f}). '
                'Get your whole upper body clearly in frame.',
                throttle_duration_sec=3.0)
            # Go-home is handled by a TIMER, not here -- see _maybe_go_home
            # and self._last_detected_time for why a check that only runs
            # inside this callback can never fire when nothing is
            # detected at all.
            return

        # Wrist visibility collapsed for a sustained stretch -- see
        # WRIST_MIN_VISIBILITY_SUSTAINED's own declaration for why this is a
        # SUSTAINED check, not a single-frame one. Treated exactly like the
        # shoulder gate failing: skip publishing this frame's (almost
        # certainly noise-driven) angles, and do NOT refresh
        # _last_detected_time, so _maybe_go_home's existing timer naturally
        # snaps to home if this keeps up past _GO_HOME_AFTER_SEC too.
        if pose_arr[R_WRIST, 3] < self.WRIST_MIN_VISIBILITY_SUSTAINED:
            if self._wrist_low_vis_since is None:
                self._wrist_low_vis_since = time.time()
            elif time.time() - self._wrist_low_vis_since >= self.WRIST_LOW_VIS_SUSTAIN_SEC:
                self.get_logger().warning(
                    f'ARM NOT MOVING: wrist visibility below '
                    f'{self.WRIST_MIN_VISIBILITY_SUSTAINED} for '
                    f'{self.WRIST_LOW_VIS_SUSTAIN_SEC:.1f}s+ '
                    f'(wrist={pose_arr[R_WRIST,3]:.2f}). Get your whole arm, '
                    'not just your shoulder, in frame.',
                    throttle_duration_sec=3.0)
                return
        else:
            self._wrist_low_vis_since = None

        self._last_detected_time = time.time()
        self._went_home = False

        target_arr = pose_arr   # default: live pose, unchanged behaviour
        if self._active_pose_predictor is not None:
            features = self._pose_extractor.extract(pose_arr)
            ready, seq = self._pose_preprocessor.update(features, pose_arr)
            if ready:
                predicted = self._active_pose_predictor.predict(seq)      # (99,) normalised
                target_arr = np.ones((33, 5), dtype=float)                # vis/presence = 1.0
                target_arr[:, :3] = predicted.reshape(33, 3)
                # landmarks_to_joint_angles() and gripper_from_hand() only use
                # relative vector differences/ratios, never absolute position,
                # so the normalised predicted vector can be used directly here
                # (same as how the gripper's predicted hand landmarks are used
                # in perception/inference_node.py) — no denormalisation needed.

        # The go-home SIGN is being held right now -- let
        # _maybe_go_home_gesture (driven off /arm/hand_spread) own the
        # arm this instant instead of overwriting it straight back to the
        # live-tracked pose. Without this, both callbacks published
        # unconditionally on every message, each undoing the other's write
        # a few times a second: the sign's own detection was solid (live
        # capture showed it staying continuously triggered), but the arm
        # never actually settled at HOME_POSE, hovering instead near
        # whatever the live-tracked "hand near face" angles were, with a
        # visible jitter every time the go-home write briefly won the
        # race -- reported directly: "see live why it jitter whn i sho the
        # sign".
        if self._home_gesture_condition():
            return

        # Capture the operator's own neutral wrist angle, waiting for a
        # brief stable period rather than trusting a single frame -- see
        # _l6_baseline's declaration for why. Retries every frame until
        # locked in; harmless since it only runs before the first lock.
        if self._l6_baseline is None:
            cal = l6_calibrate(target_arr)
            if cal is not None:
                buf = self._l6_cal_buf
                buf.append(cal)
                del buf[:-L6_CAL_STABLE_WINDOW]
                self._l6_cal_frames += 1
                stable = (len(buf) >= L6_CAL_STABLE_WINDOW
                         and _circular_spread(buf) < L6_CAL_STABLE_TOL_RAD)
                timed_out = self._l6_cal_frames >= L6_CAL_MAX_FRAMES
                if stable or timed_out:
                    self._l6_baseline = _circular_mean(buf)
                    self.get_logger().info(
                        f'L6 wrist-roll calibrated to operator neutral: '
                        f'{self._l6_baseline:.3f} rad'
                        + (' (timed out waiting for a stable pose)'
                           if timed_out and not stable else ''))

        angles = landmarks_to_joint_angles(
            target_arr, l6_baseline=self._l6_baseline or 0.0)
        if not angles:
            return   # shoulder/elbow/wrist not visible this frame

        human_msg = JointState()
        human_msg.name = list(JOINT_NAMES)
        human_msg.position = [
            self._gripper_angle if j in ('joint_L7_R', 'joint_L7_L')
            else angles.get(j, 0.0)
            for j in JOINT_NAMES
        ]
        self._pub_human_angles.publish(human_msg)

        # Teleop just engaged (CAMERA TELEOPERATION switched on): hold at
        # home while averaging a few frames of the operator's CURRENT raw
        # pose into a reference point, instead of publishing it live. See
        # OFFSET_JOINTS and TELEOP_ENGAGE_FRAMES.
        if self._teleop_engaging:
            self._teleop_engage_buf.append(
                {j: angles[j] for j in self.OFFSET_JOINTS if j in angles})
            if len(self._teleop_engage_buf) >= self.TELEOP_ENGAGE_FRAMES:
                for j in self.OFFSET_JOINTS:
                    vals = [f[j] for f in self._teleop_engage_buf if j in f]
                    if vals:
                        self._teleop_engage_ref[j] = sum(vals) / len(vals)
                self._teleop_engaging = False
                self.get_logger().info(
                    'Teleop engaged -- tracking your movement relative to '
                    f'home ({len(self._teleop_engage_ref)} joints referenced).')
            return

        # Rescale (NOT a flat offset) around the engage reference point, so
        # the arm starts at HOME_POSE but the FULL physical range is still
        # reachable on both sides -- moving from the reference toward the
        # joint's own raw max/min still reaches that joint's hi/lo exactly.
        #
        # A flat offset (published = raw - (ref - HOME_POSE)) was tried
        # first and shipped, then broke L1 badly: whatever raw angle the
        # operator's arm happened to read AT THE INSTANT they clicked the
        # switch became the new "zero", so if that raw reading was already
        # far from HOME_POSE (measured live: ref=-1.0875 vs HOME_POSE=
        # +0.2565 for L1), the flat shift ate almost the ENTIRE opposite
        # side's range -- measured live: raw swinging the full physical
        # -1.7..+1.7 only ever published -0.356..+1.7, i.e. "right" lost
        # 1.34 of its 1.7 rad of real travel while "left" lost nothing
        # (it simply clamped at the limit it was already hitting). Reported
        # directly: "tres problem rotation to the right l1 now!".
        #
        # This piecewise linear remap instead treats [raw_min, ref] and
        # [ref, raw_max] as two separate ranges, each stretched to fill
        # [lo, HOME_POSE] and [HOME_POSE, hi] respectively, so nothing is
        # ever lost on either side regardless of where the operator's arm
        # happened to be at the moment of engaging.
        if self._teleop_engage_ref:
            for j in self.OFFSET_JOINTS:
                if j not in angles or j not in self._teleop_engage_ref:
                    continue
                lo, hi = JOINT_LIMITS[j]
                home = HOME_POSE.get(j, 0.0)
                ref  = self._teleop_engage_ref[j]
                raw  = angles[j]
                if raw >= ref:
                    span = hi - ref
                    angles[j] = home + (raw - ref) / span * (hi - home) if span > 1e-9 else home
                else:
                    span = ref - lo
                    angles[j] = home - (ref - raw) / span * (home - lo) if span > 1e-9 else home

        # Hand landmarks give a better gripper reading than pose fingertips,
        # so override L7 with it. L6 (wrist roll) keeps its pose-derived value.
        angles['joint_L7_R'] = self._gripper_angle
        angles['joint_L7_L'] = self._gripper_angle
        self._have_pose = True
        self._apply_smoothing(angles)
        # Still update self._smoothed above even while idle-parked -- that
        # is what _maybe_go_home_idle's own "moved" comparison reads to
        # notice when real motion resumes, so it can't be skipped without
        # breaking the "wakes up the instant real motion starts" contract.
        # But DO skip publishing it while parked: _maybe_go_home_idle
        # firing once and setting _went_home_idle=True was never enough
        # on its own -- unlike the go-home gesture, which has an explicit
        # stand-down check right here, nothing stopped THIS callback from
        # immediately publishing the next live-tracked (still noisy)
        # frame straight over the just-published home position. Live
        # capture showed exactly that: L1/L2/L3 sitting at HOME_POSE for
        # a 9-second stretch while L5/L6 kept drifting 0.1-0.4 rad the
        # entire time, because their noise alone never happened to
        # exceed IDLE_MOTION_RAD before this fix. Reported directly: "it
        # shoukd be statics when nothing motion there!!!".
        if not self._went_home_idle:
            self._publish(self._smoothed)

    # ── gripper: hand landmarks -> L7 only ─────────────────────────────────────

    def _assert_startup_pose(self):
        """
        Arm zeroed, both jaws open and equal.

        Republished a few times rather than once: DDS discovery can still be
        settling, and losing this single message leaves the gripper resting
        closed with nothing to correct it, because this node otherwise stays
        silent until landmarks arrive.
        """
        # Wait for real joint feedback first. _publish falls back to the target
        # when a joint's current position is unknown, which would make the start
        # point identical to the end point. Not counted as an attempt, so
        # waiting here does not consume the retries.
        if not self._joint_now:
            return

        # No longer gated on control_mode == 'gesture': teleop now starts OFF
        # (MANUAL, a safety default -- see arm_gui.py), so gating this on
        # 'gesture' meant it never fired at all and the arm sat at
        # joint_trajectory_controller's own zeroed default instead of home
        # for however long teleop stayed off. Home is a safe pose to assert
        # into regardless of which mode owns the arm, so this calls
        # _publish_raw directly, bypassing that arbitration gate.
        self._startup_attempts = getattr(self, '_startup_attempts', 0) + 1

        # freeze_joints debugging mode used to skip this ENTIRE routine --
        # every joint, not just the frozen ones -- so isolating L5 with
        # "--freeze=joint_L4,joint_L6" left L1/L2/L3/L5 sitting at
        # joint_trajectory_controller's raw zero default instead of home
        # too, since nothing else was asserting it. Reported directly: a
        # live capture during that isolation showed L5's own published
        # value pinned at exactly 0.0000 for the whole first ~37s of the
        # session, only reaching HOME_POSE (0.2188) once CAMERA
        # TELEOPERATION was clicked and _on_control_mode's own home-snap
        # took over instead. Frozen joints now keep whatever they
        # currently are (self._joint_now) rather than being forced to
        # HOME_POSE, which is the only part of the old skip that still
        # matters -- everything else asserts home like normal.
        pose = {
            j: (self._joint_now.get(j, HOME_POSE.get(j, 0.0)) if j in self._frozen_set
                else HOME_POSE.get(j, 0.0))
            for j in JOINT_NAMES
        }
        pose['joint_L7_R'] = self._gripper_angle
        pose['joint_L7_L'] = self._gripper_angle
        self._smoothed = dict(pose)
        self._publish_raw(pose)

        if self._startup_attempts == 1:
            self.get_logger().info(
                f'Startup pose asserted: arm at HOME_POSE, gripper open at '
                f'{self._gripper_angle:.4f} m (both jaws)')
        if self._startup_attempts >= 5:
            self._startup_timer.cancel()
            self.get_logger().info('Startup pose assertion complete')

    def _on_gripper_opening(self, msg: Float32):
        """Authoritative gripper value, already in metres."""
        self._gripper_angle = float(msg.data)
        # Second path that writes self._smoothed directly, same trap as
        # _assert_startup_pose above: this one drives the gripper jaws from
        # /arm/gripper_opening completely outside _apply_smoothing, so
        # freezing L7 from the GUI switch panel would silently do nothing
        # without this check -- caught before shipping the freeze switch,
        # not after another "it's not working" round.
        if 'joint_L7_R' in self._frozen_set or 'joint_L7_L' in self._frozen_set:
            return
        if self._have_pose and self._smoothed:
            self._smoothed['joint_L7_R'] = self._gripper_angle
            self._smoothed['joint_L7_L'] = self._gripper_angle
            self._publish(self._smoothed)

    def _on_hand_spread(self, msg: Float32):
        """Drives the go-home gesture -- see HAND_HOME_SPREAD_MIN."""
        self._hand_spread = float(msg.data)
        self._update_home_gesture_streak()
        self._maybe_go_home_gesture()

    def _home_gesture_raw(self) -> bool:
        """Instantaneous fingers-spread reading, no debounce. See
        _home_gesture_condition for why this alone isn't used directly."""
        return self._hand_spread >= self.HAND_HOME_SPREAD_MIN

    def _update_home_gesture_streak(self):
        """
        Called once per /arm/hand_spread message (the gesture's own
        driving rate), before anything reads _home_gesture_condition this
        tick. Tracks the wall-clock time the CURRENT continuous hold
        started; any single false reading resets it to None immediately,
        so RELEASING the sign is still instant -- only ENGAGING waits out
        HOME_GESTURE_CONFIRM_SEC (see its own declaration for why this is
        time-based, not a tick count).
        """
        if self._home_gesture_raw():
            if self._home_gesture_hold_since is None:
                self._home_gesture_hold_since = time.time()
        else:
            self._home_gesture_hold_since = None

    def _home_gesture_condition(self) -> bool:
        """
        True while the fingers-spread "go home" sign has been held
        continuously for HOME_GESTURE_CONFIRM_SEC seconds. Shared by
        _on_pose_landmarks (which must stand down while this holds, see
        its own call site) and _maybe_go_home_gesture (which acts on it)
        so the two can never disagree about whether the sign
        is currently active. Read-only -- _update_home_gesture_streak is
        the only thing that advances _home_gesture_hold_since, called
        once per /arm/hand_spread message so both callers here always
        see the same, single, current answer regardless of which one
        asks first.
        """
        if self._home_gesture_hold_since is None:
            return False
        return (time.time() - self._home_gesture_hold_since) >= self.HOME_GESTURE_CONFIRM_SEC

    def _maybe_go_home_gesture(self):
        """
        Explicit "go home" hand gesture -- separate from both other
        go-home paths (not-detected, and the general hold-any-pose-still
        idle timeout). Driven by /arm/hand_spread (see HAND_HOME_SPREAD_MIN
        for what it measures and why), not gated by L7's own freeze
        state -- this is a command, not a gripper-follow action, so
        freezing the gripper jaws should not also disable it.

        Immediate, every frame, no hold-time -- a 1s confirm delay (an
        earlier version of this gesture) was reported directly as wrong:
        "i want it to home position iimeadiately... not waiting secs or
        it is realitime. it is the starting of the teleopration standarad
        psotion". Publishes home on EVERY frame the condition reads true,
        not once per episode, so it functions as a live hold: showing
        this sign pins the arm at home for as long as it's held, and
        normal tracking resumes the instant the fingers relax again, no
        lag either direction.

        See HAND_HOME_SPREAD_MIN's own declaration for why this checks
        fingers-spread-apart rather than hand openness -- two earlier
        openness-based versions of this gesture both caused real
        problems (blocking movement, then a sudden-drop jerk), because an
        open hand with fingers together is this operator's normal resting
        shape during ordinary tracking, not a distinct sign.
        """
        if not self._home_gesture_condition():
            self._went_home_gesture = False
            return
        # L7 excluded from the HOME_POSE fallback the other two go-home
        # paths use (which defaults an unlisted joint to 0.0, i.e. CLOSED)
        # -- spreading the fingers apart to trigger this also reads as a
        # wide-open hand on the gripper's own openness metric, so snapping
        # it shut the instant this fires would look like it contradicts
        # the gesture that just caused it. Left at its current (open)
        # reading instead.
        home = {j: (self._smoothed.get(j, 0.0) if j in self._frozen_set
                     else HOME_POSE.get(j, 0.0))
                for j in JOINT_NAMES}
        home['joint_L7_R'] = self._gripper_angle
        home['joint_L7_L'] = self._gripper_angle
        self._smoothed = home
        self._publish(home)
        if not self._went_home_gesture:
            self.get_logger().info(
                'Open-hand home gesture detected -- arm held at home.')
            # _on_pose_landmarks returns before _apply_smoothing runs for
            # as long as this gesture holds (see its own call site), so
            # the median-filter history and L6's jump-confirm state stop
            # updating entirely during the hold -- they still hold
            # whatever raw readings were in them from BEFORE the sign
            # was shown. Left alone, the instant the sign releases those
            # stale pre-sign entries are still sitting in the buffer
            # alongside the first fresh post-release readings, so the
            # median (and L6's confirm check) is computed over a MIX of
            # old and new for the next few frames, pulling the arm back
            # toward roughly where it was before the sign instead of
            # starting clean from home. Reported directly: "after go to
            # home... the robot move again to back to older position
            # after home sign end... why it not started from the home
            # aposition". Cleared here, once, right as the hold begins
            # (not every frame) -- by release time the buffers are empty
            # and fill from scratch with only genuinely-new readings.
            self._raw_history.clear()
            self._jump_pending.clear()
        self._went_home_gesture = True

    def _on_control_mode(self, msg: String):
        mode = msg.data.strip().lower()
        if mode not in ('gesture', 'manual'):
            self.get_logger().warn(f'Unknown control mode "{msg.data}" — ignoring')
            return
        if mode != self._control_mode:
            self._control_mode = mode
            self.get_logger().info(
                'Control mode: GESTURE — driving the arm' if mode == 'gesture'
                else 'Control mode: MANUAL — standing down, the GUI owns the arm')
            if mode == 'gesture':
                # Flipping to CAMERA TELEOPERATION previously only set the
                # mode flag -- the arm kept whatever position it already
                # happened to be at (e.g. wherever MANUAL mode's sliders
                # left it) instead of starting from home. Reported directly
                # with a screenshot of L3 reading -0.375 right at the start
                # of teleoperation instead of the true home, -0.35: "i want
                # position of the l3 at the begiining of teloperation
                # camrea value is in the middle of the slider". Same
                # frozen-joint-aware pattern as the other go-home paths
                # (_maybe_go_home, _maybe_go_home_idle,
                # _maybe_go_home_gesture) and _assert_startup_pose's own L7
                # override -- HOME_POSE has no L7 entry, so leaving it out
                # of the fallback would snap the gripper shut.
                home = {j: (self._smoothed.get(j, 0.0) if j in self._frozen_set
                             else HOME_POSE.get(j, 0.0))
                        for j in JOINT_NAMES}
                home['joint_L7_R'] = self._gripper_angle
                home['joint_L7_L'] = self._gripper_angle
                self._smoothed = home
                self._publish(home)
                self._went_home = False
                self._went_home_idle = False
                self._idle_ref_values = {}
                self._went_home_gesture = False
                # Arm a fresh engage-relative reference capture (see
                # OFFSET_JOINTS) so the arm holds at home while the next
                # few frames are averaged into the new offset, instead of
                # snapping to wherever the operator's real arm currently
                # is. Clear the median/jump-detector history too -- same
                # stale-buffer trap as the home-gesture release fix: without
                # this, MANUAL mode's live (but unpublished) tracking keeps
                # filling that history the whole time teleop is off, so the
                # very first post-engage median would already be mostly the
                # operator's current real pose regardless of this offset.
                self._teleop_engaging   = True
                self._teleop_engage_buf = []
                self._teleop_engage_ref = {}
                self._raw_history.clear()
                self._jump_pending.clear()

    # joint_L1 = atan2(dx, -dz) over the elbow->wrist vector has a real
    # discontinuity whenever -dz crosses zero (forearm roughly perpendicular
    # to the camera), and MediaPipe/depth-fusion's z estimate is noisy
    # enough to sit near it often. Measured directly across 5 live sessions
    # (2026-09-03, real camera + real operator): 24-48% of L1 commands
    # landed on the EXACT joint limit, meaning the raw angle was jumping
    # straight from one end of the range to the other. Uniform EMA smoothing
    # alone does not fix this -- at alpha=0.5 a single bad frame still moves
    # the smoothed output by half of a 3.4rad swing (~1.7rad) in one step,
    # which reads as a violent jerk, not damped noise.
    #
    # A hard per-update step cap on L1 turns that teleport into a bounded
    # move in the right direction instead: wrong for one update, corrected
    # over the next few as good frames arrive, and never a visible snap.
    # Not applied to other joints -- this is a property of L1's specific
    # formula, not general sensor noise, and capping joints that do not
    # have this failure mode would only add lag for no benefit.
    #
    # L6 dropped from this dict after its formula was rebuilt to measure
    # roll relative to the forearm instead of the raw camera image (see
    # motion_mapping._L6_wrist_roll): the 20deg/update cap was sized for
    # the OLD formula's 280deg worst-case jumps and, once that was fixed,
    # was just throttling a joint that no longer needed it -- reported
    # directly: "L6 is not sensitive when rotating my wrist". At ~2.5-3.5Hz
    # perception, 20deg/update cannot track a real wrist twist done in
    # under a second. Re-checked worst-case with the cap removed (deadband
    # still active): 55.4deg on the same real footage that used to hit
    # 280deg raw -- a real, fast step, not a snap back to the old bug.
    # L2/L5 added after the shoulder-only gate simplification let elbow/
    # wrist noise reach every arm joint directly (see the gate's own long
    # comment in _on_pose_landmarks) -- watched the robot visibly whip to
    # an extreme, twisted pose ("wtf?", with a screenshot) while L1 and L5
    # were both reading velocity right at the URDF's 3 rad/s hard limit,
    # meaning the commanded jump was large enough to saturate it. L1
    # already had a cap; L5 had NONE -- a single noisy frame could snap it
    # instantly to any value with nothing to stop it. Same 0.35 rad
    # (~20deg/update) as L1/L4, not yet independently measured against
    # real footage the way those were -- a reasonable starting point given
    # the same underlying signal class, revisit if it costs L5
    # responsiveness the way L6's own step cap once did (see below).
    #
    # L6 deliberately NOT included here even though it lost its own step
    # cap for a different reason -- see that removal's own note just
    # below: it already carries median-3 + JUMP_CONFIRM_RAD + a formula
    # rebuilt to be inherently less spiky, and a cap on top of those was
    # measured to cost real responsiveness ("L6 is not sensitive").
    #
    # L3 added after watching it happen live: a single elbow reading
    # jumped to a wildly different position for exactly one frame (high
    # reported confidence, 0.96 -- the visibility gate does not catch
    # this, it is a genuine glitch in the tracked position, not a
    # confidence drop), and L3 swung from -1.498 to +0.088 rad in
    # response -- nearly its whole range, in about a second. L3 already
    # has median-5 (see MEDIAN_WINDOW), which delays a spike but does not
    # bound how far a SINGLE accepted frame can move the output -- a step
    # cap is the complementary protection median filtering does not
    # provide. Same 0.35 rad default as the others above, same caveat:
    # not yet independently measured against real footage.
    MAX_STEP_RAD = {'joint_L1': 0.35, 'joint_L4': 0.35, 'joint_L2': 0.35,
                    'joint_L5': 0.35, 'joint_L3': 0.35}   # ~20 deg per update

    # Default alpha=0.5 halves the gap to the true value every update, so a
    # rotation completed in the 2-3 updates a fast real gesture gets (at
    # this project's ~2.5-3.5Hz perception rate) never catches up. User
    # report: "l6 can rotate more angles". This is a real range-vs-jump
    # tradeoff, not a free improvement -- swept across 3 real recordings:
    #
    #   alpha   avg range recovered   worst single-step jump
    #    0.50          63%                  119 deg
    #    0.35          69%                  155 deg
    #    0.10          81%                  251 deg
    #
    # 0.1 (this project's first instinct) was rejected: 251deg is close to
    # the old broken formula's own 280deg worst case, i.e. nearly giving
    # back the jump the whole L6 rebuild was for. 0.35 recovers a real
    # amount of range (63%->69%) for a much smaller jump increase
    # (119->155deg) -- a deliberately moderate choice pending the operator
    # trying it live, not the largest number tested.
    # L2/L3/L5 alpha=0.8 (tried, then reverted): pushed to counter what
    # looked like sustained drift in one live capture while "sitting
    # still", but cost a ~2.4s settling time on genuine gestures too --
    # reported directly right after: "it behave like shit". The earlier
    # median-5 fix (see MEDIAN_WINDOW) had ALREADY been confirmed live to
    # hold rock-steady once the operator was genuinely idle, in a separate
    # capture -- so the "sustained drift" that prompted this alpha change
    # was more likely real, if subtle, motion in that one capture than
    # noise the EMA needed to suppress further. Not worth this much
    # responsiveness cost for a problem the median fix already covers.
    ALPHA_OVERRIDE = {'joint_L6': 0.35}

    # Lowering L6's alpha to fix "not sensitive" let more of the raw
    # signal's own noise through too -- reported directly: "the l6 motion
    # is like dancing crazy". A median PRE-filter, applied to the raw
    # reading before the EMA blend above, rejects single-frame noise
    # spikes (the back-and-forth "dancing") far more effectively than
    # raising alpha back up would, without alpha's blanket lag penalty on
    # genuine motion too.
    #
    # window=3 shipped first as the smaller change, but the user reported
    # it still jittering ("make it smooth!") after also switching on L6's
    # baseline calibration (94fbd3e), so re-swept window vs alpha together
    # across all 3 hand-visible recordings post-calibration (avg wiggle
    # score / avg %% of joint range still reached):
    #
    #   alpha  median   wiggle   range%
    #    0.35    3       1745      63
    #    0.35    5        935      60   <- window is the bigger lever;
    #    0.35    7        563      57      raising alpha barely moves
    #    0.50    5        716      58      wiggle at a given window
    #
    # window=5 shipped on the strength of that table, but a median filter
    # doesn't just reduce jitter -- it delays every real step too, since
    # it needs a MAJORITY of the window to already reflect a change
    # before its own output moves. Measured that delay directly (0.35
    # alpha, simulated step change at this project's ~2.5-3.5Hz
    # perception rate, ~350ms/update):
    #
    #   median   first visible move   90%-settled
    #      3          ~700ms            ~1050ms
    #      5         ~1050ms            ~1400ms
    #      7         ~1400ms            ~1750ms
    #
    # window=5's extra ~350ms before ANY movement is enough to make a
    # quick real rotation look like it never registered at all -- reported
    # directly: "now not moving good". Reverted to window=3: the wiggle
    # score at 3 (1745) is worse than 5's (935), but this project has now
    # hit the wall on this specific tradeoff from BOTH directions --
    # smoother reads as slower, faster reads as jittery, and no window
    # size sidesteps that. A real fix needs less noise at the source
    # (L6 still reads Pose's crude fingertip landmarks, not real
    # MediaPipe Hand tracking), not another turn of this dial.
    #
    # L1 gets the same window=3 treatment as of this change, for a
    # different reason than L6: L1 is the ROOT of the kinematic chain, so
    # its own noise doesn't just wiggle L1 -- every downstream link (L2
    # through the gripper) physically inherits L1's rotation, so L1 jitter
    # reads as the WHOLE ARM shaking, out of proportion to L1's own share
    # of the noise. Reported directly: "each of joint is combined... they
    # moving together". Isolation testing (fixed shoulder+elbow, wrist/hand
    # randomised 200x) confirmed L1's math has zero contamination from
    # wrist or hand landmarks -- the "combined" look is this chain-root
    # amplification, not a mapping bug (L1 = atan2 of shoulder->elbow only,
    # see _L1_base_yaw). L1 already had MAX_STEP_RAD as jump protection
    # (unlike L6, which needed JUMP_CONFIRM_RAD instead), so only the
    # median pre-filter was missing.
    #
    # Measured on real footage (MediaPipe PoseLandmarker, all 7 recordings,
    # current filter -- EMA 0.5 + step-cap -- vs current + median-3):
    #   wiggle -18% to -37% across all 7 recordings (median ~-28%)
    #   range  preserved on 6/7; recording 6 saw a larger drop (190->92deg)
    #     that traces to a single detection glitch, not lag -- median is
    #     robust to exactly that kind of one-frame outlier by construction.
    # Same step-response cost L6's own window=3 already paid and this
    # project already accepted: +350ms to first visible move, +350ms to
    # 90%-settled (window=5's ADDITIONAL 350ms on top of that is what
    # triggered "now not moving good" -- this stays at the accepted point,
    # not that one).
    #
    # L3 gets the same window=3 treatment for a similar reason to L1: the
    # shoulder-elbow-wrist angle is computed via acos, whose derivative
    # diverges as the dot product approaches +/-1 -- exactly at flex=0
    # (fully bent) and flex=pi (fully straight), the two ends of the
    # gesture. Landmark noise near either end swings the raw angle hard.
    # Reported alongside L3's direction and range bugs: "jittery".
    # L3 had no filtering of any kind before this (no step cap, no
    # median, default EMA only) -- confirmed by grep, not assumed.
    #
    # Measured on real footage (all 7 recordings, current -- EMA 0.5
    # only -- vs current + median-3): wiggle -30% to -39% across all 7
    # (in range with L1's own -18% to -37%), same validation standard.
    #
    # L3 raised from 3 to 5 when the visibility/stability gate in front of
    # it was simplified down to shoulder-only (see _on_pose_landmarks) --
    # elbow/wrist noise that used to be blocked at the gate now reaches
    # this filter directly, so it needs to absorb more on its own.
    #
    # L1 raised from 3 to 5 for the same reason, later: recalibrating both
    # L1_YAW_RIGHT_RANGE_DEG and L1_YAW_LEFT_RANGE_DEG down to this
    # operator's comfortable range (so full rotation doesn't require
    # straining) collapsed the total input span driving the whole +/-1.70
    # output to roughly 45 degrees combined -- ordinary landmark noise
    # that used to move the (post-remap) target a few percent of the
    # range now swings it across a much larger fraction of it. Reported
    # directly: "the sppeed of rotation like not smooth... seems stuck
    # jerky a bit" -- live capture confirmed the post-remap target
    # jumping from -1.70 to +1.70 and back within about a second during
    # otherwise-ordinary movement. Same underlying failure as L3's: a
    # sensitivity-increasing change upstream means the existing filter
    # width is no longer wide enough to absorb what now reaches it.
    # L2 and L5 had NO median filtering at all until now -- EMA only,
    # confirmed by grep, same starting point L3 was in before its own fix
    # above. Reported directly: "the robot moving itslef although im doing
    # nothing... IT SHUD BE STATICS" -- a 20s live capture with the
    # operator genuinely still showed L2 swinging 0.08-0.77 rad and L5
    # 0.11-0.53 rad continuously for the WHOLE window, not brief spikes a
    # step cap or a narrow median would catch (L1's own median-5 correctly
    # rejected outliers and stayed pinned the entire time by contrast).
    # window=5, same value as L1/L3, for the same reason: this is sustained
    # noise, not single-frame glitches, so it needs a window wide enough to
    # actually average it down rather than just reject spikes.
    MEDIAN_WINDOW = {'joint_L6': 3, 'joint_L1': 5, 'joint_L2': 5,
                     'joint_L3': 5, 'joint_L5': 5}

    # L2 (shoulder pitch, raise/lower) is computed from the shoulder->
    # elbow vector's Y and Z -- and this camera has no usable depth (its
    # SDK-level alignment fails, which is why depth fusion was already
    # removed from this whole path, see fuse_depth's own notes). With
    # only MediaPipe's own coarse monocular Z, rotating the arm (L1)
    # measurably changes what LOOKS like pitch, even with the shoulder
    # genuinely staying at the same height -- live capture: L1 sweeping
    # -0.15 to +1.40 rad had L2 swinging -0.47 to +0.28 in the same
    # window, which then dragged L3 down to -1.10 through the L2-raise
    # blend just below, none of it a real raise. Reported directly:
    # "check my arm rise and check that movemnt not logically with
    # pysics!".
    #
    # First attempt at a fix, not a final calibration: while L1's OWN
    # raw target is moving fast (>= L1_FAST_ROTATION_RAD per update),
    # clamp L2's step to L2_ROTATION_DAMPED_STEP_RAD instead of its
    # normal MAX_STEP_RAD entry -- lets a genuine slow raise through at
    # full speed, but resists the rotation-induced pitch artifact long
    # enough for it to matter less. Briefly raised to 0.20 while chasing a
    # 13-second L2 lag, but that lag showed up while L1/L2 were ALSO going
    # through the (now removed, see OFFSET_JOINTS) engage-relative rescale
    # -- the rescale's own reference-point mismatch was very likely
    # compounding with this cap to produce a lag far worse than 0.08 alone
    # ever caused. Reverted back to 0.08, the exact value confirmed working
    # in arm_config_backup.json's "98% confirmed" reference snapshot, now
    # that L1/L2 are absolute again and the compounding factor is gone.
    L1_FAST_ROTATION_RAD = 0.35
    L2_ROTATION_DAMPED_STEP_RAD = 0.08

    # How far a single update may move before it must be confirmed by the
    # next frame -- see the gate in _apply_smoothing. 25 deg sits above
    # this joint's measured typical step (1.7-3.2 deg, so ordinary motion
    # never touches the gate) and below its spike range (38-82 deg).
    # Measured on 3 real recordings, current filter vs current + gate:
    #   wiggle 1461->526, 1012->495, 1515->376  (-64%, -51%, -75%)
    #   median step 2.4->1.6, 1.7->1.3, 3.2->1.2 deg (normal motion got
    #     SMOOTHER too, not just spikes clipped)
    #   range 146->142, 126->115, 175->136 deg (real motion preserved)
    #
    # joint_L1 was added here briefly (a JUMP_CONFIRM gate at 45deg) to
    # chase a live-captured atan2(dx,-dz) discontinuity during one specific
    # centred-gesture pose ("why it rotate more to the right?"). Reverted
    # immediately -- it made the arm noticeably worse overall in normal use
    # ("previous is better! whyyyy... i want only to adjust the rotation of
    # l1 for that gesture"). Back to exactly the JSON reference state
    # (L6 only). If that specific gesture's L1 instability needs revisiting,
    # do it as a narrower fix scoped to that pose, not a blanket gate on
    # every L1 update.
    JUMP_CONFIRM_RAD = {'joint_L6': math.radians(25)}

    def _apply_smoothing(self, angles: dict):
        if not self._smoothed:
            self._smoothed = dict(angles)
            if self._frozen_set:
                # Otherwise a frozen joint's baseline is whatever it
                # happened to compute from the live, raw arm position on
                # this exact first frame -- not necessarily a clean pose,
                # since the operator may not have been in position yet.
                # That produced a one-time, unwanted settling move:
                # reported directly, "L1 move once at the start... i dont
                # want it." Freezing to a known, fixed value (home = 0.0)
                # instead means genuinely zero movement for the whole
                # test, not "zero movement after one unpredictable
                # correction."
                for j in self._frozen_set:
                    self._smoothed[j] = 0.0
            return
        # joint_L1 is atan2 of the upper arm's horizontal (X,Z) projection.
        # An arm hanging near-vertically at rest -- including the operator
        # just standing still -- puts that projection near zero, where
        # atan2 is dominated by landmark noise rather than real geometry:
        # measured 73 degrees of noise-driven std at 1cm magnitude,
        # dropping to 9 degrees by 10cm (L1_MIN_MAG). Below that, HOLD the
        # last value rather than blend in an angle that is mostly noise --
        # "static, but the robot base keeps drifting" is exactly this.
        # Same treatment for L4 (forearm roll -- degenerate near full arm
        # extension) and L6 (wrist roll -- degenerate when the hand is
        # nearly edge-on to the camera). Different geometry per joint,
        # same underlying failure: an atan2 whose inputs went small.
        #
        # L2/L3 read L1's own _L1_mag (same shoulder->elbow dx,dz feeds
        # _L2_base_pitch's atan2(dy, sqrt(dx^2+dz^2)) denominator, so the
        # same near-zero-denominator singularity hits both at once) but
        # gate at L2_MIN_MAG, NOT L1_MIN_MAG -- they used to share L1's
        # 0.18 outright, which stopped the wild-swing bug below but also
        # held all three joints frozen together for 10+ second stretches
        # during completely ordinary movement, since this operator's
        # normal working magnitude apparently sits in the 0.05-0.18 band a
        # lot, not just in the genuinely-degenerate case. Reported
        # directly: "the robot arm is not follow my hand. rotation l1 also
        # broken, and the l3 is not following". A live capture of the
        # actual bug this gate exists for (see L2_MIN_MAG's own
        # declaration in motion_mapping.py) showed the worst wild swings
        # (65-85 degrees of pure noise, enough to clamp L2 -- and via the
        # raise-blend above, L3 too -- to a JOINT_LIMITS extreme and get
        # stuck there) concentrated below ~0.10, while 0.10-0.18 ran
        # elevated but bounded (55-65 degrees, nowhere near the extreme).
        # L2_MIN_MAG=0.10 catches the former and lets ordinary smoothing
        # handle the latter, so L1 alone still gets the wider, previously-
        # tuned protection its own arm-drop failure needs.
        skip = {
            # L1 also has a wrist fallback now (see motion_mapping's
            # _L1_base_yaw) -- only hold L1 when BOTH the shoulder-elbow
            # AND shoulder-wrist vectors are too degenerate to trust,
            # not just the first one. L2/L3 deliberately still gate on
            # _L1_mag alone (unrelated concern -- their OWN pitch/flexion
            # geometry off the same shoulder-elbow vector, see
            # L2_MIN_MAG's own notes) so a usable wrist reading for L1
            # does not silently un-hold L2/L3 on a vector that is still
            # just as degenerate for them.
            'joint_L1': (angles.get('_L1_mag', 1.0) < L1_MIN_MAG
                        and angles.get('_L1_wrist_mag', 1.0) < L1_MIN_MAG),
            'joint_L2': angles.get('_L1_mag', 1.0) < L2_MIN_MAG,
            'joint_L3': angles.get('_L1_mag', 1.0) < L2_MIN_MAG,
            'joint_L4': angles.get('_L4_mag', 1.0) < L4_MIN_MAG,
            'joint_L6': angles.get('_L6_mag', 1.0) < L6_MIN_MAG,
        }
        # Debug freeze: whatever's in _frozen_set holds, everything else
        # stays fully live. See freeze_joints' declaration for why this
        # exists.
        l1_raw_step = 0.0   # set while processing joint_L1 below, read by joint_L2
        for j in JOINT_NAMES:
            if skip.get(j) or j in self._frozen_set:
                continue

            if j == 'joint_L1' and angles.get('_l1_face_override', False):
                # Hard snap, bypassing median/step-cap entirely -- see
                # motion_mapping.py's L1_FACE_OVERRIDE_DIST for why this
                # gesture forces L1 to exactly 0. Routing 0.0 through the
                # normal pipeline below (median-5 + 0.35 rad/update step
                # cap) would still take several frames to visibly reach 0
                # depending on where L1 was before the gesture engaged --
                # exactly what "still not home" during that catch-up
                # window looks like. Reported directly, after the override
                # was already confirmed working offline against real
                # footage: "im tired... still same, it douse not start
                # from home position... it start from the robot arm
                # slanting prosition". History cleared too, so a later
                # genuine rotation (once the gesture releases) does not
                # have to fight a median window full of stale zeros.
                self._smoothed[j] = 0.0
                self._raw_history.pop(j, None)
                self._jump_pending.pop(j, None)
                continue

            old = self._smoothed[j]
            raw = angles[j]

            # Confirm-before-committing gate. The measured problem on L6
            # was never the TYPICAL step (already 1.7-3.2 deg, smooth) --
            # it was occasional 38-82 deg spikes. Neither dial already on
            # this joint catches those: a median-of-3 only rejects a spike
            # if it is a single bad frame out of three, and EMA at 0.35
            # passes 65% of any spike straight through. Raising either one
            # far enough to catch them costs lag on ALL motion, which this
            # project already tried and the operator rejected ("now not
            # moving good", f2696a3).
            #
            # A large jump is instead HELD for one frame and only accepted
            # if the next frame agrees with it. Noise does not persist;
            # a real fast rotation does. So normal motion pays nothing
            # (small steps bypass this entirely) and only large moves pay
            # one frame -- a targeted cost, not a blanket one.
            conf = self.JUMP_CONFIRM_RAD.get(j)
            if conf is not None:
                pend = self._jump_pending.get(j)
                if abs(math.atan2(math.sin(raw - old),
                                  math.cos(raw - old))) > conf:
                    agrees = pend is not None and abs(math.atan2(
                        math.sin(raw - pend), math.cos(raw - pend))) <= conf
                    if not agrees:
                        self._jump_pending[j] = raw   # hold, await confirmation
                        continue
                self._jump_pending[j] = None

            win = self.MEDIAN_WINDOW.get(j)
            if win:
                hist = self._raw_history.setdefault(j, [])
                hist.append(raw)
                del hist[:-win]                 # keep only the last `win`
                raw = float(np.median(hist))
            if j == 'joint_L1':
                l1_raw_step = abs(raw - old)
            alpha = self.ALPHA_OVERRIDE.get(j, self._alpha)
            new = alpha * old + (1 - alpha) * raw
            cap = self.MAX_STEP_RAD.get(j)
            # See L1_FAST_ROTATION_RAD's own declaration: L2's perceived
            # pitch is contaminated by fast L1 rotation on this camera
            # (no usable depth), so while L1 is actively moving fast,
            # clamp L2 much tighter than its normal cap.
            if j == 'joint_L2' and l1_raw_step >= self.L1_FAST_ROTATION_RAD:
                cap = self.L2_ROTATION_DAMPED_STEP_RAD
            if cap is not None:
                step = max(-cap, min(cap, new - old))
                new = old + step
            self._smoothed[j] = new

    # ── publish trajectory ────────────────────────────────────────────────────

    def _publish(self, angles: dict):
        # Arbitration: arm_gui.py commands the same robot_arm_controller, and
        # two sources publishing at once make the arm jerk between them.
        # /control_mode names the single owner; stay silent unless it is us.
        if self._control_mode != 'gesture':
            # Throttled, for the same reason as the visibility gate above:
            # silently dropping every trajectory looks exactly like broken
            # hardware from the outside. If the Arm GUI is in MANUAL, that
            # is a deliberate choice, but it should never be an invisible
            # one.
            self.get_logger().warning(
                f'ARM NOT MOVING: control mode is "{self._control_mode}", '
                'not "gesture" -- the Arm GUI owns the arm. Switch it to '
                'GESTURE to drive from the camera.',
                throttle_duration_sec=3.0)
            return
        self._publish_raw(angles)

    def _publish_raw(self, angles: dict):
        """The actual trajectory send, with no /control_mode gate.

        Split out of _publish so _assert_startup_pose can force the arm to
        HOME_POSE the moment this node comes up, before anyone has touched
        the GUI's mode switch. Teleop now starts OFF (MANUAL) as a safety
        default -- see arm_gui.py's _build_mode_panel -- and _assert_startup_
        pose used to gate on control_mode == 'gesture' too, which meant this
        one-time startup assertion never fired anymore and the arm just sat
        at whatever joint_trajectory_controller's own zeroed default was
        instead of home. Reported directly: "everyrhing initial should be
        starting from home right? ... why it already go to toher position?"
        -- this is the same bug in a new place after the safety-switch fix.
        """
        msg             = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        target          = [float(angles[j]) for j in JOINT_NAMES]

        # TWO points: where we are now, then where we want to be.
        #
        # With a single end point, joint_trajectory_controller silently does
        # nothing when the arm joints are already at their targets and only
        # the gripper differs -- measured directly: "arm 0, gripper 0->0.011"
        # never moved at either 40ms or 2s, while the same gripper change
        # alongside any arm motion moved both jaws to 0.011 exactly. That is
        # why the gripper ignored its startup pose and appeared to lag or
        # freeze whenever the operator held still: gripper-only commands were
        # being dropped. Supplying the current position as an explicit start
        # gives the controller an unambiguous path and it executes every time.
        start = JointTrajectoryPoint()
        start.positions = [float(self._joint_now.get(j, angles[j])) for j in JOINT_NAMES]
        start.time_from_start = Duration(sec=0, nanosec=10_000_000)   # 10ms

        end = JointTrajectoryPoint()
        end.positions = target
        end.time_from_start = Duration(
            sec=self._dt_ms // 1000,
            nanosec=(self._dt_ms % 1000) * 1_000_000,
        )

        # joint_trajectory_controller ignores a trajectory unless some ARM
        # joint actually changes -- measured directly: "arm already at target,
        # gripper 0 -> 0.011" never moves, at 40ms or 2s, with one point or
        # two, while the identical gripper change alongside arm motion moves
        # both jaws every time. Live, that means holding your arm still and
        # only opening your hand did nothing: the gripper command was dropped.
        #
        # joint_L2 is the carrier. Alternating it by 0.001 rad, well under
        # any visible motion, makes every trajectory a real one without
        # disturbing the pose the operator is commanding.
        #
        # Used to be joint_L5, back when L5 was a permanent stub always at
        # 0.0 (wrist pitch "not modelled") -- overwriting it cost nothing,
        # since nothing real was there. L5 is now a real, live-computed
        # joint (see motion_mapping._L5_wrist_pitch), and this carrier was
        # still OVERWRITING it wholesale on every publish, silently
        # discarding the operator's actual wrist-bend command every single
        # time. Reported directly: "joint 5 is not moving... make it
        # movee, it is pose" -- the computation was fine (verified: -36 to
        # +54 degrees on real footage), the dither was erasing it one line
        # before it ever reached the robot.
        #
        # L2 has no deadband/calibration machinery of its own (unlike L1,
        # L4, L6), so a perturbation two orders of magnitude below this
        # project's smallest measured noise floor (L1's 9deg minimum) has
        # nowhere to interact badly. Added, not assigned, as extra
        # insurance against ever doing this again to a joint that matters.
        i2 = JOINT_NAMES.index('joint_L2')
        self._dither = -getattr(self, '_dither', 0.001)
        end.positions[i2] += self._dither

        msg.points = [start, end]
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = RobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
