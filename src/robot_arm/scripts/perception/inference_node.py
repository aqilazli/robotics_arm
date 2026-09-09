#!/usr/bin/env python3
"""
inference_node.py  —  ROS 2 node
=================================
Full ML pipeline in one node, running MediaPipe POSE and MediaPipe HANDS
together on every frame — a hybrid split by what each is actually good at:

  Camera → OpenCV → MediaPipe Hands  → FeatureExtractor → Preprocessor
                                      → temporal predictor (LSTM/GRU/Transformer)
                                      → /arm/landmarks (predicted hand pose)

  Camera → OpenCV → MediaPipe Pose   → /arm/pose_landmarks (shoulder/elbow/wrist
                                      → robot_node.py drives arm joints L1-L4
                                      directly via motion_mapping.py, no IK)

Hand landmarks have no shoulder/elbow, so they only ever drove the gripper
(L7, via robot_control/ik_solver.gripper_from_hand) — Pose supplies every
arm joint directly instead (L1-L6), which is what a human's elbow bending
actually corresponds to physically. Tried moving L6 (wrist roll) onto Hand
landmarks three separate times (wrist point, MCP joints, thumb) hoping for
less noise than Pose's own fingertip approximation; measured worse every
time -- the whole Hand model is unreliable at the camera distance this
project's framing requires (arm has to stay in frame too), not a specific
landmark choice. L6 stays on Pose.

Published topics
----------------
  /arm/landmarks       std_msgs/Float32MultiArray  21×5 hand landmarks (predicted)
  /arm/pose_landmarks  std_msgs/Float32MultiArray  33×5 body-pose landmarks (arm control)
  /arm/color_image     sensor_msgs/Image           debug annotated frame (both overlays)
  /ai_inference_ms     std_msgs/Float32            active predictor's model-only inference time
  /e2e_latency_ms      std_msgs/Float32            capture-to-prediction end-to-end latency

Parameters (ros2 run … --ros-args -p key:=value)
  camera_index  int   video device index (-1 = auto-detect)
  fps           int   target capture rate (default 30)
  window_size   int   LSTM sequence length (default 30)
  show_window   bool  show OpenCV debug window (default false)
"""

import os
import sys
import threading
import ctypes
import time

# Allow imports from the scripts/ directory
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault('DISPLAY', ':0')
os.environ['QT_QPA_PLATFORM'] = 'xcb'   # force X11, avoid Wayland Qt issues

import cv2
import numpy as np
import mediapipe as mp

# ── Orbbec SDK depth reader (optional) ───────────────────────────────────────
_SDK_LIB = os.path.join(
    os.path.expanduser('~'), 'Downloads',
    'OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release',
    'OrbbecSDK_v1.10.27', 'SDK', 'lib', 'libOrbbecSDK.so',
)
OB_STREAM_DEPTH = 3
OB_FORMAT_Y11   = 11
OB_FORMAT_Y12   = 12

# This device advertises depth as both Y11 and Y12 (checked via
# ob_pipeline_get_stream_profile_list). Y12 decodes to junk: a whole 640x480
# frame collapses to ~9 distinct values in a 49-83 band, with no scene
# structure, and readings that are not even monotonic in distance
# (300mm->55, 400mm->46.5, 500mm->38, 600mm->42). Y11 on the same scene
# gives ~115 distinct values, a smooth spatial gradient matching the real
# geometry, and a per-pixel temporal std of 0.30. So Y11 is the real depth
# stream on this unit and Y12 is something else (probably IR amplitude).
OB_FORMAT_DEPTH = OB_FORMAT_Y11

import depth_calibration
_DEPTH_CALIB = depth_calibration.load()
if _DEPTH_CALIB:
    print(f'[InferenceNode] depth calibration active: '
          f'true = (reported - {_DEPTH_CALIB[1]:.2f}) / {_DEPTH_CALIB[0]:.5f}')
else:
    print('[InferenceNode] no depth calibration saved, using raw sensor values '
          '(run the Depth Calibration tool to fit one)')
COLORMAPS      = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_HOT]

_PREDICTOR_FILES = {
    'lstm':        'lstm_predictor.pt',
    'gru':         'gru_predictor.pt',
    'transformer': 'transformer_predictor.pt',
}


def _load_orbbec_sdk():
    """Load and configure OrbbecSDK. Returns lib or None on failure."""
    if not os.path.exists(_SDK_LIB):
        return None
    try:
        sdk_dir = os.path.dirname(_SDK_LIB)
        for f in os.listdir(sdk_dir):
            if '.so' in f:
                try: ctypes.CDLL(os.path.join(sdk_dir, f))
                except OSError: pass
        lib = ctypes.CDLL(_SDK_LIB)
        PP = ctypes.POINTER(ctypes.c_void_p)
        lib.ob_create_pipeline.restype  = ctypes.c_void_p
        lib.ob_create_pipeline.argtypes = [PP]
        lib.ob_delete_pipeline.restype  = None
        lib.ob_delete_pipeline.argtypes = [ctypes.c_void_p, PP]
        lib.ob_create_config.restype  = ctypes.c_void_p
        lib.ob_create_config.argtypes = [PP]
        lib.ob_delete_config.restype  = None
        lib.ob_delete_config.argtypes = [ctypes.c_void_p, PP]
        lib.ob_config_enable_video_stream.restype  = None
        lib.ob_config_enable_video_stream.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, PP]
        lib.ob_pipeline_start_with_config.restype  = None
        lib.ob_pipeline_start_with_config.argtypes = [ctypes.c_void_p, ctypes.c_void_p, PP]
        lib.ob_pipeline_stop.restype  = None
        lib.ob_pipeline_stop.argtypes = [ctypes.c_void_p, PP]
        lib.ob_pipeline_wait_for_frameset.restype  = ctypes.c_void_p
        lib.ob_pipeline_wait_for_frameset.argtypes = [ctypes.c_void_p, ctypes.c_uint32, PP]
        lib.ob_frameset_depth_frame.restype  = ctypes.c_void_p
        lib.ob_frameset_depth_frame.argtypes = [ctypes.c_void_p, PP]
        lib.ob_frame_data.restype  = ctypes.c_void_p
        lib.ob_frame_data.argtypes = [ctypes.c_void_p, PP]
        lib.ob_frame_data_size.restype  = ctypes.c_uint32
        lib.ob_frame_data_size.argtypes = [ctypes.c_void_p, PP]
        lib.ob_video_frame_width.restype  = ctypes.c_uint32
        lib.ob_video_frame_width.argtypes = [ctypes.c_void_p, PP]
        lib.ob_video_frame_height.restype  = ctypes.c_uint32
        lib.ob_video_frame_height.argtypes = [ctypes.c_void_p, PP]
        lib.ob_depth_frame_get_value_scale.restype  = ctypes.c_float
        lib.ob_depth_frame_get_value_scale.argtypes = [ctypes.c_void_p, PP]
        lib.ob_delete_frame.restype  = None
        lib.ob_delete_frame.argtypes = [ctypes.c_void_p, PP]
        lib.ob_delete_error.restype  = None
        lib.ob_delete_error.argtypes = [ctypes.c_void_p]
        return lib
    except Exception as e:
        print(f'[InferenceNode] Orbbec SDK load failed: {e}')
        return None


def _orbbec_depth_mm(lib, frame) -> np.ndarray | None:
    err = (ctypes.c_void_p * 1)(None)
    w   = lib.ob_video_frame_width(frame, err)
    h   = lib.ob_video_frame_height(frame, err)
    sz  = lib.ob_frame_data_size(frame, err)
    ptr = lib.ob_frame_data(frame, err)
    sc  = lib.ob_depth_frame_get_value_scale(frame, err)
    if not ptr or sz < 2:
        return None
    raw = np.frombuffer(
        (ctypes.c_uint16 * (sz // 2)).from_address(ptr), dtype=np.uint16
    ).copy()
    depth = raw.reshape((h, w)).astype(np.float32) * sc
    # This unit's raw depth is mis-scaled by a large but consistent affine
    # factor; data_collection/depth_calibration_gui.py fits the correction.
    # Every depth consumer (live control, the recorder, fuse_depth) reads
    # through here, so correcting once here covers all of them.
    if _DEPTH_CALIB is not None:
        depth = depth_calibration.apply(depth, _DEPTH_CALIB)
    return depth


def _colorize_depth(depth_mm, colormap=cv2.COLORMAP_TURBO, max_mm=3000):
    valid = (depth_mm > 0) & (depth_mm < max_mm)
    norm  = np.zeros(depth_mm.shape, dtype=np.uint8)
    norm[valid] = np.clip(depth_mm[valid] / max_mm * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, colormap)
    vis[~valid] = 0
    return vis

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Float32, Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image

# ── ML pipeline components ────────────────────────────────────────────────────
from data_processing.feature_extractor import FeatureExtractor
from data_processing.preprocessor      import Preprocessor

# ── MediaPipe setup ───────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode

_DEFAULT_MODEL_PATH      = os.path.join(_SCRIPTS, 'perception', 'hand_landmarker.task')
_DEFAULT_POSE_MODEL_PATH = os.path.join(_SCRIPTS, 'perception', 'pose_landmarker_full.task')

# 21-point hand skeleton connections (standard MediaPipe Hands topology)
_HAND_IDX  = list(range(21))
_HAND_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

# Arm landmark connections for overlay (right arm: shoulder-elbow-wrist)
_ARM_IDX  = [12, 14, 16]
_ARM_CONN = [(12, 14), (14, 16)]


def _vis(lm) -> float:
    """Hand landmarks don't report visibility/presence — treat as always-visible."""
    v = getattr(lm, 'visibility', None)
    return v if v is not None else 1.0


class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter('mp_model_path',      _DEFAULT_MODEL_PATH)
        self.declare_parameter('pose_model_path',    _DEFAULT_POSE_MODEL_PATH)
        self.declare_parameter('camera_index',  -1)
        self.declare_parameter('camera_fps',    30)    # camera capture fps
        self.declare_parameter('camera_width',  640)   # capture resolution width
        self.declare_parameter('camera_height', 480)   # capture resolution height
        self.declare_parameter('display_fps',   30)    # monitor window refresh rate
        self.declare_parameter('mediapipe_fps', 30)    # max MediaPipe inference rate
        self.declare_parameter('depth_fps',     15)    # depth polling rate
        self.declare_parameter('depth_max_mm',  3000)  # depth colormap range
        self.declare_parameter('window_size',   30)
        self.declare_parameter('use_depth',     True)
        self.declare_parameter('show_window',   False)

        mp_path          = self.get_parameter('mp_model_path').value
        pose_mp_path     = self.get_parameter('pose_model_path').value
        cam_idx          = self.get_parameter('camera_index').value
        self._fps        = self.get_parameter('camera_fps').value
        self._cam_w      = self.get_parameter('camera_width').value
        self._cam_h      = self.get_parameter('camera_height').value
        self._display_fps   = self.get_parameter('display_fps').value
        self._mp_fps        = self.get_parameter('mediapipe_fps').value
        self._depth_fps     = self.get_parameter('depth_fps').value
        self._depth_running = True
        self._depth_max_mm  = self.get_parameter('depth_max_mm').value
        win_size         = self.get_parameter('window_size').value
        _sw              = self.get_parameter('show_window').value
        self._show       = _sw if isinstance(_sw, bool) else str(_sw).lower() in ('true','1','yes')
        _ud              = self.get_parameter('use_depth').value
        _use_depth       = _ud if isinstance(_ud, bool) else str(_ud).lower() in ('true','1','yes')

        self.get_logger().info(
            f'FPS config — camera:{self._fps} display:{self._display_fps} '
            f'mediapipe:{self._mp_fps} depth:{self._depth_fps}'
        )

        # ── publishers ───────────────────────────────────────────────────────
        self._pub_lm      = self.create_publisher(Float32MultiArray,  '/arm/landmarks',      10)
        self._pub_pose_lm = self.create_publisher(Float32MultiArray,  '/arm/pose_landmarks', 10)
        self._pub_img     = self.create_publisher(Image,              '/arm/color_image',    10)
        self._pub_depth   = self.create_publisher(Image,              '/arm/depth_image',    10)
        self._pub_active_model = self.create_publisher(String,  '/active_ai_model', 10)
        self._pub_pred_inf_ms  = self.create_publisher(Float32, '/ai_inference_ms', 10)
        self._pub_e2e_ms       = self.create_publisher(Float32, '/e2e_latency_ms', 10)
        # Gripper opening in METRES, computed once here from the RAW landmarks.
        # Previously every consumer called gripper_from_hand() itself, on
        # whatever array it happened to hold -- and three different coordinate
        # systems reached it: raw normalised, depth-fused metric, and the
        # predictor's normalised feature space. Curling the fingers moves them
        # mostly in Z, which is tiny in normalised units and real in metres, so
        # a fist could measure as MORE spread than an open hand and the mapping
        # inverted. One number, one representation, no ambiguity.
        self._pub_grip = self.create_publisher(Float32, '/arm/gripper_opening', 10)
        # Same one-number-one-representation reasoning as gripper_opening
        # above, and computed from the exact same raw (pre-depth-fusion)
        # landmarks -- see hand_fingers_spread's own docstring for why this
        # needs to be a distinct signal from gripper_opening, and why it
        # must NOT be read off /arm/landmarks (that one goes through
        # fuse_depth, which is unreliable on this camera -- see
        # motion_mapping's own notes on why depth fusion was removed from
        # the pose/arm-control path entirely).
        self._pub_spread = self.create_publisher(Float32, '/arm/hand_spread', 10)
        self.create_subscription(String, '/ai_model', self._on_model_switch, 10)

        # robot_node.py's freeze_joints debug state, shown on the JOINT
        # ANGLES panel below so it's visible on-screen instead of only in
        # robot_node's own terminal log -- repeated confusion this session
        # over whether a freeze flag had actually reached that process.
        # transient_local: whichever of the two nodes starts first, the
        # other still gets the current value rather than missing it.
        self._frozen_joints = set()
        self.create_subscription(
            String, '/freeze_status', self._on_freeze_status,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # ── pose (arm L1-L4) model toggle -- lives in robot_node.py, this node
        # just sends switch commands and displays whatever robot_node reports
        # back as active, so both processes stay in sync ──────────────────────
        self._pub_pose_ai_model = self.create_publisher(String, '/pose_ai_model', 10)
        self._active_pose_model_name = 'none'
        self.create_subscription(String, '/active_pose_ai_model', self._on_pose_active_update, 10)

        # ── ML pipeline ──────────────────────────────────────────────────────
        self._extractor = FeatureExtractor()
        self._prep      = Preprocessor(window_size=win_size)

        # ── Temporal predictors (model toggle) ───────────────────────────────
        self._predictor_cache   = {}    # name → PredictorBase instance
        self._active_predictor  = None
        self._active_model_name = 'none'
        self._pred_inf_ms       = 0.0
        self._try_load_predictors(win_size)

        # ── MediaPipe -- Hands (predictor pipeline) ───────────────────────────
        if not os.path.exists(mp_path):
            raise FileNotFoundError(
                f'MediaPipe model not found: {mp_path}\n'
                'Download from: https://storage.googleapis.com/mediapipe-models/'
                'hand_landmarker/hand_landmarker/float16/latest/'
                'hand_landmarker.task'
            )
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=mp_path,
                delegate=BaseOptions.Delegate.GPU,
            ),
            running_mode=RunningMode.VIDEO,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_hands=1,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)

        # ── MediaPipe -- Pose (arm control: L1-L4, direct geometry) ───────────
        if not os.path.exists(pose_mp_path):
            raise FileNotFoundError(
                f'MediaPipe Pose model not found: {pose_mp_path}\n'
                'Download from: https://storage.googleapis.com/mediapipe-models/'
                'pose_landmarker/pose_landmarker_full/float16/latest/'
                'pose_landmarker_full.task'
            )
        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=pose_mp_path,
                delegate=BaseOptions.Delegate.GPU,
            ),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=1,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_opts)
        self._ts_ms      = 0

        # ── camera ───────────────────────────────────────────────────────────
        if cam_idx < 0:
            cam_idx = self._auto_detect_camera()
        self.get_logger().info(f'Opening camera index {cam_idx}')
        self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open camera {cam_idx}')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cam_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_h)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # ── Orbbec depth pipeline (optional) ─────────────────────────────────
        self._depth_mm:    np.ndarray | None = None
        self._depth_vis:   np.ndarray | None = None
        self._ob_lib       = None
        self._ob_pipeline  = None
        self._colormap_idx = 0
        # _depth_max_mm already set from parameter above

        if _use_depth:
            lib = _load_orbbec_sdk()
            if lib is not None:
                try:
                    err = (ctypes.c_void_p * 1)(None)
                    pipeline = lib.ob_create_pipeline(err)
                    config   = lib.ob_create_config(err)
                    lib.ob_config_enable_video_stream(
                        config, OB_STREAM_DEPTH, 640, 480, 30, OB_FORMAT_DEPTH, err)
                    lib.ob_pipeline_start_with_config(pipeline, config, err)
                    lib.ob_delete_config(config, err)
                    self._ob_lib      = lib
                    self._ob_pipeline = pipeline
                    # background thread to pull depth frames
                    self._depth_thread = threading.Thread(
                        target=self._depth_loop, daemon=True)
                    self._depth_thread.start()
                    self.get_logger().info('Orbbec depth pipeline started.')
                except Exception as e:
                    self.get_logger().warn(f'Orbbec depth init failed: {e}')
            else:
                self.get_logger().warn('Orbbec SDK not found — depth disabled.')

        # frame shared with main-thread display
        self._display_frame: np.ndarray | None = None
        self._last_lm_arr   = None
        self._gripper_opening = None   # metres, from raw landmarks
        self._last_pose_lm_arr = None
        self._tick_count    = 0
        self._coords_cache  = None
        self._coords_tick   = 0

        # Latest raw frame from camera (written by reader, read by processor)
        self._latest_raw: np.ndarray | None = None
        self._latest_raw_ts = 0.0   # time.monotonic() at capture, for end-to-end latency

        # Latest pose overlay frame + landmarks (written by MediaPipe thread)
        # Main thread composites these onto the live raw frame
        self._latest_pose_overlay: np.ndarray | None = None
        self._latest_landmarks    = None   # mediapipe landmarks object

        # Thread 1: reads camera as fast as possible, always keeps only latest frame
        self._reader_thread = threading.Thread(
            target=self._camera_reader, daemon=True)
        self._reader_thread.start()

        # Thread 2: runs MediaPipe on latest frame, never blocks on camera I/O
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.get_logger().info('InferenceNode ready.')

    # ── predictor loading & switching ─────────────────────────────────────────

    def _try_load_predictors(self, window_size: int = 30):
        """Load all three predictor models at startup (uses random weights if file missing)."""
        try:
            from models.lstm_predictor        import LSTMPredictor
            from models.gru_predictor         import GRUPredictor
            from models.transformer_predictor import TransformerPredictor
        except ImportError as e:
            self.get_logger().warn(f'Predictor import failed: {e}')
            return

        classes = {
            'lstm':        LSTMPredictor,
            'gru':         GRUPredictor,
            'transformer': TransformerPredictor,
        }
        for name, Cls in classes.items():
            path = os.path.join(_SCRIPTS, 'models', _PREDICTOR_FILES[name])
            pred = Cls(window_size=window_size)
            if os.path.exists(path):
                try:
                    pred.load(path)
                    self.get_logger().info(f'Predictor loaded: {name.upper()} ← {path}')
                except Exception as e:
                    pred.build()
                    self.get_logger().warn(f'Predictor {name} load error ({e}) — random weights')
            else:
                pred.build()
                self.get_logger().warn(
                    f'Predictor {name}: weights not found at {path} — architecture ready')
            self._predictor_cache[name] = pred

    def _on_freeze_status(self, msg):
        self._frozen_joints = {j for j in msg.data.split(',') if j}

    def _on_model_switch(self, msg):
        """Handle /ai_model topic — switches the active temporal predictor."""
        name = msg.data.lower().strip()
        if name in ('none', 'off', ''):
            self._active_predictor  = None
            self._active_model_name = 'none'
            self.get_logger().info('AI predictor: OFF')
        elif name in self._predictor_cache:
            self._active_predictor  = self._predictor_cache[name]
            self._active_model_name = name
            self.get_logger().info(f'AI predictor: {name.upper()}')
        else:
            self.get_logger().warn(f'Unknown model "{name}" — valid: lstm, gru, transformer, none')
            return
        status = String(); status.data = self._active_model_name
        self._pub_active_model.publish(status)

    def _on_pose_active_update(self, msg):
        """Handle /active_pose_ai_model -- robot_node.py reporting which pose
        predictor is actually active, so this window's overlay stays in sync
        even if switched by something other than this window's own keys."""
        self._active_pose_model_name = msg.data

    def _switch_pose_model(self, name: str):
        """Send a switch request to robot_node.py -- it owns the actual
        pose predictor cache, this just asks it to change the active one."""
        msg = String(); msg.data = name
        self._pub_pose_ai_model.publish(msg)

    def _switch_both_models(self, name: str):
        """
        Switch the gripper (hand) predictor AND the arm (pose) predictor
        together.

        They live in different places -- the hand predictors are cached here,
        the pose predictors in robot_node.py -- so switching used to need two
        separate actions and one keypress changed only the arm, leaving the
        overlay showing e.g. "Gripper AI: LSTM / Arm AI: GRU". Comparing two
        architectures at once is not what the keys are for.
        """
        key = name.lower().strip()
        if key in ('none', 'off', ''):
            self._active_predictor  = None
            self._active_model_name = 'none'
        elif key in self._predictor_cache:
            self._active_predictor  = self._predictor_cache[key]
            self._active_model_name = key
        else:
            self.get_logger().warn(f'Unknown model "{name}"')
            return
        status = String(); status.data = self._active_model_name
        self._pub_active_model.publish(status)
        self._switch_pose_model(key)
        self.get_logger().info(f'AI predictor (gripper + arm): {key.upper()}')

    # ── depth background loop ─────────────────────────────────────────────────

    def _depth_loop(self):
        lib, pipeline = self._ob_lib, self._ob_pipeline
        interval = 1.0 / self._depth_fps
        last_t   = 0.0
        # _depth_running lets main() stop this thread *before* tearing down
        # rclpy; relying on rclpy.ok() alone left it mid-iteration during
        # shutdown, publishing onto a dead context and core-dumping on exit.
        while self._depth_running and rclpy.ok():
            now = time.monotonic()
            if now - last_t < interval:
                time.sleep(0.005)
                continue
            last_t = now
            err      = (ctypes.c_void_p * 1)(None)
            frameset = lib.ob_pipeline_wait_for_frameset(pipeline, 50, err)
            if err[0]:
                lib.ob_delete_error(err[0]); err[0] = None
                continue
            if not frameset:
                continue
            depth_frame = lib.ob_frameset_depth_frame(frameset, err)
            if depth_frame:
                dm = _orbbec_depth_mm(lib, depth_frame)
                if dm is not None:
                    self._depth_mm  = dm
                    self._depth_vis = _colorize_depth(
                        dm, COLORMAPS[self._colormap_idx], self._depth_max_mm)
                    self._publish_depth(dm)
                lib.ob_delete_frame(depth_frame, err)
            lib.ob_delete_frame(frameset, err)

    # ── Thread 1: camera reader — runs as fast as hardware allows ─────────────

    def _camera_reader(self):
        """
        Reads frames at camera_fps.
        Updates display immediately at display_fps using last known pose overlay.
        """
        display_interval = 1.0 / self._display_fps
        last_display     = 0.0

        while rclpy.ok():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            self._latest_raw    = frame   # for MediaPipe thread
            self._latest_raw_ts = time.monotonic()   # capture timestamp, for e2e latency

            if self._show:
                now = time.monotonic()
                if now - last_display >= display_interval:
                    last_display = now
                    pose = self._latest_pose_overlay
                    display_pose = pose if pose is not None else frame.copy()
                    self._display_frame = self._make_4panel(frame.copy(), display_pose)

    # ── Thread 2: MediaPipe processor — always works on latest frame ──────────

    def _capture_loop(self):
        """Picks up the latest camera frame and runs MediaPipe at mediapipe_fps."""
        from motion_mapping import fuse_depth
        mp_interval = 1.0 / self._mp_fps
        ts_ms       = 0
        last_mp     = 0.0

        while rclpy.ok():
            now = time.monotonic()
            if now - last_mp < mp_interval:
                time.sleep(0.002)
                continue

            frame  = self._latest_raw
            cap_ts = self._latest_raw_ts   # capture timestamp, for e2e latency
            if frame is None:
                time.sleep(0.002)
                continue
            self._latest_raw = None
            last_mp = now

            self._tick_count += 1
            ts_ms += int(mp_interval * 1000)

            # periodic heartbeat every 5 s
            if self._tick_count % 100 == 0:
                self.get_logger().info(
                    f'[InferenceNode] frame {self._tick_count} | '
                    f'pose_buf={len(self._prep._buffer)}/{self._prep.window_size} | '
                    f'predictor={self._active_model_name}'
                )

            # BGR → RGB for MediaPipe
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = self._landmarker.detect_for_video(mp_img, ts_ms)
            pose_result = self._pose_landmarker.detect_for_video(mp_img, ts_ms)

            hand_lms = hand_result.hand_landmarks[0] if hand_result.hand_landmarks else None
            pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

            if hand_lms is None and pose_lms is None:
                overlay = frame.copy()
                cv2.putText(overlay, 'No hand or arm detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                brightness = int(frame.mean())
                cv2.putText(overlay, f'brightness: {brightness}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay, 'Show your hand and arm clearly to the camera', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
                self._publish_image(overlay)
                self._latest_pose_overlay = overlay
                continue

            pose_frame = frame.copy()

            # ── HAND branch: predictor pipeline, gripper source ────────────────
            if hand_lms is not None:
                lm_arr = self._lms_to_array(hand_lms)

                # Gripper opening from the RAW normalised landmarks, BEFORE
                # depth fusion. It is a ratio of distances in the image plane
                # and needs no metric depth; fusing first would put it in a
                # different unit system and, worse, the sensor quantises to
                # ~10mm while finger positions differ by centimetres.
                from robot_control.ik_solver import gripper_from_hand as _grip, hand_fingers_spread as _spread
                _g = _grip(lm_arr)
                if _g is not None:
                    self._gripper_opening = float(_g)
                    gm = Float32(); gm.data = self._gripper_opening
                    self._pub_grip.publish(gm)

                _s = _spread(lm_arr)
                if _s is not None:
                    sm = Float32(); sm.data = float(_s)
                    self._pub_spread.publish(sm)

                if self._depth_mm is not None:
                    lm_arr = fuse_depth(lm_arr, self._depth_mm)

                # always update feature buffer (needed by the temporal predictor)
                features = self._extractor.extract(lm_arr)
                ready, seq = self._prep.update(features, lm_arr)

                pub_lm = lm_arr   # default: publish raw landmarks
                if self._active_predictor is not None and ready:
                    t_pred  = time.perf_counter()
                    predicted = self._active_predictor.predict(seq)       # (63,)
                    self._pred_inf_ms = (time.perf_counter() - t_pred) * 1000.0

                    pred_lm = lm_arr.copy()
                    pred_lm[:, :3] = predicted.reshape(21, 3)
                    pub_lm  = pred_lm

                    lat = Float32(); lat.data = float(self._pred_inf_ms)
                    self._pub_pred_inf_ms.publish(lat)
                    mdl = String();  mdl.data = self._active_model_name
                    self._pub_active_model.publish(mdl)

                    # end-to-end latency: camera capture -> prediction ready to publish
                    e2e_ms = (time.monotonic() - cap_ts) * 1000.0
                    e2e = Float32(); e2e.data = float(e2e_ms)
                    self._pub_e2e_ms.publish(e2e)

                self._pub_lm.publish(self._make_lm_msg(pub_lm, 21))
                self._last_lm_arr = lm_arr
                self._draw_overlay(pose_frame, hand_lms)

            # ── POSE branch: direct arm control (L1-L4) ─────────────────────────
            if pose_lms is not None:
                # Deliberately NOT depth-fused, unlike the hand/gripper
                # landmarks below. Two rounds of patches (all-or-nothing
                # triplet fusion, then a hold-last-good cache with a 1s
                # staleness timeout) both papered over the same root
                # cause without fixing it: this camera's depth/color
                # alignment fails at the SDK level ("set align hardware
                # mode int failed!", startup log), so shoulder/elbow/
                # wrist depth reads are intermittently wrong even when
                # nominally valid, not just occasionally missing. Every
                # patch that fell back to (or cached against) raw
                # MediaPipe coordinates still had to jump BACK to fused
                # metric coordinates the next time fusion briefly
                # succeeded -- and that jump got dramatically more
                # visible once L2 (shoulder pitch, the joint with the
                # largest reach/visual impact of any of these) was
                # unfrozen. Reported directly: "untick l2 it become
                # crazy" -- confirmed in the live capture as the same
                # coordinate-scale jump pattern already chased twice
                # this session, just far more visible on this joint.
                # L1-L6 are all pure angle/ratio math (atan2, acos) that
                # does not need true metric distance to be directionally
                # correct -- only the MIN_MAG noise-floor checks
                # (L1_MIN_MAG etc.) were calibrated assuming fused-scale
                # jitter, and are a secondary safety net, not the
                # primary signal. Raw MediaPipe coordinates are always
                # internally self-consistent (no external sensor, so no
                # per-landmark fusion success/failure to flap between at
                # all) -- removing the dependency here eliminates the
                # whole class of bug instead of bounding it further.
                pose_arr = self._pose_lms_to_array(pose_lms)

                self._pub_pose_lm.publish(self._make_lm_msg(pose_arr, 33))
                self._last_pose_lm_arr = pose_arr
                self._draw_arm_overlay(pose_frame, pose_lms)

            if self._active_predictor is not None:
                cv2.putText(pose_frame,
                            f'Gripper AI: {self._active_model_name.upper()}  {self._pred_inf_ms:.1f}ms',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2, cv2.LINE_AA)
            else:
                cv2.putText(pose_frame, 'Gripper: Direct mode (no model)', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

            if self._active_pose_model_name != 'none':
                cv2.putText(pose_frame,
                            f'Arm AI: {self._active_pose_model_name.upper()}',
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2, cv2.LINE_AA)
            else:
                cv2.putText(pose_frame, 'Arm: Direct mode (no model)', (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(pose_frame, '[1]LSTM [2]GRU [3]Transformer [0]OFF  -- switches gripper + arm',
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            self._publish_image(pose_frame)
            self._latest_pose_overlay = pose_frame

    def _make_coords_panel(self, W: int, H: int) -> np.ndarray:
        from motion_mapping import ARM_CONTROL_MIN_VISIBILITY, ARM_CONTROL_MIN_VISIBILITY_WRIST
        """Bottom-right panel — light theme, one-line-per-item, no overlapping elements."""
        # ── palette (BGR) ─────────────────────────────────────────────────────
        BG       = (250, 250, 250)
        C_DARK   = ( 20,  20,  20)
        C_HEAD   = ( 20,  20, 130)
        C_OK     = (  0, 100,   0)
        C_MISS   = ( 20,  20, 160)
        C_GREY   = (155, 155, 155)
        C_DIV    = (205, 205, 205)
        C_BAR_BG = (215, 215, 215)
        C_BAR_FG = ( 15, 150,  15)
        C_BAR_0  = ( 80,  80,  80)
        TITLE_BG = (205, 205, 205)
        C_OK_BG  = (210, 240, 210)
        C_MISS_BG= (215, 215, 245)

        panel = np.full((H, W, 3), BG, dtype=np.uint8)

        P  = 12   # horizontal padding
        # LS/JROW (below) were 26/24 when this panel showed 3 landmark rows
        # and 6 joints. It now shows 9 landmark rows (P:SHOULDER/ELBOW/WRIST
        # + H:WRIST/PALM/INDEX/MIDDLE/RING/PINKY) and 8 joints (L1-L6, L7_R,
        # L7_L), which need 527px of content in a panel whose canvas height
        # is the camera frame's (480px) -- cv2 draws outside those bounds
        # without error, so the overflow was invisible: joint_L7_R and
        # joint_L7_L, the last two rows, were silently clipped off-screen.
        # Tightened spacing below keeps everything within 480px with margin.
        LS = 22   # line stride (px between text baselines)

        def T(s, x, yy, sc=0.42, col=C_DARK, bold=False):
            cv2.putText(panel, str(s), (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        sc, col, 2 if bold else 1, cv2.LINE_AA)

        # ── Title bar (height 32) ─────────────────────────────────────────────
        cv2.rectangle(panel, (0, 0), (W, 32), TITLE_BG, -1)
        cv2.line(panel, (0, 32), (W, 32), C_DIV, 1)
        T('Robot Coordinates Monitor', P, 22, sc=0.50, bold=True)

        lm_arr = getattr(self, '_last_lm_arr', None)
        if lm_arr is None:
            T('Waiting for pose...', P, 62, sc=0.48, col=C_GREY)
            return panel

        # ── helper: section header with solid line underneath ─────────────────
        def section(title, y):
            """Draw section title. Returns y of first content line."""
            T(title, P, y, sc=0.41, col=C_HEAD, bold=True)
            line_y = y + 8          # 8 px gap below text baseline
            cv2.line(panel, (P, line_y), (W - P, line_y), C_DIV, 1)
            return line_y + LS - 6  # first content baseline

        # ══════════════════════════════════════════════════════════════════════
        # LANDMARKS  — 1 row per landmark
        # ══════════════════════════════════════════════════════════════════════
        y = section('LANDMARKS', 46)

        # fixed column starts (px from left)
        #   NAME      X-lbl  X-val    Y-lbl  Y-val    Z-lbl  Z-val    VIS-badge
        NX = P
        XX = P + 84
        YX = XX + 80
        ZX = YX + 80
        VX = ZX + 80

        # The landmarks that actually DRIVE the robot, from both trackers.
        # POSE (33 landmarks) supplies the arm: shoulder/elbow/wrist geometry
        # gives L1-L4 directly, no IK. HAND (21) supplies the gripper: mean
        # fingertip-to-wrist distance over the four fingers, divided by palm
        # length. The thumb is excluded -- it folds across the palm rather than
        # curling toward the wrist, so its distance barely changes between a
        # fist and a flat hand.
        #
        # This list used to show WRIST/INDEX_TIP/THUMB_TIP, left over from when
        # the gripper measured a thumb-to-index pinch. It has not measured that
        # for a while, so the panel was showing landmarks nothing reads.
        pose_arr_ov = getattr(self, '_last_pose_lm_arr', None)
        rows = []
        if pose_arr_ov is not None:
            # P:WRIST uses its own, lower threshold -- see
            # ARM_CONTROL_MIN_VISIBILITY_WRIST's declaration in
            # motion_mapping.py. Showing this panel's red-X at the same
            # 0.78 bar robot_node no longer actually gates wrist on would
            # make the monitor lie about why the arm is or isn't moving.
            for nm, ix, thr in (('P:SHOULDER', 12, ARM_CONTROL_MIN_VISIBILITY),
                                ('P:ELBOW', 14, ARM_CONTROL_MIN_VISIBILITY),
                                ('P:WRIST', 16, ARM_CONTROL_MIN_VISIBILITY_WRIST)):
                rows.append((nm, pose_arr_ov[ix], thr))
        for nm, ix in (('H:WRIST', 0), ('H:PALM', 9), ('H:INDEX', 8),
                       ('H:MIDDLE', 12), ('H:RING', 16), ('H:PINKY', 20)):
            rows.append((nm, lm_arr[ix], 0.3))

        for name, lm, thresh in rows:
            vis = float(lm[3])
            ok  = vis >= thresh
            fc  = C_OK if ok else C_MISS

            # row background badge for vis status (right side only)
            badge_bg = C_OK_BG if ok else C_MISS_BG
            cv2.rectangle(panel, (VX - 4, y - 14), (W - P + 2, y + 6),
                          badge_bg, -1)

            # name
            T(name, NX, y, sc=0.41, col=fc, bold=True)

            # x value
            T('x', XX, y, sc=0.36, col=C_GREY)
            T(f'{lm[0]:+.3f}', XX + 12, y, sc=0.40, col=C_DARK)

            # y value
            T('y', YX, y, sc=0.36, col=C_GREY)
            T(f'{lm[1]:+.3f}', YX + 12, y, sc=0.40, col=C_DARK)

            # z value
            T('z', ZX, y, sc=0.36, col=C_GREY)
            T(f'{lm[2]:+.3f}', ZX + 12, y, sc=0.40, col=C_DARK)

            # vis
            status = 'OK' if ok else 'LOW'
            T(f'{vis:.2f}  {status}', VX, y, sc=0.40, col=fc, bold=True)

            y += LS

        # ══════════════════════════════════════════════════════════════════════
        # JOINT ANGLES  — 1 row per joint (label + value + bar)
        # ══════════════════════════════════════════════════════════════════════
        y += 10
        y = section('JOINT ANGLES  (rad)', y)

        # Loud banner when ANY joint is frozen (robot_node's debug
        # freeze_joints). Without this, the number shown per-row below is
        # what the CAMERA computes live -- it keeps changing even for a
        # frozen joint, because freezing happens downstream in robot_node,
        # not here. Seeing a live number move while the robot visibly
        # doesn't was exactly the "why is it not moving" confusion this
        # session kept hitting. Made explicit instead of silent.
        if self._frozen_joints:
            cv2.rectangle(panel, (P, y - 14), (W - P, y + 6), (200, 200, 255), -1)
            T(f'FREEZE ACTIVE -- robot HELD at 0 for: '
              f'{", ".join(sorted(j.replace("joint_","") for j in self._frozen_joints))}',
              P, y, sc=0.38, col=(140, 0, 0), bold=True)
            y += LS

        from motion_mapping import JOINT_LIMITS, JOINT_NAMES, landmarks_to_joint_angles
        pose_arr = getattr(self, '_last_pose_lm_arr', None)
        angles = landmarks_to_joint_angles(pose_arr) if pose_arr is not None else {}
        if angles:
            # gripper (L7) here is the same real hand-based mapping
            # robot_node.py uses — overrides the rough pose-fingertip guess
            _g = getattr(self, '_gripper_opening', None)
            if _g is not None:
                angles['joint_L7_R'] = _g
                angles['joint_L7_L'] = _g

        # column positions
        LX = P          # "L1"
        VLX = P + 42    # "+0.000" -- wide enough for "L7_R"/"L7_L" (4 chars),
                        # which used to run into the value ("L7_L+0.008")
                        # at the 26px offset sized for "L1".."L6" (2 chars)
        BX  = P + 80    # bar start
        BW  = W - BX - P
        BH  = 9         # bar height

        JROW = 20       # joint row stride

        for ji, joint in enumerate(JOINT_NAMES):
            val    = angles.get(joint)
            lo, hi = JOINT_LIMITS[joint]
            # "L1".."L6", "L7_R", "L7_L" -- not joint[-2:], which turned the
            # gripper joints into "_R"/"_L" once they were added.
            short  = joint.replace('joint_', '')

            jcol = C_DARK if val is not None else C_GREY

            # label
            T(short, LX, y, sc=0.41, col=jcol)

            # value — fixed width column
            T(f'{val:+.3f}' if val is not None else '---', VLX, y, sc=0.40, col=jcol)

            # bar drawn 4 px below text baseline so it never touches text
            by1 = y + 4
            by2 = by1 + BH
            cv2.rectangle(panel, (BX, by1), (BX + BW, by2), C_BAR_BG, -1)

            if val is not None:
                t    = max(0.0, min(1.0, (val - lo) / (hi - lo)))
                fill = int(t * BW)
                cv2.rectangle(panel, (BX, by1), (BX + fill, by2), C_BAR_FG, -1)
                zero_x = BX + int(max(0.0, min(1.0, (-lo) / (hi - lo))) * BW)
                cv2.line(panel, (zero_x, by1), (zero_x, by2), C_BAR_0, 1)

            y += JROW

        return panel

    def _make_4panel(self, rgb: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """2×2 grid: [RGB | Pose] / [Depth | Coords]."""
        H, W = rgb.shape[:2]
        blank = np.zeros((H, W, 3), dtype=np.uint8)

        depth = self._depth_vis
        depth = cv2.resize(depth, (W, H)) if depth is not None else blank.copy()
        if self._depth_vis is None:
            cv2.putText(depth, 'Depth: no signal', (10, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1)

        # Rebuild coords panel every 6 frames (~5 Hz) — expensive text drawing
        self._coords_tick += 1
        if self._coords_cache is None or self._coords_tick % 6 == 0:
            self._coords_cache = self._make_coords_panel(W, H)
        coords = self._coords_cache

        # labels on camera panels only (coords panel has its own title bar)
        for img, label in [(rgb,   'RGB Camera'),
                           (pose,  'Pose Overlay'),
                           (depth, 'Depth (RGBD)')]:
            # dark outline + white text for readability on any background
            cv2.putText(img, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        top    = np.hstack([rgb,   pose])
        bottom = np.hstack([depth, coords])
        return np.vstack([top, bottom])

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _lms_to_array(lms) -> np.ndarray:
        # MediaPipe Hands doesn't report visibility/presence (always None) —
        # treat every landmark as fully visible via _vis().
        arr = np.array([[lm.x, lm.y, lm.z,
                         _vis(lm),
                         1.0]
                        for lm in lms], dtype=float)   # (21, 5)
        return arr

    @staticmethod
    def _pose_lms_to_array(lms) -> np.ndarray:
        # MediaPipe Pose DOES report real visibility/presence per landmark.
        arr = np.array([[lm.x, lm.y, lm.z,
                         lm.visibility,
                         getattr(lm, 'presence', 1.0)]
                        for lm in lms], dtype=float)   # (33, 5)
        return arr

    @staticmethod
    def _make_lm_msg(lm_arr: np.ndarray, n: int) -> Float32MultiArray:
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='landmarks', size=n, stride=n * 5),
                MultiArrayDimension(label='fields',    size=5, stride=5),
            ],
            data_offset=0,
        )
        msg.data = lm_arr.flatten().tolist()
        return msg

    @staticmethod
    def _draw_overlay(frame, lms):
        h, w = frame.shape[:2]

        for a, b in _HAND_CONN:
            pa, pb = lms[a], lms[b]
            cv2.line(frame,
                     (int(pa.x * w), int(pa.y * h)),
                     (int(pb.x * w), int(pb.y * h)),
                     (0, 255, 0), 2)
        for i in _HAND_IDX:
            p = lms[i]
            cx, cy = int(p.x * w), int(p.y * h)
            color = (0, 140, 255) if i == 0 else (0, 255, 0)   # wrist highlighted
            cv2.circle(frame, (cx, cy), 5, color,           -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    @staticmethod
    def _draw_arm_overlay(frame, lms):
        h, w = frame.shape[:2]

        for a, b in _ARM_CONN:
            pa, pb = lms[a], lms[b]
            if pa.visibility < 0.3 or pb.visibility < 0.3:
                continue
            cv2.line(frame,
                     (int(pa.x * w), int(pa.y * h)),
                     (int(pb.x * w), int(pb.y * h)),
                     (255, 140, 0), 3)
        for i in _ARM_IDX:
            p = lms[i]
            if p.visibility < 0.3:
                continue
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 7, (255, 140, 0),   -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 2)

    def _publish_image(self, bgr: np.ndarray):
        # Same shutdown race as _publish_depth: this runs on the camera thread,
        # so an exception here kills that thread and the preview window with
        # it, rather than just logging an error.
        if not rclpy.ok():
            return
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_frame'
        msg.height   = bgr.shape[0]
        msg.width    = bgr.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = bgr.shape[1] * 3
        msg.data     = bgr.tobytes()
        try:
            self._pub_img.publish(msg)
        except Exception:
            pass   # shutdown raced us between the check and the publish

    def _publish_depth(self, depth_mm: np.ndarray):
        """Publish real depth (mm, float32 per pixel) so a recorder can save
        it synchronized with the color video -- a plain .mp4 has no channel
        for this, which is why offline-video-reprocessed datasets never had
        real depth (see dataset_ipynb/old/)."""
        # The depth thread outlives rclpy's shutdown by a few frames, and
        # publishing onto a torn-down context raises RCLError from inside the
        # thread, which killed the process with a core dump on every exit.
        if not rclpy.ok():
            return
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_depth_frame'
        msg.height   = depth_mm.shape[0]
        msg.width    = depth_mm.shape[1]
        msg.encoding = '32FC1'
        msg.step     = depth_mm.shape[1] * 4
        msg.data     = depth_mm.astype(np.float32).tobytes()
        try:
            self._pub_depth.publish(msg)
        except Exception:
            pass   # shutdown raced us between the check and the publish

    @staticmethod
    def _auto_detect_camera() -> int:
        """
        Pick the best colour camera:
          1. Prefer USB / Orbbec / Astra cameras over built-in webcams.
          2. Skip IR nodes.
          3. Verify frame brightness (10–245).
        Logs every candidate so the user can override with camera_index param.
        """
        import glob as _glob
        candidates = []   # list of (priority, idx, name)

        for node in sorted(_glob.glob('/sys/class/video4linux/video*/name')):
            try:
                name = open(node).read().strip()
                idx  = int(node.split('video')[2].split('/')[0])
                name_u = name.upper()
                if 'IR' in name_u:
                    print(f'[camera] skip  video{idx}: {name} (IR)')
                    continue
                # higher priority = preferred
                if any(k in name_u for k in ('ORBBEC', 'ASTRA', 'USB', 'OBSENSOR')):
                    prio = 0   # USB/Orbbec first
                else:
                    prio = 1   # built-in webcam fallback
                candidates.append((prio, idx, name))
                print(f'[camera] found video{idx}: {name}  prio={prio}')
            except Exception:
                pass

        # sort: lower prio number first; within same prio, lower idx first
        candidates.sort(key=lambda t: (t[0], t[1]))

        for prio, idx, name in candidates:
            try:
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    cap.release()
                    continue
                frame = None
                for _ in range(5):
                    ret, f = cap.read()
                    if ret:
                        frame = f
                cap.release()
                if frame is None:
                    continue
                brightness = float(np.mean(frame))
                print(f'[camera] test  video{idx}: brightness={brightness:.1f}')
                if 10 < brightness < 245:
                    print(f'[camera] selected video{idx}: {name}')
                    return idx
            except Exception:
                pass

        print('[camera] no suitable camera found, defaulting to index 0')
        return 0

    def destroy_node(self):
        self._cap.release()
        self._landmarker.close()
        self._pose_landmarker.close()
        if self._ob_lib is not None and self._ob_pipeline is not None:
            err = (ctypes.c_void_p * 1)(None)
            self._ob_lib.ob_pipeline_stop(self._ob_pipeline, err)
            time.sleep(1.0)
            self._ob_lib.ob_delete_pipeline(self._ob_pipeline, err)
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
#  Settings panel — custom-drawn OpenCV window with Apply / Cancel buttons
# ══════════════════════════════════════════════════════════════════════════════

SETTINGS_WIN = 'Gesture Control  —  Settings'

# Parameter definitions: (label, attr, min, max, step)
_PARAMS = [
    ('Camera FPS',      '_fps',         1,  60,   1),
    ('Camera Width',    '_cam_w',      160,1280,  80),
    ('Camera Height',   '_cam_h',      120, 720,  40),
    ('Display FPS',     '_display_fps', 1,  60,   1),
    ('MediaPipe FPS',   '_mp_fps',      1,  30,   1),
    ('Depth FPS',       '_depth_fps',   1,  30,   1),
    ('Depth Max mm',    '_depth_max_mm',500,8000,500),
    ('Depth Colormap',  '_colormap_idx',0,   2,   1),
]

class SettingsPanel:
    W, H   = 500, 520
    PAD    = 24
    ROW_H  = 48
    SL_H   = 8     # slider track height

    # colours (BGR)
    C_BG       = (30,  30,  30)
    C_TITLE    = (20,  20,  20)
    C_ROW_A    = (38,  38,  38)
    C_ROW_B    = (44,  44,  44)
    C_LABEL    = (200,200,200)
    C_APPLIED  = ( 60,200, 60)   # green  — applied value
    C_PENDING  = ( 30,180,255)   # amber  — changed but not applied
    C_TRACK    = ( 70, 70, 70)
    C_THUMB    = (100,180,255)
    C_APPLY_BG = ( 30,140, 30)
    C_CANCEL_BG= ( 30, 30,140)
    C_BTN_TXT  = (255,255,255)
    FONT       = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, node):
        self._node    = node
        # pending = copy of current settings (editable before Apply)
        self._pending = {p[1]: getattr(node, p[1]) for p in _PARAMS}
        # applied = snapshot of last applied settings (for Cancel)
        self._applied = dict(self._pending)

        self._drag_param = None   # which param is being dragged
        self._status_msg = ''
        self._status_col = self.C_APPLIED

        cv2.namedWindow(SETTINGS_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(SETTINGS_WIN, self._on_mouse)

    # ── layout helpers ────────────────────────────────────────────────────────

    def _row_y(self, i):
        """Top y of row i (0-indexed), below title bar."""
        return 50 + i * self.ROW_H

    def _slider_rect(self, i):
        """(x1, y, x2) of the slider track for row i."""
        y   = self._row_y(i) + self.ROW_H // 2 + 4
        x1  = self.PAD + 170
        x2  = self.W - self.PAD
        return x1, y, x2

    def _val_to_x(self, i, val):
        lo, hi = _PARAMS[i][2], _PARAMS[i][3]
        x1, _, x2 = self._slider_rect(i)
        t = (val - lo) / max(1, hi - lo)
        return int(x1 + t * (x2 - x1))

    def _x_to_val(self, i, x):
        lo, hi, step = _PARAMS[i][2], _PARAMS[i][3], _PARAMS[i][4]
        x1, _, x2 = self._slider_rect(i)
        t   = max(0.0, min(1.0, (x - x1) / max(1, x2 - x1)))
        raw = lo + t * (hi - lo)
        snapped = round(raw / step) * step
        return int(max(lo, min(hi, snapped)))

    def _btn_rects(self):
        """Returns (apply_rect, cancel_rect) as (x1,y1,x2,y2)."""
        by1 = self.H - 52
        by2 = self.H - 14
        mid = self.W // 2
        return ((self.PAD, by1, mid - 8, by2),
                (mid + 8, by1, self.W - self.PAD, by2))

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            # check buttons
            ar, cr = self._btn_rects()
            if ar[0] <= x <= ar[2] and ar[1] <= y <= ar[3]:
                self._do_apply()
                return
            if cr[0] <= x <= cr[2] and cr[1] <= y <= cr[3]:
                self._do_cancel()
                return
            # check sliders
            for i, (_, attr, lo, hi, step) in enumerate(_PARAMS):
                x1, sy, x2 = self._slider_rect(i)
                row_top = self._row_y(i)
                if row_top <= y <= row_top + self.ROW_H and x1 - 10 <= x <= x2 + 10:
                    self._drag_param = i
                    self._pending[attr] = self._x_to_val(i, x)

        elif event == cv2.EVENT_MOUSEMOVE and self._drag_param is not None:
            i    = self._drag_param
            attr = _PARAMS[i][1]
            self._pending[attr] = self._x_to_val(i, x)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_param = None

    # ── apply / cancel ────────────────────────────────────────────────────────

    def _do_apply(self):
        node = self._node
        for _, attr, *_ in _PARAMS:
            val = self._pending[attr]
            setattr(node, attr, val)
        # push camera settings to hardware
        node._cap.set(cv2.CAP_PROP_FPS,          node._fps)
        node._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  node._cam_w)
        node._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, node._cam_h)
        self._applied = dict(self._pending)
        self._status_msg = 'Settings applied!'
        self._status_col = self.C_APPLIED

    def _do_cancel(self):
        self._pending  = dict(self._applied)
        self._status_msg = 'Cancelled — reverted to last applied.'
        self._status_col = (80, 80, 220)

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), self.C_BG, dtype=np.uint8)

        # title bar
        cv2.rectangle(img, (0, 0), (self.W, 46), self.C_TITLE, -1)
        cv2.putText(img, 'Gesture Control  —  Settings',
                    (self.PAD, 30), self.FONT, 0.55, (200,200,200), 1, cv2.LINE_AA)

        for i, (label, attr, lo, hi, step) in enumerate(_PARAMS):
            row_y = self._row_y(i)
            bg = self.C_ROW_A if i % 2 == 0 else self.C_ROW_B
            cv2.rectangle(img, (0, row_y), (self.W, row_y + self.ROW_H), bg, -1)

            pending_val = self._pending[attr]
            applied_val = self._applied[attr]
            changed     = pending_val != applied_val
            val_col     = self.C_PENDING if changed else self.C_APPLIED

            # label
            cv2.putText(img, label, (self.PAD, row_y + 20),
                        self.FONT, 0.42, self.C_LABEL, 1, cv2.LINE_AA)

            # current applied value (small, grey)
            if changed:
                cv2.putText(img, f'was {applied_val}',
                            (self.PAD, row_y + 36),
                            self.FONT, 0.35, (130,130,130), 1, cv2.LINE_AA)

            # pending value (large, coloured)
            cv2.putText(img, str(pending_val),
                        (self.PAD + 155, row_y + 22),
                        self.FONT, 0.52, val_col, 1 if not changed else 2, cv2.LINE_AA)

            # slider track
            x1, sy, x2 = self._slider_rect(i)
            cv2.line(img, (x1, sy), (x2, sy), self.C_TRACK, self.SL_H)

            # filled portion
            tx = self._val_to_x(i, pending_val)
            cv2.line(img, (x1, sy), (tx, sy), val_col, self.SL_H)

            # thumb
            cv2.circle(img, (tx, sy), 9, self.C_THUMB, -1)
            cv2.circle(img, (tx, sy), 9, (200,200,200), 1)

        # status message
        if self._status_msg:
            cv2.putText(img, self._status_msg,
                        (self.PAD, self.H - 58),
                        self.FONT, 0.42, self._status_col, 1, cv2.LINE_AA)

        # Apply / Cancel buttons
        ar, cr = self._btn_rects()
        cv2.rectangle(img, ar[:2], ar[2:], self.C_APPLY_BG, -1)
        cv2.rectangle(img, ar[:2], ar[2:], (100,200,100), 1)
        cv2.putText(img, 'APPLY', (ar[0]+28, ar[3]-10),
                    self.FONT, 0.6, self.C_BTN_TXT, 2, cv2.LINE_AA)

        cv2.rectangle(img, cr[:2], cr[2:], self.C_CANCEL_BG, -1)
        cv2.rectangle(img, cr[:2], cr[2:], (100,100,200), 1)
        cv2.putText(img, 'CANCEL', (cr[0]+18, cr[3]-10),
                    self.FONT, 0.6, self.C_BTN_TXT, 2, cv2.LINE_AA)

        return img

    def close(self):
        try:
            cv2.destroyWindow(SETTINGS_WIN)
        except Exception:
            pass


# ── Settings button on main monitor ──────────────────────────────────────────

_BTN = [0, 0, 0, 0]   # x1 y1 x2 y2

def _draw_settings_btn(frame: np.ndarray, open: bool):
    global _BTN
    h, w   = frame.shape[:2]
    label  = ' X Settings ' if open else '  Settings  '
    fs, th = 0.48, 1
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = 6
    x2, y2 = w - 6, 26
    x1, y1 = x2 - tw - pad * 2, 4
    _BTN   = [x1, y1, x2, y2]
    bg = (0, 110, 200) if open else (50, 50, 50)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (150,150,150), 1)
    cv2.putText(frame, label, (x1 + pad, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)


def _btn_hit(x, y):
    return _BTN[0] <= x <= _BTN[2] and _BTN[1] <= y <= _BTN[3]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = InferenceNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    WIN     = 'Gesture Control Monitor'
    panel   = None   # SettingsPanel instance when open

    if node._show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        # Smaller, out-of-the-way startup size/position -- the previous
        # quarter-screen default (half width x half height, top-right
        # quadrant) was too big for just glancing at while working.
        # 448x381 at (1460, 727) matches exactly where the operator had it
        # sized/placed by hand when asking for this: "make it smaller a bit
        # like now size please see it and the location at the startup...
        # just for starting" -- freely resizable/movable afterward, this is
        # only the STARTUP default.
        cv2.resizeWindow(WIN, 448, 381)
        cv2.moveWindow(WIN, 1460, 727)

        def on_mouse(event, x, y, flags, param):
            nonlocal panel
            if event == cv2.EVENT_LBUTTONDOWN and _btn_hit(x, y):
                if panel is None:
                    panel = SettingsPanel(node)
                else:
                    panel.close()
                    panel = None

        cv2.setMouseCallback(WIN, on_mouse)

    try:
        while rclpy.ok():
            # main monitor
            frame = node._display_frame
            if frame is not None and node._show:
                display = frame.copy()
                _draw_settings_btn(display, panel is not None)
                cv2.imshow(WIN, display)

            # settings panel
            if panel is not None:
                cv2.imshow(SETTINGS_WIN, panel.draw())

            key = cv2.waitKey(max(1, int(1000 / node._display_fps))) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                if panel is None:
                    panel = SettingsPanel(node)
                else:
                    panel.close()
                    panel = None
            elif key == ord('1'):
                node._switch_both_models('lstm')
            elif key == ord('2'):
                node._switch_both_models('gru')
            elif key == ord('3'):
                node._switch_both_models('transformer')
            elif key == ord('0'):
                node._switch_both_models('none')

    except KeyboardInterrupt:
        pass
    finally:
        if panel:
            panel.close()
        # Stop the depth thread before tearing down rclpy, so it cannot try to
        # publish onto a dead context on the way out.
        node._depth_running = False
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if node._show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
