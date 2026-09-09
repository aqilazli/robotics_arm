#!/usr/bin/env python3

import csv
import os
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Float32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINTS = [
    ('joint_L1', -1.7,  1.7),
    ('joint_L2', -0.98, 1.0),
    ('joint_L3', -2.0,  1.3),
    ('joint_L4', -2.0,  2.0),
    ('joint_L5', -2.1,  2.1),
    ('joint_L6', -3.1,  3.1),
    # L7 is the prismatic gripper jaw: metres of travel, not radians.
    # 0.0 = closed, 0.011 = fully open.
    ('joint_L7_R', 0.0,  0.011),
    ('joint_L7_L', 0.0,  0.011),
]

# Resting pose the sliders start at, and what the MANUAL panel's "Home"
# preset sends. Back to all-zeros (straight arm) for L1-L6, gripper still
# open -- was a custom folded pose (L1=0.2565, L2=0.2923, L3=-0.35,
# L5=0.2188) chasing a specific reference hand-sign, reconsidered directly:
# "why initail postion of gazebo robort startup like this? is houdl be
# straight". Kept numerically in sync with robot_control/robot_node.py's
# OWN HOME_POSE by hand -- there is no shared import between the two
# scripts, and they have drifted out of sync before.
HOME_POSE = {
    'joint_L1': 0.0,
    'joint_L2': 0.0,
    'joint_L3': 0.0,
    'joint_L4': 0.0,
    'joint_L5': 0.0,
    'joint_L6': 0.0,
    'joint_L7_R': 0.011,
    'joint_L7_L': 0.011,
}

RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'recordings'
)
os.makedirs(RECORDINGS_DIR, exist_ok=True)


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class ArmNode(Node):
    def __init__(self):
        super().__init__('arm_gui_node')
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/robot_arm_controller/joint_trajectory',
            10,
        )
        self.current_positions  = {j[0]: 0.0 for j in JOINTS}
        self.current_velocities = {j[0]: 0.0 for j in JOINTS}
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # Raw human-tracked angle per joint, straight from the camera via
        # robot_node.py's /arm/human_angles -- see _build_human_table for
        # the window that shows this next to current_positions (the actual
        # robot). Requested directly, after too many rounds of guessing from
        # descriptions: "make me other table gui for human arm position and
        # robot arm position. so taht i can undersstnd the value and can
        # adjust accrodingly myself ratrehr than you guess like shit".
        self.human_positions = {j[0]: 0.0 for j in JOINTS}
        self.create_subscription(JointState, '/arm/human_angles', self._human_angles_cb, 10)

        # Calibration tool: min/max/mean/count per joint, accumulated only
        # while calibration_capturing is True (see _open_calibration_tool).
        # Built so the operator can measure a pose's own raw-angle
        # stability directly -- e.g. "how much does this reading actually
        # swing while I hold still?" -- instead of asking for a one-off
        # diagnostic script every time. Requested directly: "can you add
        # gui for calibartion later? easier for me to troublshooting them".
        self.calibration_capturing = False
        self.calibration_stats = {
            j[0]: {'min': None, 'max': None, 'sum': 0.0, 'count': 0}
            for j in JOINTS
        }

        # ── AI model control ──────────────────────────────────────────────────
        # Control mode arbitration. Both this GUI and robot_node.py command
        # the same robot_arm_controller; if both send at once they fight and
        # the arm jerks between two sources. /control_mode names the single
        # owner: 'gesture' (robot_node, vision) or 'manual' (this GUI).
        # Latched with transient_local so a node starting later still learns
        # the current mode instead of assuming a default.
        _mode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._mode_pub = self.create_publisher(String, '/control_mode', _mode_qos)
        # Safety default: teleop starts OFF (manual) until the switch in the
        # GUI is pushed to CAMERA TELEOPERATION. See _build_mode_panel's
        # _set_mode('manual') call, which is the actual latched publish that
        # decides startup state -- this attribute just needs to agree with it.
        self.control_mode = 'manual'

        self._ai_model_pub = self.create_publisher(String, '/ai_model', 10)
        self._active_model  = 'none'
        self._ai_latency_ms = 0.0
        self.create_subscription(String,  '/active_ai_model', self._on_active_model, 10)
        self.create_subscription(Float32, '/ai_inference_ms', self._on_ai_latency,   10)

        # Live re-trigger for L6's own zero point -- see robot_node.py's
        # _on_recalibrate_l6 for why this exists (that calibration used
        # to only ever run once, automatically, at startup).
        self._l6_recal_pub = self.create_publisher(String, '/recalibrate_l6', 10)

        # ── per-joint freeze (live, during teleoperation) ───────────────────────
        # robot_node.py's freeze was launch-time only (--freeze-l1-l5 etc,
        # relaunch to change it) -- asked for directly: "make a swtich at gui
        # so that i can off the joint that i want during teleopreation! i want
        # to analysze the joont myself!". /freeze_joints_cmd is this GUI
        # commanding which joints to hold; /freeze_status is robot_node
        # confirming what it actually applied, so the panel reflects the
        # other process's real state rather than just this button's own
        # click -- same confirm-don't-assume pattern as /active_pose_ai_model.
        # /freeze_joints_cmd is deliberately VOLATILE, not TRANSIENT_LOCAL.
        # Tested TRANSIENT_LOCAL directly against an orphaned old GUI
        # process still holding a stale retained value (gnome-terminal-
        # server has left orphans behind more than once this session): a
        # late-joining robot_node replays retained history from EVERY
        # writer it has ever seen, one sample each, in an order that isn't
        # guaranteed -- so even a fresh publish from this process, or a
        # repeated one, is not guaranteed to be the one a new subscriber
        # ends up keeping. VOLATILE removes that failure mode structurally:
        # there is nothing to replay, so an orphan's old cached value is
        # simply irrelevant. What a late-joining robot_node gets instead is
        # whatever the currently-running GUI(s) publish live -- covered by
        # the startup publish below plus _reassert_freeze's 2s heartbeat.
        _freeze_cmd_qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE)
        self._freeze_cmd_pub = self.create_publisher(String, '/freeze_joints_cmd', _freeze_cmd_qos)
        self.frozen_confirmed = set()
        # /freeze_status stays TRANSIENT_LOCAL: it has exactly one writer
        # (robot_node), which already republishes it every 2s on its own,
        # so a late-joining GUI still gets the current value on startup
        # instead of showing nothing until the next tick.
        _freeze_status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/freeze_status', self._on_freeze_status, _freeze_status_qos)

        # run_arm.sh's --freeze=... flag now passes freeze_joints to THIS
        # node too (see its own comment), not just robot_node.py. Needed
        # because this GUI is the one that keeps reasserting the freeze
        # state every 2s (see _reassert_freeze) -- without reading the same
        # flag here, that heartbeat kept forcing robot_node's command-line
        # freeze back to empty within 2 seconds of startup, every time.
        # Reported directly: "--freeze=joint_L4,joint_L6" to isolate L5 for
        # tuning still left L4/L6 live, confirmed via /freeze_status == ''.
        self.declare_parameter('freeze_joints', '')
        self.initial_frozen = {j.strip() for j in
                               self.get_parameter('freeze_joints').value.split(',')
                               if j.strip()}

        # Publish this GUI's own state once on startup (the command-line
        # freeze if one was given, otherwise unfrozen); _reassert_freeze
        # (below) repeats this every 2s so a late-joining robot_node is
        # never left waiting more than that for the truth.
        self.set_frozen(self.initial_frozen)

        # ── gripper center position (live, via TF) ──────────────────────────────
        # L7_R and L7_L's own link ORIGINS are already coincident (checked
        # directly: they differ by ~2 micrometres) -- the actual jaw
        # separation is baked into each jaw's VISUAL mesh offset instead
        # (+/-0.0055 in Y, symmetric), so the average of the two link
        # origins already IS the point centred between the jaws, not an
        # approximation of it. Uses TF (from robot_state_publisher, which
        # already computes this correctly from the same URDF) rather than
        # re-deriving forward kinematics by hand from the joint chain.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.gripper_xyz = None   # (x, y, z) in base_link frame, or None
                                   # until the first successful TF lookup

    def _on_freeze_status(self, msg):
        self.frozen_confirmed = {j for j in msg.data.split(',') if j}

    def update_gripper_xyz(self):
        """Look up the current gripper-center position via TF. Call this
        periodically (e.g. from the GUI's poll loop) -- TF lookups can
        throw while frames are still settling, so failures are silent and
        just leave the last known value in place."""
        try:
            r = self._tf_buffer.lookup_transform('base_link', 'L7_R', rclpy.time.Time())
            l = self._tf_buffer.lookup_transform('base_link', 'L7_L', rclpy.time.Time())
            self.gripper_xyz = (
                (r.transform.translation.x + l.transform.translation.x) / 2.0,
                (r.transform.translation.y + l.transform.translation.y) / 2.0,
                (r.transform.translation.z + l.transform.translation.z) / 2.0,
            )
        except Exception:
            pass

    def set_frozen(self, frozen_set):
        m = String(); m.data = ','.join(sorted(frozen_set))
        self._freeze_cmd_pub.publish(m)

    def _joint_state_cb(self, msg):
        for name, pos, vel in zip(msg.name, msg.position,
                                  msg.velocity if msg.velocity else [0.0] * len(msg.name)):
            if name in self.current_positions:
                self.current_positions[name]  = pos
                self.current_velocities[name] = vel

    def _human_angles_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.human_positions:
                self.human_positions[name] = pos
        if self.calibration_capturing:
            for name, pos in zip(msg.name, msg.position):
                st = self.calibration_stats.get(name)
                if st is None:
                    continue
                st['min'] = pos if st['min'] is None else min(st['min'], pos)
                st['max'] = pos if st['max'] is None else max(st['max'], pos)
                st['sum'] += pos
                st['count'] += 1

    def _on_active_model(self, msg):
        self._active_model = msg.data

    def _on_ai_latency(self, msg):
        self._ai_latency_ms = msg.data

    def send(self, positions, duration_sec):
        # joint_trajectory_controller drops the ENTIRE trajectory if the
        # position count does not match the joint-name count, and says so only
        # in the Gazebo log -- from the GUI it just looks like the button did
        # nothing. Fail loudly here instead.
        if self.control_mode != 'manual':
            self.get_logger().warn(
                'Ignoring manual command: control mode is '
                f'"{self.control_mode}". Switch to MANUAL in the GUI first.')
            return False
        if len(positions) != len(JOINTS):
            self.get_logger().error(
                f'Refusing to send {len(positions)} positions for '
                f'{len(JOINTS)} joints; the controller would drop it silently.')
            return False
        msg = JointTrajectory()
        msg.joint_names = [j[0] for j in JOINTS]

        # TWO points: current position, then target. A single end point makes
        # joint_trajectory_controller silently ignore commands where the arm
        # joints are already at their targets and only the gripper differs --
        # so "Send Command" appeared to do nothing whenever only the gripper
        # sliders had been touched.
        start = JointTrajectoryPoint()
        start.positions = [float(self.current_positions.get(j[0], 0.0)) for j in JOINTS]
        start.time_from_start = Duration(sec=0, nanosec=10_000_000)   # 10ms

        end = JointTrajectoryPoint()
        end.positions = [float(p) for p in positions]
        end.time_from_start = Duration(sec=int(duration_sec))

        msg.points = [start, end]
        self.publisher.publish(msg)
        return True

    def set_control_mode(self, mode: str):
        self.control_mode = mode
        m = String(); m.data = mode
        self._mode_pub.publish(m)

    def set_ai_model(self, model_name: str):
        msg = String(); msg.data = model_name
        self._ai_model_pub.publish(msg)

    def recalibrate_l6(self):
        msg = String(); msg.data = 'recalibrate'
        self._l6_recal_pub.publish(msg)


# ── GUI ───────────────────────────────────────────────────────────────────────

class ArmGUI:
    # ── palette ───────────────────────────────────────────────────────────────
    BG     = '#1e1e2e'
    PANEL  = '#313244'
    FG     = '#cdd6f4'
    ACCENT = '#89b4fa'
    GREEN  = '#a6e3a1'
    RED    = '#f38ba8'
    YELLOW = '#f9e2af'
    GREY   = '#6c7086'
    TEAL   = '#94e2d5'

    # model button colors
    _MODEL_COLORS = {
        'lstm':        ('#cba6f7', '#1e1e2e'),   # purple
        'gru':         ('#a6e3a1', '#1e1e2e'),   # green
        'transformer': ('#89b4fa', '#1e1e2e'),   # blue
        'none':        ('#6c7086', '#cdd6f4'),   # grey (OFF)
    }

    def __init__(self, root, node: ArmNode):
        self.root = root
        self.node = node
        root.title('Robot Arm Controller')
        root.resizable(False, False)
        root.configure(bg=self.BG)

        self._apply_styles()

        self.slider_vars   = []
        self.target_vars   = []
        self.current_vars  = []
        self.velocity_vars = []
        self.duration_var  = tk.DoubleVar(value=2.0)

        self._recording      = False
        self._record_data    = []
        self._record_start   = 0.0
        self._replay_thread  = None
        self._replay_running = False
        self._video_proc     = None

        self._active_model_local = 'none'   # last toggled by user

        self._build_ui()
        self._poll_joint_states()
        self._poll_ai_status()
        self._poll_freeze_status()
        self._poll_gripper_xyz()
        self._reassert_freeze()
        self._reassert_mode()

    # ── styles ────────────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.',               background=self.BG,    foreground=self.FG,  font=('Segoe UI', 10))
        s.configure('TFrame',          background=self.BG)
        s.configure('TLabel',          background=self.BG,    foreground=self.FG)
        s.configure('TScale',          background=self.BG,    troughcolor=self.PANEL, sliderlength=18)
        s.configure('TButton',         background=self.PANEL, foreground=self.FG,  padding=6)
        s.configure('Accent.TButton',  background=self.ACCENT,foreground=self.BG,  padding=6)
        s.configure('Green.TButton',   background=self.GREEN, foreground=self.BG,  padding=6)
        s.configure('Red.TButton',     background=self.RED,   foreground=self.BG,  padding=6)
        s.configure('Yellow.TButton',  background=self.YELLOW,foreground=self.BG,  padding=6)
        s.configure('TLabelframe',     background=self.BG,    foreground=self.ACCENT)
        s.configure('TLabelframe.Label', background=self.BG,  foreground=self.ACCENT, font=('Segoe UI', 10, 'bold'))
        # Treeview (Human vs Robot table, Calibration Tool) had no style of
        # its own -- the '.' catch-all above set foreground=self.FG (a pale
        # colour meant for dark backgrounds) but left Treeview's own
        # background at the 'clam' theme's default WHITE, so rows rendered
        # as pale/white text on a near-white background -- unreadable.
        # Reported directly with a screenshot: "change the white colour,
        # icant see it proerly".
        s.configure('Treeview', background=self.PANEL, fieldbackground=self.PANEL,
                    foreground=self.FG, rowheight=24, font=('Courier', 9))
        s.configure('Treeview.Heading', background=self.BG, foreground=self.ACCENT,
                    font=('Segoe UI', 9, 'bold'))
        s.map('Treeview', background=[('selected', self.ACCENT)],
                          foreground=[('selected', self.BG)])
        s.map('TButton',        background=[('active', '#45475a')])
        s.map('Accent.TButton', background=[('active', '#74c7ec')])
        s.map('Green.TButton',  background=[('active', '#94e2a1')])
        s.map('Red.TButton',    background=[('active', '#eb8ba8')])
        s.map('Yellow.TButton', background=[('active', '#f0d29f')])

    # ── UI builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=5)

        tk.Label(self.root, text='Robot Arm Controller',
                 bg=self.BG, fg=self.ACCENT,
                 font=('Segoe UI', 14, 'bold')).pack(pady=(12, 4))

        self._build_joint_panel(pad)
        self._build_duration_panel(pad)
        self._build_preset_panel(pad)
        self._build_mode_panel(pad)
        self._build_freeze_panel(pad)        # ← NEW
        self._build_gripper_xyz_panel(pad)   # ← NEW
        self._build_ai_model_panel(pad)      # ← NEW
        self._build_action_buttons()
        self._build_record_panel(pad)
        self._build_capture_panel(pad)
        self._build_statusbar()

    def _build_joint_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='Joint Control & Monitor  (L1-L6 radians, L7 metres)')
        frame.pack(fill='x', **pad)

        headers = ['Joint', 'Min', 'Slider', 'Max', 'Target', 'Current', 'Velocity']
        widths  = [8, 5, 30, 5, 8, 8, 8]
        for c, (h, w) in enumerate(zip(headers, widths)):
            tk.Label(frame, text=h, bg=self.BG, fg=self.ACCENT,
                     font=('Segoe UI', 9, 'bold'), width=w,
                     anchor='center').grid(row=0, column=c, padx=4, pady=2)

        for row, (name, lo, hi) in enumerate(JOINTS, start=1):
            var        = tk.DoubleVar(value=HOME_POSE.get(name, 0.0))
            target_var = tk.StringVar(value=f'{HOME_POSE.get(name, 0.0):.3f}')
            cur_var    = tk.StringVar(value='0.000')
            vel_var    = tk.StringVar(value='0.000')
            self.slider_vars.append(var)
            self.target_vars.append(target_var)
            self.current_vars.append(cur_var)
            self.velocity_vars.append(vel_var)
            self._make_formatter(var, target_var)

            tk.Label(frame, text=name, bg=self.BG, fg=self.FG,
                     font=('Segoe UI', 9), anchor='w').grid(row=row, column=0, padx=6, sticky='w')
            tk.Label(frame, text=f'{lo:.2f}', bg=self.BG, fg=self.GREY,
                     font=('Segoe UI', 8)).grid(row=row, column=1, padx=2)
            ttk.Scale(frame, from_=lo, to=hi, orient='horizontal',
                      variable=var, length=280).grid(row=row, column=2, padx=4, pady=3)
            tk.Label(frame, text=f'{hi:.2f}', bg=self.BG, fg=self.GREY,
                     font=('Segoe UI', 8)).grid(row=row, column=3, padx=2)
            tk.Label(frame, textvariable=target_var, bg=self.BG, fg=self.ACCENT,
                     font=('Courier', 9), width=7,
                     anchor='center').grid(row=row, column=4, padx=4)
            tk.Label(frame, textvariable=cur_var, bg=self.BG, fg=self.GREEN,
                     font=('Courier', 9), width=7,
                     anchor='center').grid(row=row, column=5, padx=4)
            tk.Label(frame, textvariable=vel_var, bg=self.BG, fg=self.YELLOW,
                     font=('Courier', 9), width=7,
                     anchor='center').grid(row=row, column=6, padx=4)

    def _build_duration_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='Motion Duration (seconds)')
        frame.pack(fill='x', **pad)
        ttk.Scale(frame, from_=0.5, to=10.0, orient='horizontal',
                  variable=self.duration_var, length=340).pack(side='left', padx=8, pady=4)
        tk.Label(frame, textvariable=self.duration_var, bg=self.BG, fg=self.ACCENT,
                 font=('Courier', 10), width=5).pack(side='left')

    def _build_preset_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='Preset Poses')
        frame.pack(fill='x', **pad)
        # One value per entry in JOINTS. The 7th is joint_L7, the gripper, in
        # METRES of jaw travel (0.0 closed, 0.011 open) -- every other value
        # here is an angle in radians. These were six-long before L7 existed,
        # which left the gripper slider untouched by every preset.
        presets = [
            ('Home', [HOME_POSE[n] for n, _, _ in JOINTS]),
            ('Pose 1', [0.8, 0.4, -0.5, 0.3, 0.4, 0.0, 0.011, 0.011]),
            ('Pose 2', [-0.8, 0.4, -0.5, -0.3, 0.4, 0.0, 0.0, 0.0]),
            ('Pose 3', [0.0, 0.5, -1.0, 0.0, 0.5, 0.5, 0.011, 0.011]),
            ('Stretch', [0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        assert all(len(p) == len(JOINTS) for _, p in presets), \
            'preset length must match JOINTS'    
        for col, (label, pos) in enumerate(presets):
            ttk.Button(frame, text=label,
                       command=lambda p=pos: self._load_preset(p)
                       ).grid(row=0, column=col, padx=6, pady=6)

    # ── AI model toggle panel ─────────────────────────────────────────────────

    def _build_mode_panel(self, pad):
        """Who owns the arm: the camera, or this window. Never both."""
        frame = ttk.LabelFrame(self.root, text='Control Mode  (only one source drives the arm at a time)')
        frame.pack(fill='x', **pad)

        btn_row = ttk.Frame(frame)
        btn_row.pack(side='left', padx=10, pady=8)

        self._mode_btns = {}
        for label, key, active_bg in (
            ('CAMERA TELEOPERATION', 'gesture', self.TEAL),
            ('MANUAL MODE',          'manual',  self.YELLOW),
        ):
            btn = tk.Button(
                btn_row, text=label, bg=self.PANEL, fg=self.FG,
                activebackground=active_bg, activeforeground=self.BG,
                font=('Segoe UI', 10, 'bold'), relief='flat',
                padx=14, pady=6, cursor='hand2',
                command=lambda k=key: self._set_mode(k),
            )
            btn.pack(side='left', padx=4)
            self._mode_btns[key] = (btn, active_bg)

        self._mode_var = tk.StringVar()
        tk.Label(frame, textvariable=self._mode_var, bg=self.BG,
                 font=('Segoe UI', 9)).pack(side='left', padx=16)

        # L6's own "zero" only ever calibrated once, automatically, at
        # startup, with no way to redo it if the wrist wasn't actually
        # relaxed during that window -- see robot_node.py's
        # _on_recalibrate_l6. Reported directly: "l6? why not center?"
        ttk.Button(frame, text='Recalibrate L6',
                   command=self._recalibrate_l6).pack(side='left', padx=10)

        # A separate window, not another row squeezed into this one: a
        # live side-by-side table of every joint's raw human-tracked angle
        # against the robot's own actual position, so numbers can be read
        # and compared directly instead of asked about. See ArmNode's
        # human_positions/current_positions and _build_human_table.
        ttk.Button(frame, text='Human vs Robot Table',
                   command=self._open_human_table).pack(side='left', padx=10)

        # A self-service version of the ad-hoc diagnostic scripts used
        # repeatedly this session to answer "how stable is this reading
        # while I hold a pose?" (min/max/mean over a capture window).
        # Requested directly: "can you add gui for calibartion later?
        # easier for me to troublshooting them".
        ttk.Button(frame, text='Calibration Tool',
                   command=self._open_calibration_tool).pack(side='left', padx=10)

        self._set_mode('manual')

    def _recalibrate_l6(self):
        self.node.recalibrate_l6()
        self.status_var.set(
            'L6 recalibrating -- hold wrist still in its neutral position now.')

    def _open_human_table(self):
        if getattr(self, '_human_table_win', None) is not None \
                and self._human_table_win.winfo_exists():
            self._human_table_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title('Human vs Robot Joint Values')
        win.configure(bg=self.BG)
        self._human_table_win = win

        tk.Label(win, text='Human (camera, raw) vs Robot (actual, published)',
                 bg=self.BG, fg=self.ACCENT,
                 font=('Segoe UI', 12, 'bold')).pack(pady=(10, 6), padx=10)

        cols = ('joint', 'human', 'robot', 'delta')
        tree = ttk.Treeview(win, columns=cols, show='headings', height=len(JOINTS))
        headings = {'joint': 'Joint', 'human': 'Human', 'robot': 'Robot', 'delta': 'Delta'}
        widths   = {'joint': 90, 'human': 110, 'robot': 110, 'delta': 110}
        for c in cols:
            tree.heading(c, text=headings[c])
            tree.column(c, width=widths[c], anchor='center')
        tree.pack(padx=10, pady=(0, 10))

        row_ids = {}
        for name, _lo, _hi in JOINTS:
            row_ids[name] = tree.insert('', 'end', values=(name.replace('joint_', ''), '', '', ''))
        self._human_table_tree = tree
        self._human_table_rows = row_ids

        tk.Label(win, text='L1-L6 in radians, L7 in metres. "Human" is the raw camera\n'
                            'reading before any engage-rescale/smoothing is applied.',
                 bg=self.BG, fg=self.FG, font=('Segoe UI', 8),
                 justify='left').pack(padx=10, pady=(0, 10), anchor='w')

        self._refresh_human_table()

    def _refresh_human_table(self):
        win = getattr(self, '_human_table_win', None)
        if win is None or not win.winfo_exists():
            return
        for name, _lo, _hi in JOINTS:
            human = self.node.human_positions.get(name, 0.0)
            robot = self.node.current_positions.get(name, 0.0)
            self._human_table_tree.item(
                self._human_table_rows[name],
                values=(name.replace('joint_', ''),
                        f'{human:+.3f}', f'{robot:+.3f}', f'{human - robot:+.3f}'))
        self.root.after(100, self._refresh_human_table)

    def _open_calibration_tool(self):
        if getattr(self, '_calibration_win', None) is not None \
                and self._calibration_win.winfo_exists():
            self._calibration_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title('Calibration Tool')
        win.configure(bg=self.BG)
        self._calibration_win = win

        tk.Label(win, text='Hold a pose steady, capture, and read off how much\n'
                            'the raw reading actually swings -- same numbers this\n'
                            'session\'s diagnostic scripts have been printing.',
                 bg=self.BG, fg=self.ACCENT, font=('Segoe UI', 11, 'bold'),
                 justify='left').pack(pady=(10, 6), padx=10, anchor='w')

        btn_row = ttk.Frame(win)
        btn_row.pack(fill='x', padx=10, pady=(0, 8))
        self._calib_capture_btn_var = tk.StringVar(value='Start Capture')
        ttk.Button(btn_row, textvariable=self._calib_capture_btn_var,
                   command=self._toggle_calibration_capture).pack(side='left', padx=(0, 6))
        ttk.Button(btn_row, text='Reset',
                   command=self._reset_calibration).pack(side='left')
        self._calib_status_var = tk.StringVar(value='Idle -- not capturing.')
        tk.Label(btn_row, textvariable=self._calib_status_var, bg=self.BG,
                 fg=self.GREY, font=('Segoe UI', 9)).pack(side='left', padx=16)

        cols = ('joint', 'current', 'min', 'max', 'range', 'mean', 'n')
        tree = ttk.Treeview(win, columns=cols, show='headings', height=len(JOINTS))
        headings = {'joint': 'Joint', 'current': 'Current', 'min': 'Min', 'max': 'Max',
                    'range': 'Range', 'mean': 'Mean', 'n': 'Samples'}
        widths = {'joint': 70, 'current': 85, 'min': 85, 'max': 85,
                  'range': 85, 'mean': 85, 'n': 70}
        for c in cols:
            tree.heading(c, text=headings[c])
            tree.column(c, width=widths[c], anchor='center')
        tree.pack(padx=10, pady=(0, 10))

        row_ids = {}
        for name, _lo, _hi in JOINTS:
            row_ids[name] = tree.insert('', 'end', values=(name.replace('joint_', ''),
                                                             '', '', '', '', '', ''))
        self._calib_tree = tree
        self._calib_rows = row_ids

        tk.Label(win, text='L1-L6 in radians, L7 in metres. Same raw signal as the\n'
                            'Human vs Robot table (/arm/human_angles) -- min/max/mean\n'
                            'are only accumulated while capturing is running.',
                 bg=self.BG, fg=self.FG, font=('Segoe UI', 8),
                 justify='left').pack(padx=10, pady=(0, 10), anchor='w')

        # Nudge/jog controls: play with a joint's own value directly from
        # this window (2% of that joint's own range per click) instead of
        # switching to the main slider panel -- MANUAL mode only, same
        # send() path/mode-gate the sliders already use. Requested
        # directly: "i want to add value either + or - value so taht i
        # can play the value".
        jog_frame = ttk.LabelFrame(win, text='Jog (MANUAL mode only, 2% of range per click)')
        jog_frame.pack(fill='x', padx=10, pady=(0, 10))
        for row, (name, lo, hi) in enumerate(JOINTS):
            step = (hi - lo) * 0.02
            tk.Label(jog_frame, text=name.replace('joint_', ''), bg=self.BG, fg=self.FG,
                     font=('Segoe UI', 9), width=8, anchor='w').grid(row=row, column=0, padx=(8, 4), pady=2)
            ttk.Button(jog_frame, text='-', width=3,
                       command=lambda n=name, s=step: self._jog_joint(n, -s)
                       ).grid(row=row, column=1, padx=2)
            val_var = tk.StringVar()
            self._calib_jog_vars = getattr(self, '_calib_jog_vars', {})
            self._calib_jog_vars[name] = val_var
            tk.Label(jog_frame, textvariable=val_var, bg=self.BG, fg=self.ACCENT,
                     font=('Courier', 9), width=8, anchor='center').grid(row=row, column=2, padx=4)
            ttk.Button(jog_frame, text='+', width=3,
                       command=lambda n=name, s=step: self._jog_joint(n, s)
                       ).grid(row=row, column=3, padx=(2, 8))

        self._refresh_calibration_tool()

    def _jog_joint(self, name, delta):
        idx = next((i for i, (n, _lo, _hi) in enumerate(JOINTS) if n == name), None)
        if idx is None:
            return
        _n, lo, hi = JOINTS[idx]
        new_val = max(lo, min(hi, self.slider_vars[idx].get() + delta))
        self.slider_vars[idx].set(new_val)
        self._send()

    def _toggle_calibration_capture(self):
        self.node.calibration_capturing = not self.node.calibration_capturing
        if self.node.calibration_capturing:
            self._calib_capture_btn_var.set('Stop Capture')
            self._calib_status_var.set('Capturing... hold the pose steady.')
        else:
            self._calib_capture_btn_var.set('Start Capture')
            self._calib_status_var.set('Stopped -- values held below. Reset to clear.')

    def _reset_calibration(self):
        for st in self.node.calibration_stats.values():
            st['min'] = None
            st['max'] = None
            st['sum'] = 0.0
            st['count'] = 0
        self._calib_status_var.set(
            'Capturing... hold the pose steady.' if self.node.calibration_capturing
            else 'Idle -- not capturing.')

    def _refresh_calibration_tool(self):
        win = getattr(self, '_calibration_win', None)
        if win is None or not win.winfo_exists():
            return
        for idx, (name, _lo, _hi) in enumerate(JOINTS):
            current = self.node.human_positions.get(name, 0.0)
            st = self.node.calibration_stats.get(name, {})
            lo, hi, n = st.get('min'), st.get('max'), st.get('count', 0)
            if n:
                mean = st['sum'] / n
                rng = hi - lo
                vals = (f'{current:+.3f}', f'{lo:+.3f}', f'{hi:+.3f}',
                        f'{rng:.3f}', f'{mean:+.3f}', str(n))
            else:
                vals = (f'{current:+.3f}', '', '', '', '', '0')
            self._calib_tree.item(self._calib_rows[name],
                                   values=(name.replace('joint_', ''), *vals))
            jog_var = getattr(self, '_calib_jog_vars', {}).get(name)
            if jog_var is not None:
                jog_var.set(f'{self.slider_vars[idx].get():+.3f}')
        self.root.after(100, self._refresh_calibration_tool)

    def _warn_wrong_mode(self):
        """
        Say why nothing moved. The mode gate refusing a command used to be a
        log line only, so from the GUI it was indistinguishable from a dead
        button -- which is exactly the silent-failure trap this project has
        hit repeatedly.
        """
        self.status_var.set(
            'IGNORED — currently in CAMERA TELEOPERATION. Click "MANUAL MODE" above first.')
        for key, (btn, active_bg) in self._mode_btns.items():
            if key == 'manual':
                orig = btn.cget('bg')
                btn.configure(bg=self.RED, fg=self.BG)
                self.root.after(1200, lambda b=btn, o=orig: b.configure(bg=o))

    def _set_mode(self, mode: str):
        self.node.set_control_mode(mode)
        for key, (btn, active_bg) in self._mode_btns.items():
            on = key == mode
            btn.configure(bg=active_bg if on else self.PANEL,
                          fg=self.BG if on else self.FG)
        if mode == 'manual':
            self._mode_var.set('Sliders and presets drive the arm.\nThe camera is ignored.')
            msg = 'MANUAL MODE — the GUI owns the arm'
        else:
            self._mode_var.set('Your hand drives the arm.\nSliders and presets are ignored.')
            msg = 'CAMERA TELEOPERATION — your hand owns the arm'
        # The status bar is built after this panel, so on the initial call it
        # does not exist yet.
        if hasattr(self, 'status_var'):
            self.status_var.set(msg)

    def _build_freeze_panel(self, pad):
        """
        Live per-joint hold, independent of Control Mode above -- freezing a
        joint here does not touch camera vs manual ownership, it just stops
        robot_node from updating that ONE joint's target while everything
        else keeps moving. No relaunch needed, unlike run_arm.sh's
        --freeze=... flags this replaces for day-to-day use. Asked for
        directly, after too many rounds of me guessing which joint was at
        fault over chat: "make a swtich at gui so that i can off the joint
        that i want during teleopreation! i want to analysze the joont
        myself!"
        """
        frame = ttk.LabelFrame(
            self.root, text='Joint Freeze  (hold selected joints live, no relaunch)')
        frame.pack(fill='x', **pad)

        row = ttk.Frame(frame)
        row.pack(side='left', padx=10, pady=8)

        # L7_R/L7_L collapsed into one "GRIPPER" switch: they are always
        # driven together (both jaws, same opening), so exposing them as two
        # separate switches would let you freeze one jaw and not the other --
        # a state the gripper never has any legitimate reason to be in.
        self._freeze_vars = {}   # 'joint_L1'..'joint_L6' or 'gripper' -> BooleanVar
        initial = self.node.initial_frozen
        for key, label in (
            ('joint_L1', 'L1'), ('joint_L2', 'L2'), ('joint_L3', 'L3'),
            ('joint_L4', 'L4'), ('joint_L5', 'L5'), ('joint_L6', 'L6'),
            ('gripper',  'L7 GRIPPER'),
        ):
            # Checkboxes start pre-checked for whatever run_arm.sh's
            # --freeze=... flag named, so _reassert_freeze's 2s heartbeat
            # republishes THAT instead of clobbering it back to unfrozen.
            start_on = ('joint_L7_R' in initial or 'joint_L7_L' in initial) \
                if key == 'gripper' else key in initial
            var = tk.BooleanVar(value=start_on)
            self._freeze_vars[key] = var
            tk.Checkbutton(
                row, text=label, variable=var,
                bg=self.PANEL, fg=self.FG, selectcolor=self.RED,
                activebackground=self.PANEL, activeforeground=self.FG,
                font=('Segoe UI', 9, 'bold'), relief='flat',
                cursor='hand2', command=self._on_freeze_toggle,
            ).pack(side='left', padx=6, pady=2)

        tk.Button(
            row, text='FREEZE ALL', bg=self.PANEL, fg=self.FG,
            activebackground=self.RED, activeforeground=self.BG,
            font=('Segoe UI', 9, 'bold'), relief='flat', padx=10,
            cursor='hand2', command=self._freeze_all,
        ).pack(side='left', padx=(16, 4))

        tk.Button(
            row, text='UNFREEZE ALL', bg=self.PANEL, fg=self.FG,
            activebackground=self.GREEN, activeforeground=self.BG,
            font=('Segoe UI', 9, 'bold'), relief='flat', padx=10,
            cursor='hand2', command=self._unfreeze_all,
        ).pack(side='left', padx=4)

        self._freeze_status_var = tk.StringVar(value='All joints live.')
        self._freeze_status_lbl = tk.Label(
            frame, textvariable=self._freeze_status_var, bg=self.BG,
            fg=self.GREY, font=('Segoe UI', 9))
        self._freeze_status_lbl.pack(side='left', padx=16)

    @staticmethod
    def _freeze_keys_to_joints(keys):
        joints = set()
        for k in keys:
            if k == 'gripper':
                joints.add('joint_L7_R')
                joints.add('joint_L7_L')
            else:
                joints.add(k)
        return joints

    def _on_freeze_toggle(self):
        active = {k for k, v in self._freeze_vars.items() if v.get()}
        self.node.set_frozen(self._freeze_keys_to_joints(active))

    def _freeze_all(self):
        for var in self._freeze_vars.values():
            var.set(True)
        self.node.set_frozen(self._freeze_keys_to_joints(self._freeze_vars.keys()))

    def _unfreeze_all(self):
        for var in self._freeze_vars.values():
            var.set(False)
        self.node.set_frozen(set())

    def _reassert_freeze(self):
        """
        Republished every 2s, not just on click/startup -- tested a
        startup-only publish directly (fresh ArmNode against an already-
        alive stale publisher still holding an old TRANSIENT_LOCAL value)
        and it lost the race: a late-joining subscriber can replay either
        publisher's retained sample in either order, so one publish at
        construction time is not guaranteed to win against an orphaned
        old process that never got killed on the last relaunch (the
        gnome-terminal-server flakiness has left orphans behind more than
        once this session). Re-publishing the checkboxes' real state on a
        timer, same reasoning as robot_node's own /freeze_status heartbeat,
        means even a lost first race gets corrected within 2s instead of
        silently sticking for the rest of the session.
        """
        active = {k for k, v in self._freeze_vars.items() if v.get()}
        self.node.set_frozen(self._freeze_keys_to_joints(active))
        self.root.after(2000, self._reassert_freeze)

    def _reassert_mode(self):
        """
        Republish /control_mode every 2s, same reasoning as _reassert_freeze
        right above -- /control_mode is TRANSIENT_LOCAL with no periodic
        heartbeat of its own (unlike /freeze_joints_cmd), so a single stray
        external publish (a diagnostic `ros2 topic pub`, an orphaned old GUI
        process, anything) sticks PERMANENTLY: robot_node obeys it forever,
        while this GUI's own buttons/display never change and keep showing
        the mode THEY think is active. Reported directly: 'wtf im in manual
        mode, still can control via telop camera?' -- caused by exactly that
        kind of stray external publish, with nothing to self-correct it
        afterward. This timer means any such desync corrects within 2s
        instead of silently sticking for the rest of the session.
        """
        self.node.set_control_mode(self.node.control_mode)
        self.root.after(2000, self._reassert_mode)

    def _poll_freeze_status(self):
        """
        Reflects what robot_node CONFIRMS is frozen (/freeze_status), not
        just this panel's own checkbox state -- if robot_node is not
        running, or a message got dropped, that becomes visible here as a
        mismatch instead of the switch silently lying about what the arm
        is actually doing. Same reasoning as every other confirm-not-assume
        readback in this project (/active_pose_ai_model, /active_ai_model).
        """
        confirmed = self.node.frozen_confirmed
        if confirmed:
            names = ', '.join(j.replace('joint_', '') for j in sorted(confirmed))
            self._freeze_status_var.set(f'HELD: {names}')
            self._freeze_status_lbl.configure(fg=self.RED)
        else:
            self._freeze_status_var.set('All joints live.')
            self._freeze_status_lbl.configure(fg=self.GREY)
        self.root.after(300, self._poll_freeze_status)

    def _build_gripper_xyz_panel(self, pad):
        """
        Live X/Y/Z of the gripper's CENTER point (midpoint between the two
        jaws), in the base_link frame -- computed via TF, not re-derived
        forward kinematics, so it's exactly as correct as
        robot_state_publisher's own model of the robot. Requested directly:
        "the coordinates x y z for gripper position ... is at the center
        of the gripper".
        """
        frame = ttk.LabelFrame(
            self.root, text='Gripper Position  (center, metres, base_link frame)')
        frame.pack(fill='x', **pad)

        row = ttk.Frame(frame)
        row.pack(side='left', padx=10, pady=8)

        self._gripper_xyz_var = tk.StringVar(value='waiting for TF…')
        tk.Label(row, textvariable=self._gripper_xyz_var, bg=self.BG,
                 fg=self.ACCENT, font=('Courier', 10, 'bold')
                 ).pack(side='left')

    def _poll_gripper_xyz(self):
        self.node.update_gripper_xyz()
        xyz = self.node.gripper_xyz
        if xyz is not None:
            x, y, z = xyz
            self._gripper_xyz_var.set(f'X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f}')
        self.root.after(150, self._poll_gripper_xyz)

    def _build_ai_model_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='AI Predictor Model  (real-time comparison)')
        frame.pack(fill='x', **pad)

        btn_row = ttk.Frame(frame)
        btn_row.pack(side='left', padx=10, pady=8)

        self._model_btns = {}
        models = [
            ('LSTM',        'lstm',        '#cba6f7'),
            ('GRU',         'gru',         '#a6e3a1'),
            ('Transformer', 'transformer', '#89b4fa'),
            ('OFF',         'none',        '#585b70'),
        ]
        for label, key, active_bg in models:
            btn = tk.Button(
                btn_row,
                text=label,
                bg=self.PANEL,
                fg=self.FG,
                activebackground=active_bg,
                activeforeground=self.BG,
                font=('Segoe UI', 10, 'bold'),
                relief='flat',
                padx=14, pady=6,
                cursor='hand2',
                command=lambda k=key, ab=active_bg: self._set_ai_model(k, ab),
            )
            btn.pack(side='left', padx=4)
            self._model_btns[key] = (btn, active_bg)

        # Status area
        status_col = ttk.Frame(frame)
        status_col.pack(side='left', padx=20, pady=6)

        self._ai_model_var   = tk.StringVar(value='Active: none  (OFF)')
        self._ai_latency_var = tk.StringVar(value='')
        self._ai_status_dot  = tk.StringVar(value='⬤')

        dot_lbl = tk.Label(status_col, textvariable=self._ai_status_dot,
                           bg=self.BG, fg=self.GREY, font=('Segoe UI', 14))
        dot_lbl.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self._ai_dot_label = dot_lbl

        tk.Label(status_col, textvariable=self._ai_model_var,
                 bg=self.BG, fg=self.GREEN,
                 font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, sticky='w')
        tk.Label(status_col, textvariable=self._ai_latency_var,
                 bg=self.BG, fg=self.YELLOW,
                 font=('Courier', 9)).grid(row=1, column=1, sticky='w')

        # highlight OFF by default
        self._highlight_model_btn('none', '#585b70')

    def _set_ai_model(self, model_key: str, active_bg: str):
        self._active_model_local = model_key
        self._highlight_model_btn(model_key, active_bg)
        self.node.set_ai_model(model_key)

        label = 'OFF' if model_key == 'none' else model_key.upper()
        self.status_var.set(f'AI model → {label}')

    def _highlight_model_btn(self, active_key: str, active_bg: str):
        for key, (btn, bg) in self._model_btns.items():
            if key == active_key:
                btn.configure(bg=active_bg, fg=self.BG, relief='sunken')
            else:
                btn.configure(bg=self.PANEL, fg=self.FG, relief='flat')

    def _poll_ai_status(self):
        model = self.node._active_model
        lat   = self.node._ai_latency_ms

        if model == 'none':
            self._ai_model_var.set('Active: none  (OFF)')
            self._ai_dot_label.configure(fg=self.GREY)
        else:
            self._ai_model_var.set(f'Active: {model.upper()}')
            # dot color matches button color
            colors = {'lstm': '#cba6f7', 'gru': '#a6e3a1', 'transformer': '#89b4fa'}
            self._ai_dot_label.configure(fg=colors.get(model, self.TEAL))

        if lat > 0 and model != 'none':
            self._ai_latency_var.set(f'Inference: {lat:.1f} ms')
        else:
            self._ai_latency_var.set('')

        self.root.after(300, self._poll_ai_status)

    # ── action buttons ────────────────────────────────────────────────────────

    def _build_action_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=6)
        ttk.Button(frame, text='Send Command', style='Accent.TButton',
                   command=self._send).grid(row=0, column=0, padx=8)
        ttk.Button(frame, text='Reset to Home', style='Green.TButton',
                   command=self._home).grid(row=0, column=1, padx=8)
        ttk.Button(frame, text='Zero Sliders',
                   command=self._zero).grid(row=0, column=2, padx=8)

    def _build_record_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='Joint Recording & Replay')
        frame.pack(fill='x', **pad)

        self._rec_btn_var    = tk.StringVar(value='Start Recording')
        self._rec_status     = tk.StringVar(value='Idle')
        self._replay_btn_var = tk.StringVar(value='Replay')

        left = ttk.Frame(frame)
        left.pack(side='left', padx=10, pady=6)

        self._rec_btn = ttk.Button(left, textvariable=self._rec_btn_var,
                                   style='Red.TButton', command=self._toggle_record)
        self._rec_btn.grid(row=0, column=0, padx=4)

        ttk.Button(left, text='Save CSV', command=self._save_csv
                   ).grid(row=0, column=1, padx=4)
        ttk.Button(left, text='Load CSV', command=self._load_csv
                   ).grid(row=0, column=2, padx=4)
        ttk.Button(left, textvariable=self._replay_btn_var,
                   style='Yellow.TButton', command=self._toggle_replay
                   ).grid(row=0, column=3, padx=4)

        right = ttk.Frame(frame)
        right.pack(side='left', padx=16, pady=6)
        tk.Label(right, text='Status:', bg=self.BG, fg=self.GREY,
                 font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w')
        tk.Label(right, textvariable=self._rec_status, bg=self.BG, fg=self.YELLOW,
                 font=('Courier', 9), width=30, anchor='w').grid(row=0, column=1, sticky='w')

    def _build_capture_panel(self, pad):
        frame = ttk.LabelFrame(self.root, text='Capture')
        frame.pack(fill='x', **pad)

        self._video_btn_var = tk.StringVar(value='Start Video')
        self._scale_var     = tk.StringVar(value='100%')

        row0 = ttk.Frame(frame)
        row0.pack(fill='x', padx=6, pady=6)

        ttk.Button(row0, text='Screenshot', style='Accent.TButton',
                   command=self._take_screenshot).grid(row=0, column=0, padx=6)
        ttk.Button(row0, textvariable=self._video_btn_var, style='Red.TButton',
                   command=self._toggle_video).grid(row=0, column=1, padx=6)
        ttk.Button(row0, text='Open Folder',
                   command=self._open_recordings_folder).grid(row=0, column=2, padx=6)

        tk.Label(row0, text='Scale:', bg=self.BG, fg=self.GREY,
                 font=('Segoe UI', 9)).grid(row=0, column=3, padx=(16, 4))
        scale_menu = ttk.Combobox(row0, textvariable=self._scale_var, width=7,
                                  values=['25%', '50%', '75%', '100%'], state='readonly')
        scale_menu.grid(row=0, column=4, padx=4)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(self.root, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.FG,
                 font=('Segoe UI', 9), anchor='w', padx=10
                 ).pack(fill='x', side='bottom', ipady=4)

    # ── formatters / poll ─────────────────────────────────────────────────────

    def _make_formatter(self, var, target_var):
        # Writes to a SEPARATE display StringVar, not back onto var itself --
        # the previous version re-set var from inside var's own write trace,
        # which re-fires the same trace on every set() (Tcl traces fire on
        # every write, matching value or not) and left the Target column
        # showing garbled, many-digit values instead of a clean 3-decimal
        # number. Reported directly with a screenshot: "the value of the
        # target at the gui robot arm controller... too long and mess
        # valeu".
        def _cb(*_):
            target_var.set(f'{var.get():.3f}')
        var.trace_add('write', _cb)

    def _poll_joint_states(self):
        for i, (name, _, _) in enumerate(JOINTS):
            self.current_vars[i].set(f'{self.node.current_positions.get(name, 0.0):.3f}')
            self.velocity_vars[i].set(f'{self.node.current_velocities.get(name, 0.0):.3f}')

        if self._recording:
            t = round(time.time() - self._record_start, 3)
            row = [t] + [self.node.current_positions.get(j[0], 0.0) for j in JOINTS]
            self._record_data.append(row)
            self._rec_status.set(f'Recording… {t:.1f}s  ({len(self._record_data)} frames)')

        self.root.after(100, self._poll_joint_states)

    # ── control helpers ───────────────────────────────────────────────────────

    def _load_preset(self, positions):
        """Set the sliders AND send. A preset button labelled "Pose 1" should
        go to pose 1; requiring a second click on Send Command just looked
        like the button was broken.

        The control-mode check lives in ArmNode.send(), not here: this is an
        ArmGUI method and has no control_mode or logger of its own."""
        if len(positions) != len(JOINTS):
            self.status_var.set(
                f'Preset has {len(positions)} values, expected {len(JOINTS)} — not sent')
            return
        for var, pos in zip(self.slider_vars, positions):
            var.set(pos)
        if self.node.send(positions, self.duration_var.get()):
            self.status_var.set(f'Moving to preset: {positions}')
        else:
            self._warn_wrong_mode()

    def _send(self):
        pos = [round(v.get(), 4) for v in self.slider_vars]
        dur = round(self.duration_var.get(), 1)
        if self.node.send(pos, dur):
            self.status_var.set(f'Sent: {pos}  |  {dur}s')
        else:
            self._warn_wrong_mode()

    def _home(self):
        # len(JOINTS), not a hardcoded 6: joint_trajectory_controller rejects
        # the WHOLE trajectory when the position count does not match the
        # joint-name count, so this button silently did nothing once L7 was
        # added.
        # Gripper rests OPEN at home, so this is not a vector of zeros.
        self._load_preset([HOME_POSE[n] for n, _, _ in JOINTS])
        self.status_var.set('Moving to home…')

    def _zero(self):
        for v in self.slider_vars:
            v.set(0.0)
        self.status_var.set('Sliders zeroed (not sent)')

    # ── recording ─────────────────────────────────────────────────────────────

    def _toggle_record(self):
        if not self._recording:
            self._record_data  = []
            self._record_start = time.time()
            self._recording    = True
            self._rec_btn_var.set('Stop Recording')
            self.status_var.set('Recording joint states…')
        else:
            self._recording = False
            self._rec_btn_var.set('Start Recording')
            self._rec_status.set(f'Stopped — {len(self._record_data)} frames captured')
            self.status_var.set(f'Recording stopped ({len(self._record_data)} frames). Save with "Save CSV".')

    def _save_csv(self):
        if not self._record_data:
            messagebox.showwarning('No data', 'Nothing recorded yet.')
            return
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(RECORDINGS_DIR, f'recording_{ts}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time'] + [j[0] for j in JOINTS])
            writer.writerows(self._record_data)
        self.status_var.set(f'Saved → {path}')
        messagebox.showinfo('Saved', f'Recording saved to:\n{path}')

    def _load_csv(self):
        path = filedialog.askopenfilename(
            initialdir=RECORDINGS_DIR,
            title='Load recording',
            filetypes=[('CSV files', '*.csv')],
        )
        if not path:
            return
        data = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t   = float(row['time'])
                pos = [float(row[j[0]]) for j in JOINTS]
                data.append((t, pos))
        self._loaded_replay = data
        self._rec_status.set(f'Loaded {len(data)} frames from {os.path.basename(path)}')
        self.status_var.set(f'Loaded: {os.path.basename(path)} — press Replay to play')

    def _toggle_replay(self):
        if self._replay_running:
            self._replay_running = False
            self._replay_btn_var.set('Replay')
            self.status_var.set('Replay stopped.')
            return
        if not hasattr(self, '_loaded_replay') or not self._loaded_replay:
            messagebox.showwarning('No data', 'Load a CSV recording first.')
            return
        # Force MANUAL mode before playing back. send() silently drops every
        # frame while CAMERA TELEOPERATION owns the arm (see send()'s mode
        # check) -- without this, clicking Replay while teleop was on used to
        # run the whole thread to completion and report "Replay finished"
        # despite the arm never having moved, the same silent-failure trap
        # documented elsewhere in this file.
        if self.node.control_mode != 'manual':
            self._set_mode('manual')
        self._replay_running = True
        self._replay_btn_var.set('Stop Replay')
        self._replay_thread = threading.Thread(target=self._run_replay, daemon=True)
        self._replay_thread.start()

    def _run_replay(self):
        data   = self._loaded_replay
        t_prev = 0.0
        for t, pos in data:
            if not self._replay_running:
                break
            gap = t - t_prev
            if gap > 0:
                time.sleep(gap)
            self.node.send(pos, 0)
            t_prev = t
        self._replay_running = False
        self.root.after(0, lambda: self._replay_btn_var.set('Replay'))
        self.root.after(0, lambda: self.status_var.set('Replay finished.'))

    # ── capture ───────────────────────────────────────────────────────────────

    def _get_scale(self):
        return int(self._scale_var.get().replace('%', ''))

    def _take_screenshot(self):
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(RECORDINGS_DIR, f'screenshot_{ts}.png')
        self.root.iconify()
        self.root.after(400, lambda: self._do_screenshot(path))

    def _do_screenshot(self, path):
        try:
            r = subprocess.run(
                ['gnome-screenshot', '-f', path],
                capture_output=True, timeout=10
            )
            if r.returncode == 0 and os.path.exists(path):
                pct = self._get_scale()
                if pct != 100:
                    from PIL import Image
                    img = Image.open(path)
                    w = img.width  * pct // 100
                    h = img.height * pct // 100
                    img.resize((w, h), Image.LANCZOS).save(path)
                self.status_var.set(f'Screenshot saved → {path}')
            else:
                err = r.stderr.decode().strip()
                self.status_var.set(
                    f'Screenshot failed: {err or "install gnome-screenshot: sudo apt install gnome-screenshot"}')
        except FileNotFoundError:
            self.status_var.set('Install gnome-screenshot: sudo apt install gnome-screenshot')
        except Exception as e:
            self.status_var.set(f'Screenshot error: {e}')
        finally:
            self.root.deiconify()

    def _toggle_video(self):
        if self._video_proc is None:
            self._start_video()
        else:
            self._stop_video()

    def _start_video(self):
        videos_dir = os.path.expanduser('~/Videos/Screencasts')
        os.makedirs(videos_dir, exist_ok=True)
        self._videos_before = set(os.listdir(videos_dir))
        self._video_proc    = True
        self._video_btn_var.set('Stop Video')
        self.status_var.set(
            'Press  Ctrl+Shift+Alt+R  to start recording — then press Stop Video here when done')
        self._monitor_videos()

    def _monitor_videos(self):
        if not self._video_proc:
            return
        videos_dir  = os.path.expanduser('~/Videos')
        new_files   = set(os.listdir(videos_dir)) - self._videos_before
        active_webm = [f for f in new_files if f.endswith('.webm')]
        if active_webm:
            self.status_var.set(
                f'Recording detected ({active_webm[-1]}) — press Stop Video when done')
        self.root.after(1000, self._monitor_videos)

    def _stop_video(self):
        if not self._video_proc:
            return
        self._video_proc = None
        self._video_btn_var.set('Start Video')
        self.status_var.set('Press  Ctrl+Shift+Alt+R  to stop recording — saving…')
        self.root.after(2500, self._collect_video)

    def _collect_video(self):
        import shutil
        videos_dir = os.path.expanduser('~/Videos/Screencasts')
        try:
            new_files = set(os.listdir(videos_dir)) - self._videos_before
            webms     = sorted(f for f in new_files if f.endswith('.webm'))
            if webms:
                src  = os.path.join(videos_dir, webms[-1])
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest = os.path.join(RECORDINGS_DIR, f'video_{ts}.webm')
                shutil.copy2(src, dest)
                self.status_var.set(f'Video saved → {dest}')
            else:
                self.status_var.set(
                    'No video found — make sure you pressed Ctrl+Shift+Alt+R to record')
        except Exception as e:
            self.status_var.set(f'Video collect error: {e}')

    def _open_recordings_folder(self):
        subprocess.Popen(['xdg-open', RECORDINGS_DIR])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArmNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    ArmGUI(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
