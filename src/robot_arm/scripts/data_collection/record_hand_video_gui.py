#!/usr/bin/env python3
"""
record_hand_video_gui.py
==========================
Records the live camera feed (with hand-skeleton overlay) to an .mp4 file,
AND saves real Orbbec depth alongside it (N_depth.npz, one frame per video
frame, same order) so the recording carries real calibrated depth, not
just color pixels. Turning this pair into dataset.npz happens separately,
step by step, in dataset_ipynb/00_video_to_dataset.ipynb — so you can
actually see how MediaPipe Hands, feature extraction, depth fusion, and
window building work, instead of it happening invisibly here.

The output filename defaults to the next unused N.mp4 (1.mp4, 2.mp4, ...)
each time -- always safe to accept as-is. Trying to record over an
existing file now asks for explicit confirmation first, instead of
silently overwriting it (which used to lose depth data from earlier
recordings when they all defaulted to the same "session.mp4").

A plain .mp4 has no channel for depth -- that's why earlier recordings
(before this depth-saving was added) could only ever produce MediaPipe's
own scale-ambiguous z guess when reprocessed, not real depth. Saving depth
as a separate synchronized array alongside the video is what actually
fixes that.

Prerequisite: inference_node.py must already be running (it opens the
camera and publishes /arm/color_image).

Usage
-----
    python3 record_hand_video_gui.py
"""

import os
import sys
import time
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import depth_calibration

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'recordings')
DEFAULT_TOPIC = '/arm/color_image'
DEPTH_TOPIC = '/arm/depth_image'
DEFAULT_DURATION = 120
VIDEO_W = 360
RECORDING_FPS = 15   # matches inference_node.py's default mediapipe_fps


def depth_path_for(video_path: str) -> str:
    """session.mp4 -> session_depth.npz -- kept next to the video, one
    stacked array of real depth (uint16, mm), one entry per video frame,
    same order, so the notebook can re-pair frame i's color with depth[i]."""
    root, _ = os.path.splitext(video_path)
    return root + '_depth.npz'


def next_available_output_path(recordings_dir: str = RECORDINGS_DIR) -> str:
    """Every recording used to default to the same 'session.mp4', so a
    second recording silently overwrote the first one's depth file before
    it ever got renamed -- three separate sessions' worth of real depth
    were lost that way in one afternoon. This picks the next unused
    N.mp4 (1.mp4, 2.mp4, 3.mp4, ...) automatically, so accepting the
    default is always safe -- no filename to remember to change."""
    os.makedirs(recordings_dir, exist_ok=True)
    n = 1
    while True:
        candidate = os.path.join(recordings_dir, f'{n}.mp4')
        if not os.path.exists(candidate) and not os.path.exists(depth_path_for(candidate)):
            return candidate
        n += 1


DEFAULT_OUT = next_available_output_path()


# ── ROS2 node ───────────────────────────────────────────────────────────────

class VideoRecorderNode(Node):
    def __init__(self, topic=DEFAULT_TOPIC):
        super().__init__('hand_video_recorder_gui')
        self.topic_name = topic
        self.total_msgs = 0
        self.last_msg_time = 0.0
        self.latest_frame = None       # RGB numpy array, for the GUI preview
        self.last_frame_time = 0.0

        self.recording = False
        self.record_start = 0.0
        self.video_path = None
        self.video_writer = None
        self.frames_written = 0

        # depth, synchronized frame-for-frame with the color video
        self._latest_depth_mm = None   # most recent depth frame, (480,640) float32 mm
        self.depth_frames = []         # accumulated during recording, one per video frame
        self.depth_msgs_seen = 0

        self.sub = self.create_subscription(RosImage, topic, self._image_cb, 5)
        self.depth_sub = self.create_subscription(RosImage, DEPTH_TOPIC, self._depth_cb, 5)

    def resubscribe(self, topic):
        self.destroy_subscription(self.sub)
        self.topic_name = topic
        self.total_msgs = 0
        self.last_msg_time = 0.0
        self.sub = self.create_subscription(RosImage, topic, self._image_cb, 5)

    def _depth_cb(self, msg):
        if msg.encoding != '32FC1':
            return
        self.depth_msgs_seen += 1
        self._latest_depth_mm = np.frombuffer(
            msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()

    def start_recording(self, video_path: str):
        self.video_path = video_path
        self.video_writer = None   # opened lazily once we know the frame size
        self.frames_written = 0
        self.depth_frames = []
        self.record_start = time.monotonic()
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.depth_frames:
            depth_path = depth_path_for(self.video_path)
            stacked = np.stack(self.depth_frames).astype(np.uint16)   # mm, integer precision is plenty
            # Whether inference_node had a calibration loaded when these frames
            # were captured. Recordings made before one existed hold the
            # sensor's raw mis-scaled values; ones made after hold true mm.
            # Same filename and same array either way, so this flag is what
            # stops a consumer from correcting an already-corrected file.
            # Files with no flag at all predate this and are raw.
            np.savez_compressed(depth_path, depth_mm=stacked,
                                calibrated=np.uint8(1 if depth_calibration.load() else 0),
                                depth_format='Y11')
            self.get_logger().info(f'Saved depth: {stacked.shape} -> {depth_path}')

    def _image_cb(self, msg):
        if msg.encoding != 'bgr8':
            return
        self.total_msgs += 1
        self.last_msg_time = time.monotonic()

        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_frame = bgr[:, :, ::-1].copy()   # BGR -> RGB, for the GUI preview

        if self.recording:
            if self.video_writer is None:
                h, w = bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, RECORDING_FPS, (w, h))
            self.video_writer.write(bgr)
            self.frames_written += 1
            # pair this video frame with whatever the latest depth reading is --
            # depth publishes slower (depth_fps, default 15) than color usually
            # arrives, so consecutive video frames may reuse the same depth
            # frame; still far better than no depth at all (MediaPipe's guess).
            if self._latest_depth_mm is not None:
                self.depth_frames.append(self._latest_depth_mm)
            else:
                self.depth_frames.append(np.zeros((msg.height, msg.width), dtype=np.float32))


# ── GUI ─────────────────────────────────────────────────────────────────────

class RecorderGUI:
    BG     = '#1e1e2e'
    PANEL  = '#313244'
    FG     = '#cdd6f4'
    ACCENT = '#89b4fa'
    GREEN  = '#a6e3a1'
    RED    = '#f38ba8'
    YELLOW = '#f9e2af'
    GREY   = '#6c7086'

    def __init__(self, root, node: VideoRecorderNode):
        self.root = root
        self.node = node
        root.title('Hand Video Recorder')
        root.resizable(True, True)
        root.minsize(560, 480)
        root.configure(bg=self.BG)

        self._apply_styles()

        self._target_duration = DEFAULT_DURATION
        self._out_path_var = tk.StringVar(value=os.path.normpath(DEFAULT_OUT))
        self._duration_var = tk.IntVar(value=DEFAULT_DURATION)
        self._photo = None

        self._build_ui()
        self._poll()

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.',              background=self.BG,    foreground=self.FG, font=('Segoe UI', 10))
        s.configure('TFrame',         background=self.BG)
        s.configure('TLabel',         background=self.BG,    foreground=self.FG)
        s.configure('TLabelframe',    background=self.BG,    foreground=self.ACCENT)
        s.configure('TLabelframe.Label', background=self.BG, foreground=self.ACCENT, font=('Segoe UI', 10, 'bold'))
        s.configure('TButton',        background=self.PANEL, foreground=self.FG, padding=6)
        s.configure('Green.TButton',  background=self.GREEN, foreground=self.BG, padding=8)
        s.configure('Red.TButton',    background=self.RED,   foreground=self.BG, padding=8)
        s.configure('TEntry',         fieldbackground=self.PANEL, foreground=self.FG)
        s.configure('TSpinbox',       fieldbackground=self.PANEL, foreground=self.FG)
        s.configure('rec.Horizontal.TProgressbar', troughcolor=self.PANEL, background=self.GREEN)
        s.map('TButton',       background=[('active', '#45475a')])
        s.map('Green.TButton', background=[('active', '#94e2a1')])
        s.map('Red.TButton',   background=[('active', '#eb8ba8')])

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        tk.Label(self.root, text='Hand Video Recorder',
                 bg=self.BG, fg=self.ACCENT, font=('Segoe UI', 14, 'bold')).grid(
            row=0, column=0, pady=(12, 8))

        body = ttk.Frame(self.root)
        body.grid(row=1, column=0, padx=12, sticky='n')

        self._build_video_panel(body)
        self._build_connection_panel(body)
        self._build_recording_panel(body)
        self._build_statusbar()

    def _build_video_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=f'{DEFAULT_TOPIC}  (live camera feed)')
        frame.pack()
        # no width/height on the Label — those are text character units until
        # an image is set, which renders huge; a blank image pins real pixels.
        self._blank_photo = ImageTk.PhotoImage(Image.new('RGB', (VIDEO_W, int(VIDEO_W * 0.75)), '#000000'))
        self._video_lbl = tk.Label(frame, image=self._blank_photo)
        self._video_lbl.pack(padx=6, pady=6)
        self._video_status_var = tk.StringVar(value='waiting for frames…')
        tk.Label(frame, textvariable=self._video_status_var, bg=self.BG, fg=self.GREY,
                 font=('Courier', 9)).pack(pady=(0, 6))

    def _build_connection_panel(self, parent):
        frame = ttk.LabelFrame(parent, text='Camera Feed — is inference_node.py even running?')
        frame.pack(fill='x', pady=(10, 0))

        status_row = ttk.Frame(frame)
        status_row.pack(fill='x', padx=10, pady=10)

        self._dot_var = tk.StringVar(value='⬤')
        self._dot_lbl = tk.Label(status_row, textvariable=self._dot_var,
                                  bg=self.BG, fg=self.RED, font=('Segoe UI', 16))
        self._dot_lbl.grid(row=0, column=0, rowspan=2, padx=(0, 10))

        self._conn_text_var = tk.StringVar(value='NO DATA — start inference_node.py first')
        tk.Label(status_row, textvariable=self._conn_text_var, bg=self.BG, fg=self.RED,
                 font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, sticky='w')

        self._conn_detail_var = tk.StringVar(value='frames received: 0')
        tk.Label(status_row, textvariable=self._conn_detail_var, bg=self.BG, fg=self.GREY,
                 font=('Courier', 9)).grid(row=1, column=1, sticky='w')

    def _build_recording_panel(self, parent):
        frame = ttk.LabelFrame(parent, text='Record')
        frame.pack(fill='x', pady=(10, 0))

        r1 = ttk.Frame(frame); r1.pack(fill='x', padx=10, pady=(10, 4))
        ttk.Label(r1, text='Duration (s):').pack(side='left')
        ttk.Spinbox(r1, from_=5, to=3600, textvariable=self._duration_var, width=8).pack(side='left', padx=6)

        r2 = ttk.Frame(frame); r2.pack(fill='x', padx=10, pady=(4, 10))
        ttk.Label(r2, text='Output file:').pack(side='left')
        ttk.Entry(r2, textvariable=self._out_path_var, width=26).pack(side='left', padx=6)
        ttk.Button(r2, text='Browse…', command=self._browse_out).pack(side='left')

        self._rec_btn_var = tk.StringVar(value='Start Recording')
        self._rec_btn = ttk.Button(frame, textvariable=self._rec_btn_var,
                                    style='Green.TButton', command=self._toggle_recording)
        self._rec_btn.pack(padx=10, pady=(6, 6))

        self._progress = ttk.Progressbar(frame, style='rec.Horizontal.TProgressbar',
                                          length=280, maximum=100, value=0)
        self._progress.pack(padx=10, pady=4)

        self._rec_status_var = tk.StringVar(value='Idle — press Start Recording')
        tk.Label(frame, textvariable=self._rec_status_var, bg=self.BG, fg=self.YELLOW,
                 font=('Courier', 10)).pack(padx=10, pady=(0, 10))

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(self.root, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.FG, font=('Segoe UI', 9),
                 anchor='w', padx=10).grid(row=2, column=0, sticky='ew', ipady=4, pady=(10, 0))

    # ── actions ───────────────────────────────────────────────────────────

    def _browse_out(self):
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self._out_path_var.get()) or '.',
            initialfile=os.path.basename(self._out_path_var.get()),
            title='Save video as',
            defaultextension='.mp4',
            filetypes=[('MP4 video', '*.mp4')],
        )
        if path:
            self._out_path_var.set(path)

    def _toggle_recording(self):
        if not self.node.recording:
            self._target_duration = self._duration_var.get()
            out_path = self._out_path_var.get()
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

            # Never silently overwrite -- this is exactly how 3 earlier
            # recordings lost their depth data (each new take reused the
            # same default filename before being renamed). Require an
            # explicit "yes, overwrite" instead of just clobbering it.
            existing = [p for p in (out_path, depth_path_for(out_path)) if os.path.exists(p)]
            if existing:
                ok = messagebox.askyesno(
                    'File already exists',
                    'This would overwrite:\n' + '\n'.join(existing) +
                    '\n\nRecord anyway and replace it?')
                if not ok:
                    self.status_var.set('Recording cancelled -- pick a different output file.')
                    return

            self.node.start_recording(out_path)
            self._rec_btn_var.set('Stop && Save')
            self._rec_btn.configure(style='Red.TButton')
            self.status_var.set('Recording…')
        else:
            self._finish_recording()

    def _finish_recording(self):
        path = self.node.video_path
        n = self.node.frames_written
        self.node.stop_recording()
        self._rec_btn_var.set('Start Recording')
        self._rec_btn.configure(style='Green.TButton')

        if n < 30:
            if path and os.path.exists(path):
                os.remove(path)
            depth_path = depth_path_for(path) if path else None
            if depth_path and os.path.exists(depth_path):
                os.remove(depth_path)
            messagebox.showwarning(
                'Too short',
                f'Only {n} frames captured — record for longer.\nNothing was saved.')
            self._rec_status_var.set(f'Stopped — only {n} frames, not saved')
            self.status_var.set('Recording discarded (too short).')
            return

        depth_path = depth_path_for(path)
        has_depth = os.path.exists(depth_path)
        self._rec_status_var.set(f'Saved {n} frames → {os.path.basename(path)}'
                                  + (' (+ depth)' if has_depth else ' (no depth!)'))
        self.status_var.set(f'Saved: {path}')
        messagebox.showinfo(
            'Video saved',
            f'Saved to:\n{path}\n'
            + (f'Depth:\n{depth_path}\n\n' if has_depth else
               '\n[WARNING] No depth was captured -- is inference_node.py running '
               'with use_depth:=true and an Orbbec camera connected?\n\n')
            + f'{n} frames (~{n / RECORDING_FPS:.0f}s)\n\n'
            'Next: open dataset_ipynb/00_video_to_dataset.ipynb to convert this '
            'into dataset.npz.')

        # Auto-advance to the next unused number, so the NEXT recording
        # can't accidentally reuse (and overwrite) this one either.
        self._out_path_var.set(next_available_output_path())

    # ── poll loop ─────────────────────────────────────────────────────────

    def _poll(self):
        now = time.monotonic()

        age = now - self.node.last_msg_time if self.node.total_msgs else float('inf')
        alive = age < 1.0
        self._dot_lbl.configure(fg=self.GREEN if alive else self.RED)
        if self.node.total_msgs == 0:
            self._conn_text_var.set('NO DATA — start inference_node.py first')
        elif alive:
            self._conn_text_var.set(f'LIVE on {self.node.topic_name}')
        else:
            self._conn_text_var.set(f'STALLED — last frame {age:.1f}s ago')
        self._conn_detail_var.set(f'frames received: {self.node.total_msgs}')

        frame_age = now - self.node.last_frame_time if self.node.last_frame_time else float('inf')
        if self.node.latest_frame is not None and frame_age < 2.0:
            img = Image.fromarray(self.node.latest_frame)
            w, h = img.size
            new_h = int(VIDEO_W * h / w)
            img = img.resize((VIDEO_W, new_h))
            self._photo = ImageTk.PhotoImage(img)
            self._video_lbl.configure(image=self._photo)
            self._video_status_var.set(f'live — {frame_age*1000:.0f} ms old')
        else:
            self._video_status_var.set('no recent frame — check inference_node.py is running')

        if self.node.recording:
            elapsed = now - self.node.record_start
            pct = min(100, 100 * elapsed / self._target_duration)
            self._progress['value'] = pct
            self._rec_status_var.set(
                f'Recording… {elapsed:.1f}s / {self._target_duration}s   '
                f'({self.node.frames_written} frames)')
            if elapsed >= self._target_duration:
                self._finish_recording()

        self.root.after(100, self._poll)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = VideoRecorderNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    RecorderGUI(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
