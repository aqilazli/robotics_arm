#!/usr/bin/env python3
"""
depth_calibration_gui.py
========================
Recalibrate the Orbbec Astra Pro's depth sensor against distances you measure
yourself with a tape measure.

WHY THIS EXISTS
---------------
This unit reports depth that is wrong by a large factor: a hand held at a real
500 mm reads as 41 mm, and at a real 1000 mm reads as 56 mm. The readings are
rock-steady (1-2 mm spread over 300 samples), so the sensor is not noisy, it is
mis-scaled -- likely because this Astra Pro runs through the SDK's legacy
OpenNI2 compatibility path, whose calibration setup calls partly fail on
startup. A stable-but-wrong sensor can be corrected in software; that is what
this tool fits.

HOW TO USE IT
-------------
  1. Close the Gesture Monitor first. Only one process can open the camera.
  2. Point the camera at a flat surface (a wall, a book, a box).
  3. Measure the real distance from the CAMERA LENS to that surface with a
     tape measure. Keep the surface filling the green box in the preview.
  4. Type that distance in cm, press "Capture Point".
  5. Repeat at 3 or more different distances, well spread out
     (30 / 60 / 100 / 150 cm is a good set).
  6. Press "Fit + Save". The live "corrected" readout should now agree with
     your tape measure.

WHY 3 POINTS AND NOT 2
----------------------
Any two points fit a straight line perfectly, so two points can never tell you
whether the sensor is actually linear -- the fit looks flawless either way. The
third point is the first one that can disagree, so it is the first real test.
Watch the R-squared and the per-point error after fitting: if they stay good
across the whole range, the correction is trustworthy.

Writes ../config/depth_calibration.json, which perception/inference_node.py
loads at startup and applies to every depth frame.
"""

import ctypes
import importlib.util
import os
import statistics
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)

import depth_calibration as calib_io


def _import_inference_node():
    """
    Borrow the proven SDK bindings from perception/inference_node.py rather
    than keeping a second copy of them in sync. Loaded by file path so this
    works whether or not perception/ is an importable package.
    """
    path = os.path.join(_SCRIPTS_DIR, 'perception', 'inference_node.py')
    spec = importlib.util.spec_from_file_location('_inference_node_for_calib', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAPTURE_LOG = os.path.join(_SCRIPTS_DIR, 'config', 'depth_capture_log.txt')

DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30
ROI_HALF = 20          # half-width of the centre sampling box, in pixels
SAMPLES_PER_CAPTURE = 30
SAMPLE_INTERVAL_MS = 66


class DepthSource:
    """Owns the Orbbec pipeline on a background thread."""

    def __init__(self, ob):
        self._ob = ob
        self._lib = None
        self._pipe = None
        self._config = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._depth = None
        self.error = None
        self.frames_seen = 0

    def start(self):
        self._lib = self._ob._load_orbbec_sdk()
        if self._lib is None:
            self.error = 'OrbbecSDK not found. Is the SDK unpacked in ~/Downloads?'
            return False
        err = (ctypes.c_void_p * 1)(None)
        try:
            self._pipe = self._lib.ob_create_pipeline(err)
            self._config = self._lib.ob_create_config(err)
            self._lib.ob_config_enable_video_stream(
                self._config, self._ob.OB_STREAM_DEPTH, DEPTH_W, DEPTH_H,
                DEPTH_FPS, self._ob.OB_FORMAT_DEPTH, err)
            self._lib.ob_pipeline_start_with_config(self._pipe, self._config, err)
        except Exception as e:
            self.error = f'Could not start the camera: {e}'
            return False
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        err = (ctypes.c_void_p * 1)(None)
        while not self._stop.is_set():
            fs = self._lib.ob_pipeline_wait_for_frameset(self._pipe, 200, err)
            if not fs:
                continue
            df = self._lib.ob_frameset_depth_frame(fs, err)
            if df:
                arr = self._ob._orbbec_depth_mm(self._lib, df)
                if arr is not None:
                    with self._lock:
                        self._depth = arr
                        self.frames_seen += 1
                self._lib.ob_delete_frame(df, err)
            self._lib.ob_delete_frame(fs, err)

    def latest(self):
        with self._lock:
            return None if self._depth is None else self._depth.copy()

    def stop(self):
        self._stop.set()
        time.sleep(0.25)
        err = (ctypes.c_void_p * 1)(None)
        try:
            if self._pipe:
                self._lib.ob_pipeline_stop(self._pipe, err)
                self._lib.ob_delete_pipeline(self._pipe, err)
            if self._config:
                self._lib.ob_delete_config(self._config, err)
        except Exception:
            pass


def centre_reading(depth: np.ndarray) -> float | None:
    """Median of the valid pixels inside the centre box, in raw reported mm."""
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    roi = depth[cy - ROI_HALF:cy + ROI_HALF, cx - ROI_HALF:cx + ROI_HALF]
    valid = roi[roi > 0]
    if valid.size < 10:
        return None
    return float(np.median(valid))


def colorize(depth: np.ndarray) -> np.ndarray:
    """
    Autoscaled preview. A fixed 0-3000 mm range would render this sensor's
    broken 40-60 mm readings as a uniform black image, which hides exactly the
    thing the user needs to aim.
    """
    valid = depth > 0
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        vals = depth[valid]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        vis[valid] = np.clip((vals - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out


class CalibApp:
    BG, PANEL, FG = '#1e1e2e', '#313244', '#cdd6f4'
    ACCENT, GREEN, RED, YELLOW, GREY = '#89b4fa', '#a6e3a1', '#f38ba8', '#f9e2af', '#6c7086'

    def __init__(self, root, source):
        self.root = root
        self.source = source
        self.points = []                 # list of (true_mm, reported_mm)
        self.calib = calib_io.load()     # existing saved fit, if any
        self._capture_buf = None

        root.title('Depth Sensor Calibration')
        root.configure(bg=self.BG)
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._style()
        self._build()
        self._tick()

    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', background=self.BG, foreground=self.FG, font=('Segoe UI', 10))
        s.configure('TFrame', background=self.BG)
        s.configure('TLabel', background=self.BG, foreground=self.FG)
        s.configure('TLabelframe', background=self.BG, foreground=self.ACCENT)
        s.configure('TLabelframe.Label', background=self.BG, foreground=self.ACCENT,
                    font=('Segoe UI', 10, 'bold'))
        s.configure('TButton', background=self.PANEL, foreground=self.FG, padding=6)
        s.configure('Green.TButton', background=self.GREEN, foreground=self.BG, padding=8)
        s.configure('Red.TButton', background=self.RED, foreground=self.BG, padding=8)
        s.configure('TEntry', fieldbackground=self.PANEL, foreground=self.FG)
        s.configure('Treeview', background=self.PANEL, fieldbackground=self.PANEL,
                    foreground=self.FG, rowheight=22)
        s.configure('Treeview.Heading', background=self.BG, foreground=self.ACCENT)

    def _build(self):
        tk.Label(self.root, text='Depth Sensor Calibration', bg=self.BG, fg=self.ACCENT,
                 font=('Segoe UI', 14, 'bold')).grid(row=0, column=0, columnspan=2,
                                                     pady=(10, 6))

        left = ttk.Frame(self.root)
        left.grid(row=1, column=0, padx=10, pady=4, sticky='n')
        right = ttk.Frame(self.root)
        right.grid(row=1, column=1, padx=10, pady=4, sticky='n')

        # ── preview ──────────────────────────────────────────────────────
        pv = ttk.LabelFrame(left, text=' Depth preview ')
        pv.grid(row=0, column=0, sticky='ew')
        self._canvas = tk.Label(pv, bg='black', width=480, height=360)
        self._canvas.grid(row=0, column=0, padx=6, pady=6)
        tk.Label(pv, text='Fill the green box with a flat surface at a known distance',
                 bg=self.BG, fg=self.GREY, font=('Segoe UI', 9)).grid(row=1, column=0,
                                                                      pady=(0, 6))

        # ── live readings ────────────────────────────────────────────────
        rd = ttk.LabelFrame(left, text=' Live reading ')
        rd.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        self._raw_var = tk.StringVar(value='--')
        self._corr_var = tk.StringVar(value='--')
        tk.Label(rd, text='Raw sensor:', bg=self.BG, fg=self.FG).grid(
            row=0, column=0, sticky='e', padx=6, pady=3)
        tk.Label(rd, textvariable=self._raw_var, bg=self.BG, fg=self.YELLOW,
                 font=('Segoe UI', 15, 'bold')).grid(row=0, column=1, sticky='w', padx=6)
        tk.Label(rd, text='Corrected:', bg=self.BG, fg=self.FG).grid(
            row=1, column=0, sticky='e', padx=6, pady=3)
        tk.Label(rd, textvariable=self._corr_var, bg=self.BG, fg=self.GREEN,
                 font=('Segoe UI', 15, 'bold')).grid(row=1, column=1, sticky='w', padx=6)

        # ── capture ──────────────────────────────────────────────────────
        cap = ttk.LabelFrame(right, text=' 1. Capture points ')
        cap.grid(row=0, column=0, sticky='ew')
        tk.Label(cap, text='True distance, lens to surface (cm):', bg=self.BG,
                 fg=self.FG).grid(row=0, column=0, columnspan=2, padx=6, pady=(6, 2),
                                  sticky='w')
        self._dist_var = tk.StringVar(value='50')
        ttk.Entry(cap, textvariable=self._dist_var, width=10).grid(
            row=1, column=0, padx=6, pady=2, sticky='w')
        self._cap_btn = ttk.Button(cap, text='Capture Point', style='Green.TButton',
                                   command=self._capture)
        self._cap_btn.grid(row=1, column=1, padx=6, pady=2, sticky='e')
        self._cap_status = tk.StringVar(value='')
        tk.Label(cap, textvariable=self._cap_status, bg=self.BG, fg=self.YELLOW,
                 font=('Segoe UI', 9)).grid(row=2, column=0, columnspan=2, pady=(0, 6))

        # ── points table ─────────────────────────────────────────────────
        tbl = ttk.LabelFrame(right, text=' 2. Captured points ')
        tbl.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        self._tree = ttk.Treeview(tbl, columns=('true', 'raw', 'err'), show='headings',
                                  height=7)
        for col, head, w in (('true', 'True (cm)', 90), ('raw', 'Raw (mm)', 90),
                             ('err', 'Fit err (cm)', 100)):
            self._tree.heading(col, text=head)
            self._tree.column(col, width=w, anchor='center')
        self._tree.grid(row=0, column=0, columnspan=2, padx=6, pady=6)
        ttk.Button(tbl, text='Remove selected', command=self._remove_selected).grid(
            row=1, column=0, padx=6, pady=(0, 6))
        ttk.Button(tbl, text='Clear all', command=self._clear).grid(
            row=1, column=1, padx=6, pady=(0, 6))

        # ── fit ──────────────────────────────────────────────────────────
        ft = ttk.LabelFrame(right, text=' 3. Fit and save ')
        ft.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        self._fit_var = tk.StringVar(value=self._describe_saved())
        tk.Label(ft, textvariable=self._fit_var, bg=self.BG, fg=self.FG,
                 font=('Segoe UI', 9), justify='left').grid(row=0, column=0,
                                                            columnspan=2, padx=6,
                                                            pady=6, sticky='w')
        ttk.Button(ft, text='Fit + Save', style='Green.TButton',
                   command=self._fit_and_save).grid(row=1, column=0, padx=6, pady=(0, 8))
        ttk.Button(ft, text='Delete saved calibration', style='Red.TButton',
                   command=self._delete_saved).grid(row=1, column=1, padx=6, pady=(0, 8))

        self._conn_var = tk.StringVar(value='connecting...')
        tk.Label(self.root, textvariable=self._conn_var, bg=self.BG, fg=self.GREY,
                 font=('Segoe UI', 9)).grid(row=2, column=0, columnspan=2, pady=(4, 8))

    def _describe_saved(self):
        if self.calib is None:
            return 'No calibration saved yet.\nRaw values are passing through uncorrected.'
        s, o = self.calib
        return (f'Saved:  reported = {s:.5f} x true + {o:.2f}\n'
                f'Applied as:  true = (reported - {o:.2f}) / {s:.5f}')

    # ── live loop ────────────────────────────────────────────────────────
    def _tick(self):
        depth = self.source.latest()
        if depth is None:
            self._conn_var.set(self.source.error or 'waiting for depth frames...')
        else:
            self._conn_var.set(f'camera streaming  |  {self.source.frames_seen} frames  '
                               f'|  {depth.shape[1]}x{depth.shape[0]}')
            raw = centre_reading(depth)
            self._raw_var.set('--' if raw is None else f'{raw:.1f} mm')
            if raw is None or self.calib is None:
                self._corr_var.set('--' if self.calib else 'not calibrated')
            else:
                s, o = self.calib
                self._corr_var.set(f'{(raw - o) / s / 10.0:.1f} cm')
            self._draw(depth)
        self.root.after(66, self._tick)

    def _draw(self, depth):
        vis = colorize(depth)
        h, w = vis.shape[:2]
        cy, cx = h // 2, w // 2
        cv2.rectangle(vis, (cx - ROI_HALF, cy - ROI_HALF),
                      (cx + ROI_HALF, cy + ROI_HALF), (161, 227, 166), 2)
        vis = cv2.resize(vis, (480, 360))
        img = ImageTk.PhotoImage(Image.fromarray(vis[:, :, ::-1]))
        self._canvas.configure(image=img, width=480, height=360)
        self._canvas.image = img

    # ── capture ──────────────────────────────────────────────────────────
    def _capture(self):
        try:
            true_cm = float(self._dist_var.get())
        except ValueError:
            messagebox.showerror('Bad distance', 'Enter the distance as a number, in cm.')
            return
        if true_cm <= 0:
            messagebox.showerror('Bad distance', 'Distance must be greater than zero.')
            return
        if any(abs(p[0] - true_cm * 10.0) < 1.0 for p in self.points):
            messagebox.showwarning('Duplicate',
                                   f'There is already a point at {true_cm:g} cm. '
                                   'Remove it first, or use a different distance.')
            return
        self._capture_buf = []
        self._cap_btn.configure(state='disabled')
        self._collect(true_cm)

    def _collect(self, true_cm):
        depth = self.source.latest()
        if depth is not None:
            r = centre_reading(depth)
            if r is not None:
                self._capture_buf.append(r)
        got, need = len(self._capture_buf), SAMPLES_PER_CAPTURE
        self._cap_status.set(f'sampling... {got}/{need}')
        if got < need:
            self.root.after(SAMPLE_INTERVAL_MS, lambda: self._collect(true_cm))
            return

        self._cap_btn.configure(state='normal')
        reported = statistics.median(self._capture_buf)
        spread = max(self._capture_buf) - min(self._capture_buf)
        self.points.append((true_cm * 10.0, reported))
        self.points.sort()
        self._cap_status.set(f'captured {true_cm:g} cm -> {reported:.1f} mm '
                             f'(spread {spread:.1f} mm)')
        self._refresh_table()
        self._log_points()

    def _log_points(self):
        """
        Append every capture to a plain log as soon as it is taken. A fit that
        is refused (or simply never saved) otherwise leaves no trace of what
        was measured, which makes diagnosing "it did not work" guesswork.
        """
        try:
            with open(CAPTURE_LOG, 'a') as f:
                f.write(f'--- capture at {datetime.now():%H:%M:%S} ---\n')
                for t, r in self.points:
                    f.write(f'    {t/10:g} cm  ->  {r:.1f}\n')
        except OSError:
            pass

    def _refresh_table(self):
        errs = {}
        if len(self.points) >= 2:
            s, o, _ = calib_io.fit(self.points)
            if abs(s) > 1e-9:
                for t, r in self.points:
                    errs[t] = ((r - o) / s - t) / 10.0
        for row in self._tree.get_children():
            self._tree.delete(row)
        for t, r in self.points:
            e = errs.get(t)
            self._tree.insert('', 'end', values=(f'{t / 10:g}', f'{r:.1f}',
                                                 '--' if e is None else f'{e:+.1f}'))

    def _remove_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        del self.points[idx]
        self._refresh_table()

    def _clear(self):
        self.points.clear()
        self._refresh_table()
        self._cap_status.set('')

    # ── fit ──────────────────────────────────────────────────────────────
    def _fit_and_save(self):
        if len(self.points) < 2:
            messagebox.showerror('Not enough points',
                                 'Capture at least 2 points. 3 or more spread across '
                                 'the range you actually use is much better.')
            return
        scale, offset, r2 = calib_io.fit(self.points)
        if scale < 1e-9:
            # Not a fixable miscalibration: a sensor whose readings fail to
            # grow with distance is not reporting distance. Saving this would
            # swap near for far everywhere downstream, so it is refused
            # outright rather than offered as a "save anyway".
            ordered = '\n'.join(f'    {t/10:g} cm  ->  {r:.1f}' for t, r in self.points)
            messagebox.showerror(
                'These readings are not distance',
                'The captured values do not increase with distance:\n\n'
                f'{ordered}\n\n'
                'A depth sensor must report a larger number for a farther '
                'object, and two different distances can never give the same '
                'reading. No scale-and-offset correction can fix this, so '
                'nothing has been saved.\n\n'
                'This means the frames being read are not a depth map (most '
                'likely IR amplitude, or an uninitialised buffer). That is a '
                'decoding or driver problem, not a calibration one.')
            return

        worst = max(abs((r - offset) / scale - t) / 10.0 for t, r in self.points)
        msg = (f'reported = {scale:.5f} x true + {offset:.2f}\n'
               f'R-squared = {r2:.5f}\n'
               f'Worst point error = {worst:.1f} cm\n\n')
        if len(self.points) == 2:
            msg += ('Only 2 points: a line always fits 2 points perfectly, so the '
                    'R-squared here is meaningless. Capture a 3rd point at a different '
                    'distance to actually test it.\n\nSave anyway?')
        elif worst > 5.0:
            msg += ('The fit misses by more than 5 cm at some point, so the sensor may '
                    'not be linear over this range.\n\nSave anyway?')
        else:
            msg += 'Save this calibration?'

        if not messagebox.askyesno('Fit result', msg):
            return
        calib_io.save(scale, offset, self.points, r2)
        self.calib = (scale, offset)
        self._fit_var.set(self._describe_saved())
        messagebox.showinfo('Saved',
                            f'Written to {calib_io.CALIB_PATH}\n\n'
                            'Restart the Gesture Monitor for it to take effect on the '
                            'live pipeline. Recordings made from now on will carry '
                            'corrected depth.')

    def _delete_saved(self):
        if not os.path.exists(calib_io.CALIB_PATH):
            messagebox.showinfo('Nothing to delete', 'No calibration file is saved.')
            return
        if not messagebox.askyesno('Delete calibration',
                                   'Delete the saved calibration? Depth will go back to '
                                   'raw uncorrected values.'):
            return
        os.remove(calib_io.CALIB_PATH)
        self.calib = None
        self._fit_var.set(self._describe_saved())

    def _on_close(self):
        self.source.stop()
        self.root.destroy()


def main():
    ob = _import_inference_node()
    source = DepthSource(ob)
    ok = source.start()

    root = tk.Tk()
    if not ok:
        messagebox.showerror('Camera unavailable',
                             f'{source.error}\n\nClose the Gesture Monitor if it is '
                             'running -- only one process can open the camera at a time.')
        root.destroy()
        return
    CalibApp(root, source)
    root.mainloop()


if __name__ == '__main__':
    main()
