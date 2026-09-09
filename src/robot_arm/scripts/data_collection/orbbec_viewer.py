#!/usr/bin/env python3
"""
Orbbec Astra Pro RGBD Viewer
- Color: /dev/video0  (standard UVC, read via OpenCV)
- Depth: OrbbecSDK    (depth sensor via libOrbbecSDK.so)

Controls:
  q / ESC  - quit (either window)
  s        - save snapshot (color PNG + raw depth 16-bit + colorized depth)
  d        - cycle depth colormap  (TURBO → JET → HOT)
  +  / -   - increase / decrease depth display range
  Depth Control window - drag the slider to set the depth range directly
"""

import ctypes
import os
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

# ── Global cleanup state ──────────────────────────────────────────────────────
_cleanup_done = False
_cap          = None
_pipeline     = None
_lib          = None

def _cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    cv2.destroyAllWindows()
    if _cap is not None:
        _cap.release()
    if _pipeline is not None and _lib is not None:
        err = (ctypes.c_void_p * 1)(None)
        _lib.ob_pipeline_stop(_pipeline, err)
        # SDK spawns background USB threads that need time to cancel their
        # async transfers before the pipeline is deleted; without this delay
        # those threads keep the IR projector active after exit.
        time.sleep(1.5)
        _lib.ob_delete_pipeline(_pipeline, err)
    # Force-exit so no SDK background threads linger after cleanup
    os._exit(0)

def _signal_handler(sig, frame):
    _cleanup()

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGHUP,  _signal_handler)

# ── Paths ─────────────────────────────────────────────────────────────────────
SDK_DIR = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release",
    "OrbbecSDK_v1.10.27",
    "SDK", "lib",
)
LIB_PATH = os.path.join(SDK_DIR, "libOrbbecSDK.so")

COLOR_DEVICE = None  # auto-detected by name at runtime

# ── SDK enums ─────────────────────────────────────────────────────────────────
OB_STREAM_DEPTH = 3
OB_FORMAT_Y12   = 12   # Astra Pro depth; SDK unpacks to 16-bit in memory

COLORMAPS      = [cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_HOT]
COLORMAP_NAMES = ["TURBO", "JET", "HOT"]

WIN_NAME        = "Orbbec Astra Pro — RGBD"
DEPTH_MAX_MIN   = 500
DEPTH_MAX_LIMIT = 10000
DEPTH_STEP      = 100    # mm per +/- keypress
DEPTH_DEFAULT   = 3000


# ── SDK loader ────────────────────────────────────────────────────────────────
def load_sdk():
    for f in os.listdir(SDK_DIR):
        if ".so" in f:
            try:
                ctypes.CDLL(os.path.join(SDK_DIR, f))
            except OSError:
                pass
    return ctypes.CDLL(LIB_PATH)


def setup_api(lib):
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
        ctypes.c_int, ctypes.c_int, ctypes.c_int, PP,
    ]

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

    lib.ob_error_message.restype  = ctypes.c_char_p
    lib.ob_error_message.argtypes = [ctypes.c_void_p]


def new_err():
    return (ctypes.c_void_p * 1)(None)


def check_error(lib, err_ptr):
    if err_ptr[0]:
        msg = lib.ob_error_message(err_ptr[0])
        lib.ob_delete_error(err_ptr[0])
        err_ptr[0] = None
        raise RuntimeError(f"OrbbecSDK: {msg.decode()}")


# ── Depth decoder ─────────────────────────────────────────────────────────────
def get_depth_mm(lib, frame):
    """Return float32 depth array in mm from a depth frame."""
    err = new_err()
    w     = lib.ob_video_frame_width(frame, err)
    h     = lib.ob_video_frame_height(frame, err)
    sz    = lib.ob_frame_data_size(frame, err)
    ptr   = lib.ob_frame_data(frame, err)
    scale = lib.ob_depth_frame_get_value_scale(frame, err)
    if not ptr or sz < 2:
        return None
    raw = np.frombuffer(
        (ctypes.c_uint16 * (sz // 2)).from_address(ptr), dtype=np.uint16
    ).copy()
    return raw.reshape((h, w)).astype(np.float32) * scale


def colorize_depth(depth_mm, colormap_idx, depth_max_mm):
    valid = (depth_mm > 0) & (depth_mm < depth_max_mm)
    norm  = np.zeros(depth_mm.shape, dtype=np.uint8)
    norm[valid] = np.clip(
        depth_mm[valid] / depth_max_mm * 255, 0, 255
    ).astype(np.uint8)
    vis = cv2.applyColorMap(norm, COLORMAPS[colormap_idx])
    vis[~valid] = 0
    return vis


# ── Auto-detect color camera ──────────────────────────────────────────────────
def find_color_device():
    """Return the /dev/videoN index whose sysfs name contains 'Astra Pro HD'."""
    import glob
    for node in sorted(glob.glob('/sys/class/video4linux/video*/name')):
        try:
            name = open(node).read().strip()
            if 'Astra Pro HD' in name:
                idx = int(node.split('video')[2].split('/')[0])
                return idx
        except Exception:
            pass
    return None


# ── Depth Control window (Tkinter) ─────────────────────────────────────────────
# A real cv2.createTrackbar/setMouseCallback slider was tried first, but the
# Qt/Wayland combo this runs on either fails to paint it or doesn't deliver
# mouse events to it — both fail silently with no exception. Tkinter is a
# completely different toolkit (same one arm_gui.py already relies on) and
# doesn't share that failure mode, so the depth-range control lives here
# instead, in its own window/thread, updating depth_state under the GIL.
def _run_depth_control_window(depth_state: dict, stop_event: threading.Event):
    root = tk.Tk()
    root.title('Depth Control')
    root.attributes('-topmost', True)
    root.geometry('420x110')

    label_var = tk.StringVar(value=f"Depth Max: {depth_state['max_mm']} mm")
    dragging  = {'active': False}

    def _on_slide(val):
        snapped = round(float(val) / DEPTH_STEP) * DEPTH_STEP
        depth_state['max_mm'] = int(snapped)
        label_var.set(f"Depth Max: {int(snapped)} mm")

    tk.Label(root, textvariable=label_var, font=('Segoe UI', 12, 'bold')).pack(pady=(12, 4))

    scale = ttk.Scale(
        root, from_=DEPTH_MAX_MIN, to=DEPTH_MAX_LIMIT,
        orient='horizontal', command=_on_slide, length=380,
    )
    scale.pack(padx=20, pady=4)
    scale.bind('<ButtonPress-1>',   lambda e: dragging.__setitem__('active', True))
    scale.bind('<ButtonRelease-1>', lambda e: dragging.__setitem__('active', False))

    # ttk.Scale needs its geometry realized before .set() lands on the right
    # spot, otherwise it can snap to an end of the range (same class of bug
    # as the earlier Qt trackbar not being realized before use).
    root.update_idletasks()
    scale.set(depth_state['max_mm'])

    range_row = tk.Frame(root)
    range_row.pack(fill='x', padx=20)
    tk.Label(range_row, text=f'{DEPTH_MAX_MIN} mm', font=('Segoe UI', 8)).pack(side='left')
    tk.Label(range_row, text=f'{DEPTH_MAX_LIMIT} mm', font=('Segoe UI', 8)).pack(side='right')

    def _on_close():
        stop_event.set()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', _on_close)

    def _poll_stop():
        if stop_event.is_set():
            root.destroy()
            return
        # keep the slider in sync when +/- keys change depth_max_mm in the
        # OpenCV window — but never fight the user's own drag in progress
        if not dragging['active']:
            current = depth_state['max_mm']
            if round(scale.get()) != current:
                scale.set(current)
                label_var.set(f"Depth Max: {current} mm")
        root.after(150, _poll_stop)

    root.after(150, _poll_stop)
    root.mainloop()


# ── OpenCV capture/display loop ────────────────────────────────────────────────
def _run_capture_loop(lib, pipeline, cap, depth_state: dict, stop_event: threading.Event):
    colormap_idx = 0
    snapshot_n   = 0
    fps_t        = time.time()
    fps_count    = 0
    fps_val      = 0.0
    depth_mm_last = None

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 480)

    print("Streaming — q/ESC quit | s save | d colormap | +/- or Depth Control window: depth range")

    try:
        while not stop_event.is_set():
            depth_max_mm = depth_state['max_mm']

            # ── color frame ───────────────────────────────────────────────────
            ret, color_bgr = cap.read()
            if not ret or color_bgr is None:
                color_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

            # ── depth frame ───────────────────────────────────────────────────
            err      = new_err()
            frameset = lib.ob_pipeline_wait_for_frameset(pipeline, 33, err)
            if err[0]:
                lib.ob_delete_error(err[0]); err[0] = None
                frameset = None

            depth_vis = None
            if frameset:
                depth_frame = lib.ob_frameset_depth_frame(frameset, err)
                if depth_frame:
                    depth_mm_last = get_depth_mm(lib, depth_frame)
                    lib.ob_delete_frame(depth_frame, err)
                lib.ob_delete_frame(frameset, err)

            if depth_mm_last is not None:
                depth_vis = colorize_depth(depth_mm_last, colormap_idx, depth_max_mm)

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_val   = fps_count / (now - fps_t)
                fps_count = 0
                fps_t     = now

            # ── compose display ───────────────────────────────────────────────
            H   = 480
            W   = 640
            blank = np.zeros((H, W, 3), dtype=np.uint8)

            left  = cv2.resize(color_bgr, (W, H)) if color_bgr is not None else blank.copy()
            right = cv2.resize(depth_vis,  (W, H)) if depth_vis  is not None else blank.copy()

            cv2.putText(left,  f"Color  FPS {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(right, f"Depth  max {depth_max_mm} mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(right, COLORMAP_NAMES[colormap_idx], (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(right, "Drag depth range in 'Depth Control' window", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow(WIN_NAME, np.hstack([left, right]))

            key = cv2.waitKey(1) & 0xFF

            # quit if window was closed via the X button
            # WND_PROP_AUTOSIZE returns -1 when the window no longer exists;
            # WND_PROP_VISIBLE is unreliable on Wayland/XWayland
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                break

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"color_{ts}.png", left)
                if depth_mm_last is not None:
                    raw16 = np.clip(depth_mm_last, 0, 65535).astype(np.uint16)
                    cv2.imwrite(f"depth_raw_{ts}.png", raw16)
                    cv2.imwrite(f"depth_vis_{ts}.png", right)
                snapshot_n += 1
                print(f"Snapshot #{snapshot_n} saved: {ts}")
            elif key == ord('d'):
                colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            elif key in (ord('+'), ord('='), ord('-')):
                step = DEPTH_STEP if key != ord('-') else -DEPTH_STEP
                depth_state['max_mm'] = max(DEPTH_MAX_MIN, min(DEPTH_MAX_LIMIT, depth_max_mm + step))

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _cap, _pipeline, _lib

    # ── Depth sensor (OrbbecSDK) — open first so USB settles ─────────────────
    print("Loading OrbbecSDK …")
    try:
        lib  = load_sdk()
        _lib = lib
        setup_api(lib)
    except OSError as e:
        sys.exit(f"Cannot load libOrbbecSDK.so: {e}")

    err      = new_err()
    pipeline  = lib.ob_create_pipeline(err); check_error(lib, err)
    _pipeline = pipeline
    config   = lib.ob_create_config(err);   check_error(lib, err)

    lib.ob_config_enable_video_stream(
        config, OB_STREAM_DEPTH, 640, 480, 30, OB_FORMAT_Y12, err
    )
    check_error(lib, err)

    print("Starting depth pipeline …")
    lib.ob_pipeline_start_with_config(pipeline, config, err); check_error(lib, err)
    lib.ob_delete_config(config, err); check_error(lib, err)

    # ── Color camera — open after SDK so device index is stable ──────────────
    dev = find_color_device()
    if dev is None:
        print("Warning: Astra Pro HD Camera not found by name, trying index 0")
        dev = 0
    print(f"Opening color camera /dev/video{dev} …")
    _cap = cv2.VideoCapture(dev)
    cap  = _cap
    if not cap.isOpened():
        sys.exit(f"Cannot open /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    depth_state = {'max_mm': DEPTH_DEFAULT}
    stop_event  = threading.Event()

    capture_thread = threading.Thread(
        target=_run_capture_loop,
        args=(lib, pipeline, cap, depth_state, stop_event),
        daemon=True,
    )
    capture_thread.start()

    try:
        _run_depth_control_window(depth_state, stop_event)   # blocks until closed
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        capture_thread.join(timeout=3.0)
        print("Stopping …")
        _cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
