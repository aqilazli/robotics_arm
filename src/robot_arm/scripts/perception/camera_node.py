#!/usr/bin/env python3
"""
camera_node.py  —  ROS 2 node
===============================
Camera-only sanity check. Opens the camera, shows a live window, and
publishes /arm/color_image — nothing else. No MediaPipe, no depth fusion,
no predictor. Use this to confirm the camera/hardware works BEFORE dealing
with the full inference_node.py pipeline.

Usage
-----
    python3 perception/camera_node.py
    python3 perception/camera_node.py --ros-args -p camera_index:=2
    python3 perception/camera_node.py --ros-args -p show_window:=false   # headless, topic only

Published topics
----------------
  /arm/color_image   sensor_msgs/Image   raw camera frame (bgr8)

Parameters
----------
  camera_index   int   video device index (-1 = auto-detect, default)
  camera_fps     int   target capture rate (default 30)
  camera_width   int   capture resolution width  (default 640)
  camera_height  int   capture resolution height (default 480)
  show_window    bool  show a live OpenCV window (default true)
"""

import os
import sys
import time

os.environ.setdefault('DISPLAY', ':0')
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

WIN = 'Camera Check'


def _auto_detect_camera() -> int:
    """Prefer USB/Orbbec/Astra cameras, skip IR nodes, verify brightness."""
    import glob as _glob
    candidates = []

    for node in sorted(_glob.glob('/sys/class/video4linux/video*/name')):
        try:
            name = open(node).read().strip()
            idx  = int(node.split('video')[2].split('/')[0])
            name_u = name.upper()
            if 'IR' in name_u:
                print(f'[camera] skip  video{idx}: {name} (IR)')
                continue
            prio = 0 if any(k in name_u for k in ('ORBBEC', 'ASTRA', 'USB', 'OBSENSOR')) else 1
            candidates.append((prio, idx, name))
            print(f'[camera] found video{idx}: {name}  prio={prio}')
        except Exception:
            pass

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


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('camera_index',  -1)
        self.declare_parameter('camera_fps',    30)
        self.declare_parameter('camera_width',  640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('show_window',   True)

        cam_idx = self.get_parameter('camera_index').value
        self._fps   = self.get_parameter('camera_fps').value
        self._cam_w = self.get_parameter('camera_width').value
        self._cam_h = self.get_parameter('camera_height').value
        _sw = self.get_parameter('show_window').value
        self._show  = _sw if isinstance(_sw, bool) else str(_sw).lower() in ('true', '1', 'yes')

        self._pub_img = self.create_publisher(Image, '/arm/color_image', 10)

        if cam_idx < 0:
            cam_idx = _auto_detect_camera()
        self.get_logger().info(f'Opening camera index {cam_idx}')
        self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open camera {cam_idx}')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cam_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_h)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if self._show:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        self._frame_count = 0
        self.get_logger().info('CameraNode ready — publishing /arm/color_image')

    def spin_loop(self):
        interval = 1.0 / self._fps
        while rclpy.ok():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.get_logger().warn('No frame read from camera')
                time.sleep(0.05)
                continue

            self._frame_count += 1
            self._publish_image(frame)

            if self._show:
                brightness = float(np.mean(frame))
                overlay = frame.copy()
                cv2.putText(overlay, f'frame {self._frame_count}  {frame.shape[1]}x{frame.shape[0]}  '
                                      f'brightness={brightness:.0f}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, 'q / Esc to quit', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(WIN, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

            rclpy.spin_once(self, timeout_sec=0.0)
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _publish_image(self, bgr: np.ndarray):
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_frame'
        msg.height   = bgr.shape[0]
        msg.width    = bgr.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = bgr.shape[1] * 3
        msg.data     = bgr.tobytes()
        self._pub_img.publish(msg)

    def destroy_node(self):
        self._cap.release()
        if self._show:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        node.spin_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
