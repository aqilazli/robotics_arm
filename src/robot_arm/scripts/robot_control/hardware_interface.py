#!/usr/bin/env python3
"""
hardware_interface.py  —  ROS 2 node
======================================
Bridges ROS 2 joint commands to physical hardware via USB serial.

Subscribed topics
-----------------
  /robot_arm_controller/joint_trajectory  trajectory_msgs/JointTrajectory
      → parses joint positions, sends to Arduino/STM32 over serial

Serial protocol (sent to Arduino)
----------------------------------
  Joint command:   J:L1:0.20,L2:0.50,L3:-0.30,L4:0.00,L5:0.00,L6:0.00,L7:0.000\n
                   L7 is the gripper and is in METRES of finger travel,
                   not radians. The real PAROL6 firmware is a 6-axis
                   protocol and drives its gripper separately, so a real
                   controller may ignore or reject the L7 field; this is
                   only exercised in simulation today.
  Stop command:    STOP\n
  Arduino replies: OK\n  (optional — used for health check)

Parameters (ros2 run … --ros-args -p key:=value)
  port      str   serial port  (default /dev/ttyUSB0)
  baud      int   baud rate    (default 115200)
  enabled   bool  set false to run without hardware (dry-run mode)
"""

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from motion_mapping import JOINT_NAMES


class HardwareInterfaceNode(Node):

    def __init__(self):
        super().__init__('hardware_interface')

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter('port',    '/dev/ttyUSB0')
        self.declare_parameter('baud',    115200)
        self.declare_parameter('enabled', True)

        port     = self.get_parameter('port').value
        baud     = self.get_parameter('baud').value
        enabled  = self.get_parameter('enabled').value

        # ── serial connection ─────────────────────────────────────────────────
        self._ser = None
        if enabled:
            try:
                import serial
                self._ser = serial.Serial(port, baud, timeout=1)
                self.get_logger().info(f'Serial open: {port} @ {baud}')
            except Exception as e:
                self.get_logger().warn(
                    f'Serial failed ({e}) — running in dry-run mode')
        else:
            self.get_logger().info('Hardware disabled — dry-run mode')

        # ── subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            JointTrajectory,
            '/robot_arm_controller/joint_trajectory',
            self._on_trajectory, 10,
        )

        self.get_logger().info('HardwareInterfaceNode ready.')

    # ── trajectory callback ───────────────────────────────────────────────────

    def _on_trajectory(self, msg: JointTrajectory):
        if not msg.points:
            return

        positions = msg.points[0].positions
        name_map  = dict(zip(msg.joint_names, positions))

        angles = [name_map.get(j, 0.0) for j in JOINT_NAMES]
        parts  = ','.join(
            f'{j[-2:]}:{a:.4f}' for j, a in zip(JOINT_NAMES, angles)
        )
        cmd = f'J:{parts}\n'
        self._send(cmd)

    # ── serial send ──────────────────────────────────────────────────────────

    def _send(self, cmd: str):
        self.get_logger().debug(f'TX: {cmd.strip()}')
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(cmd.encode())
            except Exception as e:
                self.get_logger().warn(f'Serial write error: {e}')

    def destroy_node(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = HardwareInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
