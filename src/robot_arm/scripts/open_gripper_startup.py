#!/usr/bin/env python3
"""
open_gripper_startup.py — park the gripper OPEN once everything is up.

Why this exists as its own step
--------------------------------
The gripper must rest open, and three earlier attempts to arrange that did
not survive contact with reality:

  * ros2_control's `initial_value` (0.011 on both jaw joints) is present in
    the URDF Gazebo actually loads, and the joints still come up at 0 --
    gz_ros2_control does not drive the joint there.
  * load_controllers._send_home() reports success and the joints do not move,
    even sending the same two-point trajectory shape that works elsewhere.
  * robot_node's startup assertion fires (confirmed in /rosout) but lands too
    early to take effect.

What does work, measured repeatedly, is a standalone publisher that waits for
the controller to subscribe, reads real positions from /joint_states, and
sends current-then-target. That is precisely what this does, late enough that
nothing is still initialising.

The two points matter: joint_trajectory_controller silently ignores a
trajectory whose single point already matches the arm joints, even when the
gripper differs. Sending only an endpoint is why gripper-only commands were
dropped.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

TOPIC = '/robot_arm_controller/joint_trajectory'
JOINTS = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4',
          'joint_L5', 'joint_L6', 'joint_L7_R', 'joint_L7_L']
GRIPPER = ('joint_L7_R', 'joint_L7_L')
OPEN_M = 0.011

# How far to swing L1 so the gripper command travels with it. Small enough to
# be unobtrusive, large enough that the controller treats it as real motion.
NUDGE_RAD = 0.05


def main():
    rclpy.init()
    node = Node('open_gripper_startup')
    pub = node.create_publisher(JointTrajectory, TOPIC, 10)

    now = {}
    node.create_subscription(
        JointState, '/joint_states',
        lambda m: now.update(dict(zip(m.name, m.position))), 10)

    # Wait for the controller to subscribe AND for real joint feedback.
    # Publishing before discovery completes loses the message outright.
    deadline = time.time() + 60.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() > 0 and 'joint_L7_R' in now:
            break
    if pub.get_subscription_count() == 0 or 'joint_L7_R' not in now:
        print('[open_gripper] Gave up waiting for controller / joint states',
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    def send(l1_target, label):
        """One trajectory: arm to l1_target, gripper open."""
        msg = JointTrajectory()
        msg.joint_names = list(JOINTS)

        start = JointTrajectoryPoint()
        start.positions = [float(now.get(j, 0.0)) for j in JOINTS]
        start.time_from_start = Duration(sec=0, nanosec=10_000_000)

        end = JointTrajectoryPoint()
        pos = [0.0] * len(JOINTS)
        pos[JOINTS.index('joint_L1')] = l1_target
        for g in GRIPPER:
            pos[JOINTS.index(g)] = OPEN_M
        end.positions = pos
        end.time_from_start = Duration(sec=2)

        msg.points = [start, end]
        pub.publish(msg)
        stop = time.time() + 3.0
        while time.time() < stop:
            rclpy.spin_once(node, timeout_sec=0.05)

    # The gripper command has to ride along with real arm motion.
    # joint_trajectory_controller does not act on a trajectory unless some ARM
    # joint actually changes -- measured directly: "arm already at 0, gripper
    # 0->0.011" never moved, at 40ms or 2s, with one point or two, while the
    # identical gripper change alongside arm motion moved both jaws every
    # time. So nudge L1 a little, carrying the gripper open with it, then
    # bring L1 back to 0 while holding the gripper open. Both legs move an arm
    # joint, so both execute, and the arm finishes where it started.
    for attempt in range(1, 4):
        send(NUDGE_RAD, 'nudge out')
        send(0.0, 'return')

        r, l = now.get('joint_L7_R', 0.0), now.get('joint_L7_L', 0.0)
        if abs(r - OPEN_M) < 0.002 and abs(l - OPEN_M) < 0.002:
            print(f'[open_gripper] Gripper open: R={r:.5f} L={l:.5f} '
                  f'(attempt {attempt})')
            node.destroy_node()
            rclpy.shutdown()
            return 0
        print(f'[open_gripper] attempt {attempt}: R={r:.5f} L={l:.5f}, retrying')

    print(f'[open_gripper] Could not open the gripper after 5 attempts',
          file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()
    return 1


if __name__ == '__main__':
    sys.exit(main())
