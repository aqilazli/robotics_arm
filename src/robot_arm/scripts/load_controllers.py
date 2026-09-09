#!/usr/bin/env python3
"""
load_controllers.py
====================
Waits until the Ignition hardware plugin has exported all joint interfaces,
then spawns and activates the controllers.  Called by gazebo.launch.py.

Uses ros2 service calls only (does NOT require ros2_control CLI package).

Usage:
  load_controllers.py <ros2_controllers_yaml_path>
"""

import subprocess
import sys
import time

# Must match robot_arm_controller's joint list in config/ros2_controllers.yaml
# exactly: joint_trajectory_controller rejects the WHOLE trajectory if the
# names differ, with "Joints on incoming trajectory don't match the controller
# joints" -- it does not partially apply. joint_L7 is the prismatic gripper.
JOINTS = ['joint_L1', 'joint_L2', 'joint_L3',
          'joint_L4', 'joint_L5', 'joint_L6',
          'joint_L7_R', 'joint_L7_L']

# Arm values match robot_node.py's HOME_POSE exactly: all-zeros, a fully
# straight arm -- was a folded reference pose (L1=0.2565, L2=0.2923,
# L3=-0.35, L5=0.2188), reconsidered directly: "why initail postion of
# gazebo robort startup like this? is houdl be straight". Kept in sync so
# the arm lands directly on that pose at boot instead of flashing through
# a different pose for the ~2s before robot_node's own startup assertion
# corrects it. The two gripper jaws rest OPEN (0.011 m of travel), which is
# the safe idle state for a gripper -- zero would park it clamped shut.
HOME = {j: 0.0 for j in JOINTS}
HOME['joint_L7_R'] = 0.011
HOME['joint_L7_L'] = 0.011


def sh(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def wait_for_controller_manager(timeout=60) -> bool:
    print('[load_controllers] Waiting for /controller_manager …')
    for _ in range(timeout):
        r = sh(['ros2', 'service', 'list'])
        if '/controller_manager/list_controllers' in r.stdout:
            print('[load_controllers] controller_manager found.')
            return True
        time.sleep(1)
    print('[load_controllers] ERROR: controller_manager never appeared.')
    return False


def wait_for_hardware(timeout=60) -> bool:
    """
    Poll via service call until all 6 joint interfaces appear.
    Uses ros2 service call instead of ros2 control (avoids needing ros2controlcli).
    """
    print('[load_controllers] Waiting for hardware interfaces …')
    for _ in range(timeout):
        r = sh([
            'ros2', 'service', 'call',
            '/controller_manager/list_hardware_interfaces',
            'controller_manager_msgs/srv/ListHardwareInterfaces', '{}'
        ])
        if all(j in r.stdout for j in JOINTS):
            print('[load_controllers] All hardware interfaces ready.')
            return True
        time.sleep(1)
    print('[load_controllers] WARNING: hardware interfaces did not appear — trying anyway.')
    return False


def list_controllers() -> str:
    """Return stdout of list_controllers service call."""
    r = sh([
        'ros2', 'service', 'call',
        '/controller_manager/list_controllers',
        'controller_manager_msgs/srv/ListControllers', '{}'
    ])
    return r.stdout


def unload_if_stuck(name: str):
    """Force-remove a controller in any non-active state."""
    out = list_controllers()
    if name not in out:
        return
    print(f'[load_controllers] {name} already exists — force removing …')
    # deactivate first (ignore errors)
    sh(['ros2', 'service', 'call',
        '/controller_manager/deactivate_controllers',
        'controller_manager_msgs/srv/SwitchController',
        f'{{deactivate_controllers: ["{name}"]}}'])
    time.sleep(0.5)
    # unload
    sh(['ros2', 'service', 'call',
        '/controller_manager/unload_controller',
        'controller_manager_msgs/srv/UnloadController',
        f'{{name: "{name}"}}'])
    time.sleep(1)


def spawn(name: str, ctrl_type: str, param_file: str = None, retries: int = 5) -> bool:
    """
    Spawn via controller_manager's own spawner, which already waits for the
    manager and handles load/configure/activate ordering.

    Deliberately does NOT call unload_if_stuck() first. Force-deactivating and
    unloading an existing controller before respawning was the cause of
    repeated startup failures: if anything went wrong mid-sequence it left
    joint_state_broadcaster deactivated and robot_arm_controller absent, i.e.
    a robot that publishes joint states, looks healthy, and silently discards
    every trajectory. Spawning an already-active controller is harmless, so
    there is nothing to gain by tearing it down first.
    """
    for attempt in range(1, retries + 1):
        cmd = [
            'ros2', 'run', 'controller_manager', 'spawner',
            name,
            '-c', '/controller_manager',
            '--controller-type', ctrl_type,
            '--controller-manager-timeout', '30',
        ]
        if param_file:
            cmd += ['--param-file', param_file]

        print(f'[load_controllers] Spawning {name} (attempt {attempt}) …')
        r = subprocess.run(cmd)
        if r.returncode == 0:
            print(f'[load_controllers] {name}: active ✓')
            return True
        print(f'[load_controllers] {name}: failed (code {r.returncode}), retrying in 3 s …')
        time.sleep(3)
    print(f'[load_controllers] ERROR: could not activate {name}.')
    return False


def main():
    yaml = sys.argv[1] if len(sys.argv) > 1 else None

    if not wait_for_controller_manager():
        sys.exit(1)

    # No hand-rolled hardware-interface poll here any more. It spawned a
    # "ros2 service call" subprocess per iteration, took minutes under any
    # contention (move_group starting alongside it, for instance), and the
    # spawner's own --controller-manager-timeout covers the same ground.
    time.sleep(2)

    ok1 = spawn(
        'joint_state_broadcaster',
        'joint_state_broadcaster/JointStateBroadcaster',
        param_file=yaml,
    )
    if not ok1:
        print('[load_controllers] joint_state_broadcaster failed — continuing.')

    time.sleep(1)

    ok2 = spawn(
        'robot_arm_controller',
        'joint_trajectory_controller/JointTrajectoryController',
        param_file=yaml,
    )
    if ok2:
        print('[load_controllers] robot_arm_controller active — arm ready ✓')
        time.sleep(1)
        # Used to also send its own HOME trajectory here (_send_home,
        # removed). robot_node.py's own _assert_startup_pose() sends home
        # too, ~20s later once it starts -- two independent, uncoordinated
        # publishers both targeting /robot_arm_controller/joint_trajectory
        # at different times during the same launch, each one discarding
        # the other's still-interpolating (2s) trajectory and restarting
        # it from wherever the arm had gotten to. That is what a live
        # capture of a completely idle relaunch (nobody at the camera,
        # zero pose/gripper messages the whole time) showed: joints
        # ramping toward home, yanked back, then ramping again a few
        # seconds later -- reported directly as "the robot is dancing and
        # jitter while no one there... also when relaunch the robot
        # dancing". robot_node.py's version is the one built to be aware
        # of freeze state and control mode, so it's kept as the sole
        # authority; this call now only verifies the controller came up,
        # same as it always did at the end of the removed function.
        verify_active()
    else:
        print('[load_controllers] ERROR: robot_arm_controller failed.')
        sys.exit(1)


def verify_active(name: str = 'robot_arm_controller') -> bool:
    """
    Confirm the controller really ended up active.

    Worth the extra service call: when this step is skipped and the spawn has
    quietly failed, the robot still publishes /joint_states from the
    broadcaster and looks entirely healthy, while every trajectory sent to it
    is dropped on the floor. That failure mode is indistinguishable from a
    stuck joint unless you go and check, and it cost a long debugging session.
    """
    out = list_controllers()
    ok = f"name='{name}'" in out and "state='active'" in out
    if ok:
        print(f'[load_controllers] verified {name} is active \u2713')
    else:
        print('')
        print('=' * 68)
        print(f'[load_controllers] ERROR: {name} is NOT active.')
        print('  The robot will publish /joint_states and look fine, but every')
        print('  trajectory will be silently ignored and nothing will move.')
        print('  Check with:')
        print('    ros2 service call /controller_manager/list_controllers \\')
        print('      controller_manager_msgs/srv/ListControllers "{}"')
        print('=' * 68)
        print('')
    return ok


if __name__ == '__main__':
    main()
