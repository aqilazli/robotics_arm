#!/usr/bin/env python3
"""
spawn_robot.py — waits for Gazebo to be ready, then spawns the robot.
Called by gazebo.launch.py as an ExecuteProcess.

Usage: python3 spawn_robot.py <urdf_path>
"""
import subprocess
import sys
import time

WORLD = 'empty'

def ign_services():
    r = subprocess.run(['ign', 'service', '--list'],
                       capture_output=True, text=True)
    return r.stdout

def wait_for_gazebo(timeout=120):
    target = f'/world/{WORLD}/create'
    print(f'[spawn] Waiting for Gazebo world "{WORLD}"...')
    for i in range(timeout):
        if target in ign_services():
            print(f'[spawn] Gazebo ready ({i}s)')
            return True
        time.sleep(1)
        if i % 10 == 9:
            print(f'[spawn] Still waiting... ({i+1}s)')
    print('[spawn] ERROR: Gazebo never became ready.')
    return False

def spawn(urdf_path, retries=5):
    for attempt in range(1, retries + 1):
        print(f'[spawn] Attempt {attempt}: spawning robot_arm ...')
        r = subprocess.run([
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', WORLD,
            '-file',  urdf_path,
            '-name',  'robot_arm',
            # z=0, not 0.1. base_link.STL's lowest point is exactly at its
            # link origin (z range 0.0000 to 0.0705) and base_joint fixes that
            # origin to world z=0, so any spawn offset lifts the whole robot
            # clear of the ground -- it visibly floated above its own shadow.
            '-x', '0', '-y', '0', '-z', '0',
        ], capture_output=True, text=True)
        # ros_gz_sim reports "OK creation of entity" as a ROS [INFO] line,
        # which goes to stderr, not stdout. Checking stdout alone made every
        # successful spawn look like a failure, so this retried until the
        # retries ran out, spawning the robot repeatedly and then exiting
        # non-zero and taking the rest of the launch down with it.
        combined = (r.stdout or '') + (r.stderr or '')
        print(combined.strip())
        if r.returncode == 0 and 'OK creation of entity' in combined:
            print('[spawn] Robot spawned successfully ✓')
            return True
        print(f'[spawn] Failed (rc={r.returncode}) — retrying in 3s...')
        time.sleep(3)
    print('[spawn] ERROR: could not spawn robot after all retries.')
    return False

if __name__ == '__main__':
    urdf_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/robot_arm_resolved.urdf'
    if not wait_for_gazebo():
        sys.exit(1)
    time.sleep(1)   # small margin after service appears
    ok = spawn(urdf_path)
    sys.exit(0 if ok else 1)
