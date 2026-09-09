"""
motion_control.launch.py
========================
Launches the full hand-to-robot-arm pipeline:

  depth_provider   →  /arm/depth
  pose_detector    →  /arm/landmarks
  arm_controller   →  /robot_arm_controller/joint_trajectory

Run the Gazebo sim separately first:
  ros2 launch robot_arm gazebo.launch.py

Then start this launch file:
  ros2 launch robot_arm motion_control.launch.py

To disable depth fusion (RGB-only mode):
  ros2 launch robot_arm motion_control.launch.py use_depth:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── launch arguments ─────────────────────────────────────────────────────
    use_depth_arg = DeclareLaunchArgument(
        'use_depth', default_value='true',
        description='Fuse Orbbec depth into landmark Z coordinates',
    )
    smoothing_arg = DeclareLaunchArgument(
        'smoothing', default_value='0.3',
        description='EMA smoothing factor [0=none, 1=frozen]',
    )
    camera_index_arg = DeclareLaunchArgument(
        'camera_index', default_value='-1',
        description='Color camera /dev/videoN index (-1 = auto-detect Astra Pro HD)',
    )
    show_window_arg = DeclareLaunchArgument(
        'show_window', default_value='true',
        description='Show live camera overlay window (hand tracking + joint bars)',
    )

    use_depth    = LaunchConfiguration('use_depth')
    smoothing    = LaunchConfiguration('smoothing')
    camera_index = LaunchConfiguration('camera_index')
    show_window  = LaunchConfiguration('show_window')

    # ── nodes ────────────────────────────────────────────────────────────────

    depth_provider = Node(
        package='robot_arm',
        executable='depth_provider.py',
        name='depth_provider',
        output='screen',
        parameters=[{
            'fps':         30,
            'depth_max_mm': 3000,
        }],
    )

    pose_detector = Node(
        package='robot_arm',
        executable='pose_detector.py',
        name='pose_detector',
        output='screen',
        parameters=[{
            'camera_index': camera_index,
            'fps':          30,
            'show_window':  True,   # hardcoded bool — avoids string→bool conversion issue
        }],
    )

    arm_controller = Node(
        package='robot_arm',
        executable='arm_controller.py',
        name='arm_controller',
        output='screen',
        parameters=[{
            'use_depth':        use_depth,
            'smoothing':        smoothing,
            'trajectory_dt_ms': 100,
        }],
    )

    return LaunchDescription([
        use_depth_arg,
        smoothing_arg,
        camera_index_arg,
        show_window_arg,
        depth_provider,
        pose_detector,
        arm_controller,
    ])
