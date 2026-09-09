import os
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, ExecuteProcess,
                             TimerAction, SetEnvironmentVariable,
                             DeclareLaunchArgument)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share      = get_package_share_directory('robot_arm')
    ros_gz_share   = get_package_share_directory('ros_gz_sim')
    ign_ctrl_prefix = get_package_prefix('ign_ros2_control')

    mesh_resource_path  = os.path.dirname(pkg_share)                   # …/share/
    ign_plugin_path     = os.path.join(ign_ctrl_prefix, 'lib')

    # ── Resolve URDF ─────────────────────────────────────────────────────────
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot_arm.urdf')
    with open(urdf_file) as f:
        robot_description = f.read()
    robot_description = robot_description.replace('$(find robot_arm)', pkg_share)
    if robot_description.lstrip().startswith('<?xml'):
        robot_description = robot_description[robot_description.index('?>') + 2:].lstrip()

    resolved_urdf = '/tmp/robot_arm_resolved.urdf'
    with open(resolved_urdf, 'w') as f:
        f.write(robot_description)

    controllers_yaml = os.path.join(pkg_share, 'config', 'ros2_controllers.yaml')
    load_ctrl_script = os.path.join(
        pkg_share, '..', '..', 'lib', 'robot_arm', 'load_controllers.py')
    # Two RViz configs: the default has no MoveIt panel, because this launch
    # file does not start move_group, and MoveIt's MotionPlanning display
    # errors out for 10s looking for an SRDF that will never arrive. The
    # MoveIt launcher (Desktop "MoveIt Control") passes the other one, which
    # does include the panel, so both paths get a single RViz window with
    # exactly the displays that work for them.
    #   ros2 launch robot_arm gazebo.launch.py rviz_config:=robot_arm_moveit.rviz
    rviz_config_name = LaunchConfiguration('rviz_config')
    rviz_config = PathJoinSubstitution([pkg_share, 'config', rviz_config_name])

    # ── Actions ──────────────────────────────────────────────────────────────

    # Pass mesh + plugin paths to ALL child processes via launch env vars
    set_resource_path = SetEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH', mesh_resource_path)
    set_plugin_path = SetEnvironmentVariable(
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ign_plugin_path)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
    )

    spawn_script = os.path.join(
        pkg_share, '..', '..', 'lib', 'robot_arm', 'spawn_robot.py')

    spawn_entity = ExecuteProcess(
        cmd=['python3', spawn_script, resolved_urdf],
        output='screen',
    )

    load_controllers = ExecuteProcess(
        cmd=['python3', load_ctrl_script, controllers_yaml],
        output='screen',
    )

    # Park the gripper open once everything is up. A separate late step
    # because neither ros2_control's initial_value nor load_controllers' home
    # actually moves these joints; see open_gripper_startup.py.
    open_gripper_script = os.path.join(
        pkg_share, '..', '..', 'lib', 'robot_arm', 'open_gripper_startup.py')
    open_gripper = ExecuteProcess(
        cmd=['python3', open_gripper_script],
        output='screen',
    )

    # RViz needs the SRDF and kinematics, not just the URDF. Without them
    # MoveIt's MotionPlanning panel comes up with an empty Joints tab and no
    # way to plan: robot_description_semantic reads back "Parameter not set".
    # Harmless for the plain (non-MoveIt) config, which simply ignores them.
    srdf_file = os.path.join(pkg_share, 'config', 'robot_arm.srdf')
    with open(srdf_file) as f:
        robot_description_semantic = f.read()
    with open(os.path.join(pkg_share, 'config', 'kinematics.yaml')) as f:
        kinematics_yaml = yaml.safe_load(f)

    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_description},
            {'robot_description_semantic': robot_description_semantic},
            {'robot_description_kinematics': kinematics_yaml},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config', default_value='robot_arm.rviz',
            description='RViz config in robot_arm/config/. Use '
                        'robot_arm_moveit.rviz when move_group is also running.'),
        set_resource_path,
        set_plugin_path,
        gazebo,
        robot_state_publisher,
        clock_bridge,
        spawn_entity,                                    # polls until Gazebo ready
        TimerAction(period=3.0, actions=[load_controllers]),
        TimerAction(period=3.0, actions=[rviz]),
        TimerAction(period=12.0, actions=[open_gripper]),
    ])
