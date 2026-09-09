"""
move_group.launch.py
======================
Starts MoveIt's move_group node ONLY -- assumes the robot is already
running (robot_state_publisher, robot_arm_controller, joint_state_broadcaster),
same as launched by:
    ros2 launch robot_arm gazebo.launch.py

Run that FIRST, then this, then rviz.launch.py in a third terminal.

move_group plans motions and executes them through robot_arm_controller's
FollowJointTrajectory action -- the SAME controller the vision
teleoperation pipeline (robot_node.py) drives. Don't run both control
paths at once; they'd both try to command the same joints.
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_share = get_package_share_directory(package_name)
    full_path = os.path.join(package_share, file_path)
    with open(full_path) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    moveit_share = get_package_share_directory('robot_arm')
    robot_arm_share = get_package_share_directory('robot_arm')

    # ── robot_description -- same loading + $(find robot_arm) resolution
    #    gazebo.launch.py uses, so MoveIt's model matches what's spawned ──
    urdf_file = os.path.join(robot_arm_share, 'urdf', 'robot_arm.urdf')
    with open(urdf_file) as f:
        robot_description_xml = f.read()
    robot_description_xml = robot_description_xml.replace('$(find robot_arm)', robot_arm_share)
    robot_description = {'robot_description': robot_description_xml}

    # ── robot_description_semantic (the SRDF) ────────────────────────────
    srdf_file = os.path.join(moveit_share, 'config', 'robot_arm.srdf')
    with open(srdf_file) as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    kinematics_yaml = load_yaml('robot_arm', 'config/kinematics.yaml')
    joint_limits_yaml = load_yaml('robot_arm', 'config/joint_limits.yaml')
    ompl_yaml = load_yaml('robot_arm', 'config/ompl_planning.yaml')
    controllers_yaml = load_yaml('robot_arm', 'config/moveit_controllers.yaml')

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            {'robot_description_kinematics': kinematics_yaml},
            {'robot_description_planning': joint_limits_yaml},
            {'planning_pipelines': ['ompl']},
            {'ompl': ompl_yaml},
            {'ompl.planning_plugin': 'ompl_interface/OMPLPlanner'},
            controllers_yaml,
            {'use_sim_time': True},
            {'moveit_manage_controllers': True},
            {'publish_robot_description_semantic': True},
        ],
    )

    return LaunchDescription([move_group_node])
