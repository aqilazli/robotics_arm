"""
rviz.launch.py
=================
Opens RViz with MoveIt's MotionPlanning panel -- this is the manual
control interface: drag the interactive marker on the arm's end (L5) to
where you want it, hit "Plan", check the preview, hit "Execute".

Run gazebo.launch.py and move_group.launch.py FIRST, then this.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_share = get_package_share_directory('robot_arm')
    robot_arm_share = get_package_share_directory('robot_arm')

    urdf_file = os.path.join(robot_arm_share, 'urdf', 'robot_arm.urdf')
    with open(urdf_file) as f:
        robot_description_xml = f.read()
    robot_description_xml = robot_description_xml.replace('$(find robot_arm)', robot_arm_share)
    robot_description = {'robot_description': robot_description_xml}

    srdf_file = os.path.join(moveit_share, 'config', 'robot_arm.srdf')
    with open(srdf_file) as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    import yaml
    def load_yaml(path):
        with open(os.path.join(moveit_share, path)) as f:
            return yaml.safe_load(f)

    kinematics_yaml = load_yaml('config/kinematics.yaml')

    rviz_config = os.path.join(moveit_share, 'config', 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            {'robot_description_kinematics': kinematics_yaml},
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription([rviz_node])
