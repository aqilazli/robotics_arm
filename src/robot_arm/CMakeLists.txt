cmake_minimum_required(VERSION 3.5)
project(robot_arm)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch meshes config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/motion_mapping.py
  scripts/load_controllers.py
  scripts/spawn_robot.py
  scripts/open_gripper_startup.py
  scripts/arm_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts/perception
  scripts/data_processing
  scripts/models
  scripts/evaluation
  scripts/data_collection
  scripts/robot_control
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
