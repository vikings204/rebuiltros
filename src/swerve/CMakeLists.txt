cmake_minimum_required(VERSION 3.8)
project(swerve)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# deps
include(/home/team204/ros2_ws/src/commondeps.cmake)

add_executable(swerve src/swerve.cpp src/swerve_module.cpp src/swerve_node.cpp)
target_include_directories(swerve PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(swerve PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(swerve rclcpp rclcpp_action std_msgs vikings_msgs)
target_link_libraries(swerve revlib wpilibc hal phoenix6)

install(TARGETS swerve
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
