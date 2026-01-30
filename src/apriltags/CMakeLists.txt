cmake_minimum_required(VERSION 3.8)
project(apriltags LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# deps
include(/home/team204/ros2_ws/src/commondeps.cmake)

# libcuapriltags
include_directories("/home/team204/ros2_ws/src/apriltags/lib/cuapriltags/cuapriltags/")
add_library(libcuapriltags STATIC IMPORTED)
set_property(TARGET libcuapriltags PROPERTY IMPORTED_LOCATION "/home/team204/ros2_ws/src/apriltags/lib/cuapriltags/lib_aarch64_jetpack51/libcuapriltags.a")

find_package(CUDAToolkit REQUIRED)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(sample_node src/sample_node.cpp)
target_include_directories(sample_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(sample_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(sample_node rclcpp std_msgs geometry_msgs)
target_link_libraries(sample_node ${CUDA_LIBRARIES} ${OpenCV_LIBS} libcuapriltags)

add_executable(ntsender src/ntsender.cpp)
target_link_libraries(ntsender ${CUDA_LIBRARIES} ${OpenCV_LIBS} libcuapriltags wpilibc apriltag)

# frc971 detector testing
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD 20)
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 87)
#endif()
#if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
#  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
#  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
#endif()
#set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
#add_compile_options(-Wno-pedantic)
#add_executable(newdetector
#        src/newdetector.cpp
#        src/frc971/orin/971apriltag.cu
#        src/frc971/orin/apriltag_detect.cu
#        src/frc971/orin/labeling_allegretti_2019_BKE.cu
#        src/frc971/orin/line_fit_filter.cu
#        src/frc971/orin/points.cu
#        src/frc971/orin/threshold.cu
#        src/frc971/orin/cuda.cc
#)
#target_link_libraries(newdetector ${CUDA_LIBRARIES} ${OpenCV_LIBS} wpiutil ntcore "/home/team204/ros2_ws/src/apriltags/src/third_party/apriltag/build/libapriltag.a")
#target_include_directories(newdetector PRIVATE "/home/team204/ros2_ws/src/apriltags/src/")

install(TARGETS sample_node ntsender #newdetector
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
