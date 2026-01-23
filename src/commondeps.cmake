cmake_minimum_required(VERSION 3.8)
set(FRC_YEAR "2025")

# ros dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(vikings_msgs REQUIRED)

# regular libs
find_package(cppzmq)
add_library(msgpack INTERFACE)
target_include_directories(msgpack INTERFACE "/home/team204/thirdparty/cppack/msgpack/include/")

# wpilib
set(CMAKE_PREFIX_PATH "/home/team204/thirdparty/${FRC_YEAR}/wpilib")
find_package(wpilibc CONFIG REQUIRED)
find_package(hal CONFIG REQUIRED)

# vendordeps, some related to each other

# revlib driver
include_directories("/home/team204/thirdparty/${FRC_YEAR}/revlib-driver/include/")
add_library(revlib-driver SHARED IMPORTED)
set_property(TARGET revlib-driver PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/${FRC_YEAR}/revlib-driver/lib/linux/arm64/shared/libREVLibDriver.so")
# revlib
include_directories("/home/team204/thirdparty/${FRC_YEAR}/revlib/include/")
add_library(revlib SHARED IMPORTED)
set_property(TARGET revlib PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/${FRC_YEAR}/revlib/lib/linux/arm64/shared/libREVLib.so")
target_link_libraries(revlib INTERFACE wpilibc)
target_link_libraries(revlib INTERFACE revlib-driver)

# phoenix6 tools
include_directories("/home/team204/thirdparty/${FRC_YEAR}/phoenix6-tools/include/")
add_library(phoenix6-tools SHARED IMPORTED)
set_property(TARGET phoenix6-tools PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/${FRC_YEAR}/phoenix6-tools/lib/linux/arm64/shared/libCTRE_PhoenixTools.so")
# phoenix6 wpiapi
include_directories("/home/team204/thirdparty/${FRC_YEAR}/phoenix6/include/")
add_library(phoenix6 SHARED IMPORTED)
set_property(TARGET phoenix6 PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/${FRC_YEAR}/phoenix6/lib/linux/arm64/shared/libCTRE_Phoenix6_WPI.so")
target_link_libraries(phoenix6 INTERFACE wpilibc)
target_link_libraries(phoenix6 INTERFACE phoenix6-tools)

#message(NOTICE "dependencies have been loaded from commondeps.cmake")