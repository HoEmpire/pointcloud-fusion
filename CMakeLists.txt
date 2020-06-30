cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

project(lidar_camera_calibration)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -O3")

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(catkin 
  REQUIRED COMPONENTS
  std_msgs
  rospy  
  roscpp
  cv_bridge
  image_transport
  image_geometry
  pcl_ros
)

# generate_messages(DEPENDENCIES
#                   std_msgs
#                   )

# catkin_package(CATKIN_DEPENDS std_msgs)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS}) 

include_directories(include ${catkin_INCLUDE_DIRS})

# g2o 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_DEFINITIONS}
)

set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

add_library(optimization src/optimization.cpp)
target_link_libraries(optimization ${Eigen3_INCLUDE_DIRS} g2o_core g2o_stuff g2o_types_slam3d)

add_executable(pointcloud_fusion src/main.cpp) 
target_link_libraries(pointcloud_fusion ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS} ${DBoW3_LIBS} optimization)  
