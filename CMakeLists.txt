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

add_library(datalog_lib src/mylog.cpp)

add_executable(fuse_point_cloud src/single_color_pointcloud.cpp) 
target_link_libraries(fuse_point_cloud ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})  
# target_link_libraries(find_transform)

add_executable(dataloger src/dataloger.cpp) 
target_link_libraries(dataloger datalog_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})  

add_executable(multi_pointcloud src/multi_color_pointcloud.cpp) 
target_link_libraries(multi_pointcloud ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})  
