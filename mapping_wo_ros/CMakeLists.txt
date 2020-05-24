cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

project(mapping)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS}) 

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_DEFINITIONS}
)

# add_executable(mapping src/mapping.cpp) 
# target_link_libraries(mapping ${PCL_LIBRARIES} ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS})  

add_executable(test src/test.cpp) 
target_link_libraries(test ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS})  
