cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

project(mapping)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS}) 

find_package(teaserpp REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_DEFINITIONS}
)

set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )



# add_executable(mapping src/mapping.cpp) 
# target_link_libraries(mapping ${PCL_LIBRARIES} ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS})  

add_executable(test src/test.cpp) 
target_link_libraries(test ${PCL_LIBRARIES} ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS} ${DBoW3_LIBS})  

# add_executable(teaser_cpp_ply src/test.cpp)
# target_link_libraries(teaser_cpp_ply teaserpp::teaser_registration teaserpp::teaser_io ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS} ${PCL_LIBRARIES})
