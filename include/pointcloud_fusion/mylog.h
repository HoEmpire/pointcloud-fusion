#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

class mylog
{
public:
    mylog();

    ~mylog();

    void save();

    std::vector<sensor_msgs::ImageConstPtr> img_array;
    std::vector<sensor_msgs::PointCloud2ConstPtr> pc_array;

    std::string save_path;
    std::ofstream outfile_pointcloud;
    std::ofstream outfile_img_time;
};
