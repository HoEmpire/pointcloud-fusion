#include <iostream>

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include "pointcloud_fusion/util.h"
#include "pointcloud_fusion/mylog.h"

#include <ros/package.h>

mylog logger;
void callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc,
              const sensor_msgs::ImageConstPtr &msg_img)
{
    ROS_INFO_STREAM("Lidar received at " << msg_pc->header.stamp.toSec());
    ROS_INFO_STREAM("Camera received at " << msg_img->header.stamp.toSec());
    logger.pc_array.push_back(msg_pc);
    logger.img_array.push_back(msg_img);
}

int main(int argc, char **argv)
{
    readConfig();
    ros::init(argc, argv, "pointcloud_fusion");

    ros::NodeHandle n;

    std::string pkg_loc = ros::package::getPath("pointcloud_fusion");

    std::string lidar_topic, camera_topic, save_path;
    n.getParam("/pointcloud_fusion/lidar_topic", lidar_topic);
    n.getParam("/pointcloud_fusion/camera_topic", camera_topic);
    n.getParam("/pointcloud_fusion/save_path", save_path);
    logger.save_path = pkg_loc + save_path;
    ROS_INFO_STREAM("Save Path: " << logger.save_path);

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(
        n, lidar_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> camera_sub(
        n, camera_topic, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, camera_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    logger.save();
    return EXIT_SUCCESS;
}
