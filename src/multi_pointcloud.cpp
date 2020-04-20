#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/UInt16.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include "pointcloud_fusion/util.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

int global_flag = 1;
std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds;

void callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc,
              const sensor_msgs::ImageConstPtr &msg_img)
{
    if (global_flag == 0)
        return;
    else if (global_flag == 1)
    {
        ROS_INFO("ADDing new point cloud");

        pcl::PointCloud<pcl::PointXYZI> point_cloud_livox;
        pcl::PointCloud<pcl::PointXYZRGB> point_cloud_color;
        pcl::fromROSMsg(*msg_pc, point_cloud_livox);

        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);

        point_cloud_color = paintPointCloud(point_cloud_livox, cv_ptr->image);
        clouds.push_back(point_cloud_color);
        pcl::io::savePCDFile("/home/tim/pc.pcd", point_cloud_color);
        global_flag = 0;
    }
    else if (global_flag == 2)
    {
        pcl::PointCloud<pcl::PointXYZRGB> origin = clouds[0];
        ROS_INFO("Fusing point clouds");
        for (int i = 1; i < clouds.size(); i++)
        {
            pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
            icp.setInputTarget(origin.makeShared());
            icp.setInputSource(clouds[i].makeShared());
            // Set the max correspondence distance to 1cm (e.g., correspondences with higher distances will be ignored)
            icp.setMaxCorrespondenceDistance(0.01);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations(100);
            pcl::PointCloud<pcl::PointXYZRGB> Final;
            icp.align(Final);
            ROS_INFO_STREAM("ICP has converged?: " << icp.hasConverged());
            std::cout << "Final Transformation: " << std::endl
                      << icp.getFinalTransformation() << std::endl;
            pcl::PointCloud<pcl::PointXYZRGB> new_cloud;
            pcl::transformPointCloud(clouds[i], new_cloud, icp.getFinalTransformation());
            origin += new_cloud;
        }
        pcl::io::savePCDFile("/home/tim/result_icp.pcd", origin);
        ROS_INFO("Fusion Complete!!");
        ros::shutdown();
    }

    // ros::shutdown();
}

void KeyBoardCallBack(const std_msgs::UInt16 &msg)
{
    if (msg.data == '1')
    {
        global_flag = 1;
        ROS_INFO("Receive ADD point cloud command!");
    }
    else if (msg.data == '2')
    {
        global_flag = 2;
        ROS_INFO("Receive FUSE point cloud command!");
    }
}

int main(int argc, char **argv)
{
    readConfig();
    ros::init(argc, argv, "pointcloud_fusion");

    ros::NodeHandle n;

    std::string lidar_topic, camera_topic;
    n.getParam("/pointcloud_fusion/lidar_topic", lidar_topic);
    n.getParam("/pointcloud_fusion/camera_topic", camera_topic);

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(
        n, lidar_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> camera_sub(
        n, camera_topic, 1);
    ros::Subscriber key = n.subscribe("/keyboard/key", 1, KeyBoardCallBack);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, camera_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();

    return EXIT_SUCCESS;
}
