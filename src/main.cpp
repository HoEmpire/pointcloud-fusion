#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/UInt16.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include "pointcloud_fusion/data_io.h"
#include "pointcloud_fusion/icp.h"
#include "pointcloud_fusion/optimization.h"
#include "pointcloud_fusion/point_cloud_registraion.h"
#include "pointcloud_fusion/type.h"
#include "pointcloud_fusion/util.h"
#include "pointcloud_fusion/visual_odometry.h"

using namespace std;
using namespace Eigen;

int cunt = 0;
int global_flag = 0;
string save_path;
vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
vector<cv::Mat> imgs;
vector<cv::Mat> depths;

Eigen::Matrix4f final_T;

void callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc, const sensor_msgs::ImageConstPtr &msg_img)
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

    cv::Mat image_undistorted;
    undistortImage(cv_ptr->image, image_undistorted);
    imgs.push_back(image_undistorted);

    cv::Mat depth_map;
    // paintPointCloud(point_cloud_livox, image_undistorted, point_cloud_color, depth_map);
    // depths.push_back(depth_map);

    // point_cloud_color.resize(point_cloud_livox.size());
    for (pcl::PointCloud<pcl::PointXYZI>::iterator pt = point_cloud_livox.points.begin();
         pt < point_cloud_livox.points.end(); pt++)
    {
      pcl::PointXYZRGB new_point;
      new_point.x = pt->x;
      new_point.y = pt->y;
      new_point.z = pt->z;
      point_cloud_color.push_back(new_point);
    }
    clouds.push_back(point_cloud_color.makeShared());

    global_flag = 0;
  }
  else if (global_flag == 2)  // register point clouds and mapping
  {
    struct imageType image_data;
    image_data.imgs = imgs;
    // image_data.depths = depths;
    struct pointcloudType pc_data(clouds);
    pointCloudRegistration(image_data, pc_data, config.pc_save_path);
    ROS_INFO("Fusion Complete!!");
    ros::shutdown();
  }
  else if (global_flag == 3)
  {
    ROS_INFO_STREAM("Logging No." << cunt << " data...");
    pcl::PointCloud<pcl::PointXYZI> point_cloud_livox;
    pcl::PointCloud<pcl::PointXYZRGB> point_cloud_color;
    pcl::fromROSMsg(*msg_pc, point_cloud_livox);

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);

    cv::Mat image_undistorted;
    undistortImage(cv_ptr->image, image_undistorted);

    cv::Mat depth_map;
    paintPointCloud(point_cloud_livox, image_undistorted, point_cloud_color, depth_map);  // TODO low in efficiency

    pcl::io::savePCDFile(save_path + "/" + to_string(cunt) + ".pcd", point_cloud_color);
    cv::imwrite(save_path + "/" + to_string(cunt) + ".jpg", image_undistorted);
    cv::imwrite(save_path + "/" + to_string(cunt) + ".png", depth_map);
    cunt++;
    global_flag = 0;
  }
}

void keyBoardCallBack(const std_msgs::UInt16 &msg)
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
  else if (msg.data == '3')
  {
    global_flag = 3;
    ROS_INFO("Receive logging data command!");
  }
  else if (msg.data == '4')
  {
    ROS_INFO_STREAM("Logging over! Shutting down...");
    ofstream outfile(save_path + "/description.txt", ios_base::trunc);

    outfile << cunt;
    outfile.close();
    ros::shutdown();
  }
}

int main(int argc, char **argv)
{
  readConfig();
  final_T.setIdentity(4, 4);
  ros::init(argc, argv, "pointcloud_fusion");

  ros::NodeHandle n;
  string lidar_topic, camera_topic;
  n.getParam("/pointcloud_fusion/save_path", save_path);
  n.getParam("/pointcloud_fusion/lidar_topic", lidar_topic);
  n.getParam("/pointcloud_fusion/camera_topic", camera_topic);
  loadConfig(n);
  // n.getParam("/pointcloud_fusion/icp_config/MaxCorrespondenceDistance", icp_configs.MaxCorrespondenceDistance);
  // n.getParam("/pointcloud_fusion/icp_config/TransformationEpsilon", icp_configs.TransformationEpsilon);
  // n.getParam("/pointcloud_fusion/icp_config/MaximumIterations", icp_configs.MaximumIterations);
  // n.getParam("/pointcloud_fusion/icp_config/EuclideanFitnessEpsilon", icp_configs.EuclideanFitnessEpsilon);
  // n.getParam("/pointcloud_fusion/icp_config/RANSACOutlierRejectionThreshold",
  //            icp_configs.RANSACOutlierRejectionThreshold);

  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(n, lidar_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> camera_sub(n, camera_topic, 1);
  ros::Subscriber key = n.subscribe("/keyboard/key", 1, keyBoardCallBack);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, camera_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return EXIT_SUCCESS;
}
