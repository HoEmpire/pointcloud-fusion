#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include "pointcloud_fusion/util.h"

void callback(const sensor_msgs::PointCloud2ConstPtr &msg_pc,
              const sensor_msgs::ImageConstPtr &msg_img)
{
  ROS_INFO_STREAM("Lidar received at " << msg_pc->header.stamp.toSec());
  ROS_INFO_STREAM("Camera received at " << msg_img->header.stamp.toSec());

  pcl::PointCloud<pcl::PointXYZI> point_cloud_livox;
  pcl::PointCloud<pcl::PointXYZRGB> point_cloud_color;
  pcl::fromROSMsg(*msg_pc, point_cloud_livox);

  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);

  point_cloud_color = paintPointCloud(point_cloud_livox, cv_ptr->image);

  pcl::io::savePCDFile("/home/tim/color_point_cloud.pcd", point_cloud_color);
  // ros::shutdown();
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

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, camera_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return EXIT_SUCCESS;
}
