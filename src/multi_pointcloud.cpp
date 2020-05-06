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

#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <boost/make_shared.hpp>
#include "pointcloud_fusion/util.h"

// convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation<PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
  MyPointRepresentation()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray(const PointNormalT &p, float *out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

int cunt = 0;
int global_flag = 0;
std::string save_path;
std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds;
struct icp_config
{
  double MaxCorrespondenceDistance;
  double TransformationEpsilon;
  double MaximumIterations;
  double EuclideanFitnessEpsilon;
  double RANSACOutlierRejectionThreshold;
} icp_configs;

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

    point_cloud_color = paintPointCloud(point_cloud_livox, cv_ptr->image);
    clouds.push_back(point_cloud_color);
    pcl::io::savePCDFile("/home/tim/pc.pcd", point_cloud_color);
    global_flag = 0;
  }
  else if (global_flag == 2)
  {
    pcl::PointCloud<pcl::PointXYZRGB> origin = clouds[0];
    pcl::PointCloud<pcl::PointXYZRGB> wo_icp = clouds[0];
    ROS_INFO("Fusing point clouds");
    for (int i = 0; i < clouds.size() - 1; i++)
    {
      PointCloud::Ptr src(new PointCloud);
      PointCloud::Ptr tgt(new PointCloud);
      pcl::VoxelGrid<PointT> grid;

      grid.setLeafSize(0.01, 0.01, 0.01);
      grid.setInputCloud(clouds[i + 1].makeShared());
      grid.filter(*src);

      grid.setInputCloud(clouds[i].makeShared());
      grid.filter(*tgt);

      // Compute surface normals and curvature
      PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
      PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

      pcl::NormalEstimation<pcl::PointXYZRGB, PointNormalT> norm_est;
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
      norm_est.setSearchMethod(tree);
      norm_est.setKSearch(30);

      norm_est.setInputCloud(src);
      norm_est.compute(*points_with_normals_src);
      pcl::copyPointCloud(*src, *points_with_normals_src);

      norm_est.setInputCloud(tgt);
      norm_est.compute(*points_with_normals_tgt);
      pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

      // Instantiate our custom point representation (defined above) ...
      MyPointRepresentation point_representation;
      // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
      float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
      point_representation.setRescaleValues(alpha);

      pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> icp;
      icp.setInputSource(points_with_normals_src);
      icp.setInputTarget(points_with_normals_tgt);
      // icp.setInputTarget(clouds[i].makeShared());
      // icp.setInputSource(clouds[i + 1].makeShared());

      icp.setMaxCorrespondenceDistance(icp_configs.MaxCorrespondenceDistance);  // 0.10
      icp.setTransformationEpsilon(icp_configs.TransformationEpsilon);          // 1e-10
      // icp.setEuclideanFitnessEpsilon(icp_configs.EuclideanFitnessEpsilon);                  // 0.001
      // icp.setMaximumIterations(icp_configs.MaximumIterations);                              // 100
      // icp.setRANSACOutlierRejectionThreshold(icp_configs.RANSACOutlierRejectionThreshold);  // 1.5

      icp.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
      /*
      pcl::PointCloud<PointNormalT> Final;
      icp.align(Final);
      */
      // Run the same optimization in a loop and visualize the results

      Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
      PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
      icp.setMaximumIterations(2);
      for (int i = 0; i < 100; ++i)
      {
        PCL_INFO("Iteration Nr. %d.\n", i);

        // save cloud for visualization purpose
        points_with_normals_src = reg_result;

        // Estimate
        icp.setInputSource(points_with_normals_src);
        icp.align(*reg_result);

        // accumulate transformation between each Iteration
        Ti = icp.getFinalTransformation() * Ti;

        // if the difference between this transformation and the previous one
        // is smaller than the threshold, refine the process by reducing
        // the maximal correspondence distance
        if (std::abs((icp.getLastIncrementalTransformation() - prev).sum()) < icp.getTransformationEpsilon())
          icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() - 0.001);

        prev = icp.getLastIncrementalTransformation();
      }

      ROS_INFO_STREAM("ICP has converged?: " << icp.hasConverged());
      ROS_INFO_STREAM("Fitness Score: " << icp.getFitnessScore());
      // final_T = final_T * icp.getFinalTransformation();
      final_T = final_T * Ti;
      std::cout << "Final Transformation: " << std::endl << final_T << std::endl;
      pcl::PointCloud<pcl::PointXYZRGB> new_cloud;

      pcl::transformPointCloud(clouds[i + 1], new_cloud, final_T);
      origin += new_cloud;
      wo_icp += clouds[i + 1];
    }
    pcl::io::savePCDFile("/home/tim/result_icp.pcd", origin);
    pcl::io::savePCDFile("/home/tim/result_wo_icp.pcd", wo_icp);
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

    point_cloud_color = paintPointCloud(point_cloud_livox, cv_ptr->image);

    pcl::io::savePCDFile(save_path + "/" + std::to_string(cunt) + ".pcd", point_cloud_color);
    cv::imwrite(save_path + "/" + std::to_string(cunt) + ".jpg", cv_ptr->image);
    cunt++;
    global_flag = 0;
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
  else if (msg.data == '3')
  {
    global_flag = 3;
    ROS_INFO("Receive logging data command!");
  }
  else if (msg.data == '4')
  {
    ROS_INFO_STREAM("Logging over! Shutting down...");
    std::ofstream outfile(save_path + "/description.txt", std::ios_base::trunc);
    if (cunt == 0)
      outfile << cunt;
    else
      outfile << cunt - 1;
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
  std::string lidar_topic, camera_topic;
  n.getParam("/pointcloud_fusion/save_path", save_path);
  n.getParam("/pointcloud_fusion/lidar_topic", lidar_topic);
  n.getParam("/pointcloud_fusion/camera_topic", camera_topic);
  n.getParam("/pointcloud_fusion/icp_config/MaxCorrespondenceDistance", icp_configs.MaxCorrespondenceDistance);
  n.getParam("/pointcloud_fusion/icp_config/TransformationEpsilon", icp_configs.TransformationEpsilon);
  n.getParam("/pointcloud_fusion/icp_config/MaximumIterations", icp_configs.MaximumIterations);
  n.getParam("/pointcloud_fusion/icp_config/EuclideanFitnessEpsilon", icp_configs.EuclideanFitnessEpsilon);
  n.getParam("/pointcloud_fusion/icp_config/RANSACOutlierRejectionThreshold",
             icp_configs.RANSACOutlierRejectionThreshold);

  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(n, lidar_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> camera_sub(n, camera_topic, 1);
  ros::Subscriber key = n.subscribe("/keyboard/key", 1, KeyBoardCallBack);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, camera_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return EXIT_SUCCESS;
}
