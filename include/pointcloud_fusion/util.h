#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <ros/package.h>

// config util
struct config_settings
{
  Eigen::Matrix4d extrinsic_matrix, camera_matrix, projection_matrix;
  void print()
  {
    std::cout << "Extrinsic matrix: \n"
              << extrinsic_matrix << "\n";
    std::cout << "Camera matrix: \n"
              << camera_matrix << "\n";
    std::cout << "Projection matrix: \n"
              << projection_matrix << "\n";
  }
} config;

void readConfig()
{
  std::string pkg_loc = ros::package::getPath("pointcloud_fusion");
  // std::cout<< "The conf file location: " << pkg_loc <<"/conf/config_file.txt"
  // << std::endl;
  std::ifstream infile(pkg_loc + "/cfg/config.txt");

  Eigen::Vector3d initial_rot_vec;
  Eigen::Matrix4d initial_T;
  for (int i = 0; i < 3; i++)
  {
    infile >> initial_rot_vec(i);
  }
  initial_T.setIdentity(4, 4);

  Eigen::AngleAxisd rollAngle(initial_rot_vec[0], Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(initial_rot_vec[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(initial_rot_vec[2], Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d initial_rot_matrix;
  initial_rot_matrix = yawAngle * pitchAngle * rollAngle;
  initial_T.topLeftCorner(3, 3) = initial_rot_matrix;

  config.extrinsic_matrix.setIdentity(4, 4);
  for (int i = 0; i < 3; i++)
    infile >> config.extrinsic_matrix(i, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      infile >> config.extrinsic_matrix(i, j);
  config.extrinsic_matrix = config.extrinsic_matrix * initial_T;

  config.camera_matrix.setIdentity(4, 4);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      infile >> config.camera_matrix(i, j);

  config.projection_matrix = config.camera_matrix * config.extrinsic_matrix;

  infile.close();
  config.print();
}

pcl::PointCloud<pcl::PointXYZRGB>
paintPointCloud(pcl::PointCloud<pcl::PointXYZI> point_cloud, cv::Mat img)
{
  pcl::PointCloud<pcl::PointXYZRGB> point_cloud_color;
  int row = img.rows;
  int col = img.cols;
  Eigen::Vector4d p;
  p << 0, 0, 0, 1;
  for (pcl::PointCloud<pcl::PointXYZI>::iterator pt =
           point_cloud.points.begin();
       pt < point_cloud.points.end(); pt++)
  {
    p(0) = pt->x;
    p(1) = pt->y;
    p(2) = pt->z;
    p = config.projection_matrix * p;
    int x = int(p(0) / p(2));
    int y = int(p(1) / p(2));
    if (x >= 0 && x < col && y >= 0 && y < row)
    {
      pcl::PointXYZRGB new_point;
      uchar *img_ptr = img.ptr<uchar>(y);
      new_point.x = pt->x;
      new_point.y = pt->y;
      new_point.z = pt->z;
      new_point.b = img_ptr[3 * x];
      new_point.g = img_ptr[3 * x + 1];
      new_point.r = img_ptr[3 * x + 2];
      point_cloud_color.push_back(new_point);
    }
  }
  return point_cloud_color;
}