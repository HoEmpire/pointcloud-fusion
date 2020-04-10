#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

// config util
struct config_settings {
  Eigen::Matrix4d extrinsic_matrix, camera_matrix, projection_matrix;
  void print() {
    std::cout << "Extrinsic matrix: \n" << extrinsic_matrix << "\n";
    std::cout << "Camera matrix: \n" << camera_matrix << "\n";
    std::cout << "Projection matrix: \n" << projection_matrix << "\n";
  }
} config;

void readConfig() {
  std::string pkg_loc = ros::package::getPath("pointcloud_fusion");
  // std::cout<< "The conf file location: " << pkg_loc <<"/conf/config_file.txt"
  // << std::endl;
  std::ifstream infile(pkg_loc + "/conf/config.txt");

  Eigen::Vector3d initial_rot_vec;
  Eigen::Matrix4d initial_T;
  for (int i = 0; i < 3; i++) {
    infile >> initial_rot_vec(i);
  }
  initial_T.setIdentity(4, 4);
  initial_T.topLeftCorner(3, 3) =
      Eigen::AngleAxisd(initial_rot_vec[2], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(initial_rot_vec[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(initial_rot_vec[0], Eigen::Vector3d::UnitX());

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

void paint