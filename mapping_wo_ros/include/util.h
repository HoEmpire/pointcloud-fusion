#pragma once
#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

const float PI = 3.1415926535;
using namespace Eigen;
Vector3f rotationMatrixToEulerAngles(Matrix3f R)
{
  float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

  bool singular = sy < 1e-6;  // If

  float x, y, z;
  if (!singular)
  {
    x = atan2(R(2, 1), R(2, 2));
    y = atan2(-R(2, 0), sy);
    z = atan2(R(1, 0), R(0, 0));
  }
  else
  {
    x = atan2(-R(1, 2), R(1, 1));
    y = atan2(-R(2, 0), sy);
    z = 0;
  }
  Vector3f result;
  result << x, y, z;
  return result;
}

// config util
struct ConfigSetting
{
  std::string data_path;
  Eigen::Matrix4f extrinsic_matrix;
  Eigen::Matrix3f camera_matrix;
  double k1, k2, k3, p1, p2;
  double max_cor_dis, trans_eps;
  int iter_num;
  void print()
  {
    std::cout << "Data root: " << data_path << std::endl;
    std::cout << "Extrinsic matrix: \n" << extrinsic_matrix << std::endl;
    std::cout << "Camera matrix: \n" << camera_matrix << std::endl;
    std::cout << "Distortion coeff: \n" << k1 << " " << k2 << " " << k3 << " " << p1 << " " << p2 << std::endl;
  }
} config;

void readConfig()
{
  std::ifstream infile("./cfg/config.txt");
  infile >> config.data_path;
  config.extrinsic_matrix.setIdentity(4, 4);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      infile >> config.extrinsic_matrix(i, j);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      infile >> config.camera_matrix(i, j);

  infile >> config.k1;
  infile >> config.k2;
  infile >> config.p1;
  infile >> config.p2;
  infile >> config.k3;
  infile >> config.max_cor_dis;
  infile >> config.trans_eps;
  infile >> config.iter_num;

  infile.close();
  config.print();
}

void readData(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &pcs, std::vector<cv::Mat> &imgs,
              std::vector<cv::Mat> &depths)
{
  int data_len;
  std::ifstream infile(config.data_path + "description.txt");
  infile >> data_len;
  if (!data_len)
    std::cout << "\n NO data to read!" << std::endl;
  else
  {
    std::cout << "The length of the data is: " << data_len << std::endl;
    std::cout << "Reading data..." << std::endl;
  }

  infile.close();
  for (int i = 0; i < data_len; i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + std::to_string(i) + ".pcd", *tmp);
    pcs.push_back(tmp);
    imgs.push_back(cv::imread(config.data_path + std::to_string(i) + ".jpg"));
    depths.push_back(cv::imread(config.data_path + std::to_string(i) + ".png", CV_16UC1));
  }
}