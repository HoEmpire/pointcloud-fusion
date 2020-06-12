#pragma once
#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

const float PI = 3.1415926535;
using namespace Eigen;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
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

Matrix3f hat(Vector3f v)
{
  Matrix3f v_hat;
  v_hat << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return v_hat;
}

MatrixXf leastSquareMethod(MatrixXf A, VectorXf b)
{
  return (A.transpose() * A).inverse() * A.transpose() * b;
}

struct imageType
{
  vector<Mat> imgs, depths, descriptors;
  vector<vector<KeyPoint>> keypoints;
  void init()
  {
    Ptr<SIFT> detector = SIFT::create(1000);
    for (Mat &image : imgs)
    {
      vector<KeyPoint> keypoint;
      Mat descriptor;
      detector->detectAndCompute(image, Mat(), keypoint, descriptor);
      descriptors.push_back(descriptor);
      keypoints.push_back(keypoint);
    }
  };
};

// config util
struct ConfigSetting
{
  string data_path;
  Eigen::Matrix4f extrinsic_matrix;
  Eigen::Matrix3f camera_matrix;
  double k1, k2, k3, p1, p2;
  double max_cor_dis, trans_eps;
  int iter_num;
  double c1, c2, c3;
  void print()
  {
    cout << "Data root: " << data_path << endl;
    cout << "Extrinsic matrix: \n" << extrinsic_matrix << endl;
    cout << "Camera matrix: \n" << camera_matrix << endl;
    cout << "Distortion coeff: \n" << k1 << " " << k2 << " " << k3 << " " << p1 << " " << p2 << endl;
  }
} config;

void readConfig()
{
  ifstream infile("./cfg/config.txt");
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

  infile >> config.c1;
  infile >> config.c2;
  infile >> config.c3;

  infile.close();
  config.print();
}

void readData(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &pcs, struct imageType &image_data)
{
  int data_len;
  ifstream infile(config.data_path + "description.txt");
  infile >> data_len;
  if (!data_len)
    cout << "\n NO data to read!" << endl;
  else
  {
    cout << "The length of the data is: " << data_len << endl;
    cout << "Reading data..." << endl;
  }

  infile.close();

  for (int i = 0; i < data_len; i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());

    pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + to_string(i) + ".pcd", *tmp);
    pcs.push_back(tmp);
    image_data.imgs.push_back(cv::imread(config.data_path + to_string(i) + ".jpg"));
    image_data.depths.push_back(cv::imread(config.data_path + to_string(i) + ".png", CV_16UC1));
  }
}