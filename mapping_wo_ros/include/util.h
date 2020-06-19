#pragma once
#include <chrono>
#include <iostream>
#include <vector>

#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>  //统计滤波器头文件
#include <pcl/filters/voxel_grid.h>
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
using namespace std::chrono;

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
    cout << "Extracting features in images!" << endl;
    Ptr<SIFT> detector = SIFT::create(2000);
    for (Mat &image : imgs)
    {
      vector<KeyPoint> keypoint;
      Mat descriptor;
      detector->detectAndCompute(image, Mat(), keypoint, descriptor);
      descriptors.push_back(descriptor);
      keypoints.push_back(keypoint);
    }
    cout << "Features extraction finished!" << endl;
  };
};

struct pointcloudType
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_origin, pc_resample, pc_filtered;

  pointcloudType(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_origin,
                 vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_resample,
                 vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_filtered)
  {
    this->pc_origin = pc_origin;
    this->pc_resample = pc_resample;
    this->pc_filtered = pc_filtered;
  }

  pointcloudType(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pc_origin)
  {
    this->pc_origin = pc_origin;
  }

  void filter()
  {
    // point cloud preprocess
    cout << "Start filtering the point cloud!" << endl;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statisticalFilter;
    statisticalFilter.setMeanK(50);  // TODO hardcode in here
    statisticalFilter.setStddevMulThresh(1);

    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_origin)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
      statisticalFilter.setInputCloud(pc);
      statisticalFilter.filter(*cloud_filtered);
      pc_filtered.push_back(cloud_filtered);
    }
    cout << "Filtering finished!" << endl;
  }

  void resample(double resolution)
  {
    cout << "Start resampling the point cloud!" << endl;
    pcl::VoxelGrid<pcl::PointXYZRGB> grid;
    grid.setLeafSize(resolution, resolution, resolution);
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_filtered)
    {
      grid.setInputCloud(pc);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_resample(new pcl::PointCloud<pcl::PointXYZRGB>);
      grid.filter(*cloud_resample);
      pc_resample.push_back(cloud_resample);
    }
    cout << "Resampling finished!" << endl;
  }
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
  double point_cloud_resolution;
  int num_of_result = 4;
  int frame_distance_threshold = 2;
  float score_threshold = 0.8;

  void print()
  {
    cout << "Data root: " << data_path << endl;
    cout << "Extrinsic matrix: \n" << extrinsic_matrix << endl;
    cout << "Camera matrix: \n" << camera_matrix << endl;
    cout << "Distortion coeff: \n" << k1 << " " << k2 << " " << k3 << " " << p1 << " " << p2 << endl;
  }
} config;

struct timer
{
  steady_clock::time_point t_start, t_end;
  void tic()
  {
    t_start = steady_clock::now();
  }

  double toc()
  {
    t_end = steady_clock::now();
    return duration_cast<duration<double>>(t_end - t_start).count();
  }
};

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
  infile >> config.point_cloud_resolution;

  infile >> config.num_of_result;
  infile >> config.frame_distance_threshold;
  infile >> config.score_threshold;

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

void readDataWithID(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &pcs, struct imageType &image_data)
{
  vector<int> ids;
  int tmp;
  ifstream infile(config.data_path + "id.txt");
  while (1)
  {
    infile >> tmp;
    if (infile.eof())
      break;
    cout << "read id: " << tmp << endl;
    ids.push_back(tmp);
  }

  infile.close();

  for (int i = 0; i < ids.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());

    pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + to_string(ids[i]) + ".pcd", *tmp);
    pcs.push_back(tmp);
    image_data.imgs.push_back(cv::imread(config.data_path + to_string(ids[i]) + ".jpg"));
    image_data.depths.push_back(cv::imread(config.data_path + to_string(ids[i]) + ".png", CV_16UC1));
  }
}