#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>

const float PI = 3.1415926535;
using namespace Eigen;
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

// config util
struct ConfigSetting
{
  Eigen::Matrix4f extrinsic_matrix;
  Eigen::Matrix3f camera_matrix;
  double k1, k2, k3, p1, p2;
  double max_cor_dis, trans_eps;
  int iter_num;
  double trans_eps_ndt, step_size_ndt, resolution_ndt;
  double point_cloud_resolution;
  int num_of_result = 4;
  int frame_distance_threshold = 2;
  float score_threshold = 0.8;
  int filter_meanK = 100;
  float filter_std_threshold = 1.0;
  float uncertainty_translation = 0.1;  // m
  float uncertainty_rotation = 0.5;     // deg
  float depth_filter_ratio = 5.0;
  string pkg_loc;
  string pc_save_path;

  void print()
  {
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
  config.pkg_loc = ros::package::getPath("pointcloud_fusion");
  ifstream infile(config.pkg_loc + "/cfg/config.txt");
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

  // infile >> config.max_cor_dis;
  // infile >> config.trans_eps;
  // infile >> config.iter_num;

  // infile >> config.c1;
  // infile >> config.c2;
  // infile >> config.c3;
  // infile >> config.point_cloud_resolution;

  // infile >> config.num_of_result;
  // infile >> config.frame_distance_threshold;
  // infile >> config.score_threshold;

  // infile >> config.filter_meanK;
  // infile >> config.filter_std_threshold;

  infile.close();
  config.print();
}

void loadConfig(ros::NodeHandle n)
{
  n.getParam("/icp_nolinear/max_correspondence_distance", config.max_cor_dis);
  n.getParam("/icp_nolinear/transformation_epsilon", config.trans_eps);
  n.getParam("/ndt/num_iteration", config.iter_num);

  n.getParam("/ndt/transformation_epsilon", config.trans_eps_ndt);
  n.getParam("/ndt/step_size", config.step_size_ndt);
  n.getParam("/ndt/resolution", config.resolution_ndt);

  n.getParam("/point_cloud_preprocess/resample_resolution", config.point_cloud_resolution);
  n.getParam("/point_cloud_preprocess/statistical_filter_meanK", config.filter_meanK);
  n.getParam("/point_cloud_preprocess/statistical_filter_std", config.filter_std_threshold);
  n.getParam("/point_cloud_preprocess/depth_filter_ratio", config.depth_filter_ratio);

  n.getParam("/loop_closure/num_of_result", config.num_of_result);
  n.getParam("/loop_closure/frame_distance_threshold", config.frame_distance_threshold);
  n.getParam("/loop_closure/score_threshold", config.score_threshold);
  n.getParam("/loop_closure/num_of_result", config.num_of_result);
  n.getParam("/loop_closure/translation_uncertainty", config.uncertainty_translation);
  n.getParam("/loop_closure/rotation_uncertainty", config.uncertainty_rotation);

  n.getParam("/io/point_cloud_save_path", config.pc_save_path);
}

void paintPointCloud(pcl::PointCloud<pcl::PointXYZI> point_cloud, const cv::Mat img,
                     pcl::PointCloud<pcl::PointXYZRGB> &point_cloud_color, cv::Mat &depth_map)
{
  int row = img.rows;
  int col = img.cols;
  depth_map = cv::Mat::zeros(row, col, CV_16UC1);
  Eigen::Vector4f p;
  float fx, fy, cx, cy;
  fx = config.camera_matrix(0, 0);
  fy = config.camera_matrix(1, 1);
  cx = config.camera_matrix(0, 2);
  cy = config.camera_matrix(1, 2);
  for (pcl::PointCloud<pcl::PointXYZI>::iterator pt = point_cloud.points.begin(); pt < point_cloud.points.end(); pt++)
  {
    p << 0, 0, 0, 1;
    p(0) = pt->x;
    p(1) = pt->y;
    p(2) = pt->z;
    if (p(2) == 0)
      continue;

    p = config.extrinsic_matrix * p;
    if (p(2) > 65.0)
      continue;
    float depth = p(2) * 1000;
    p = p / p(2);

    // double r2 = p(0) * p(0) + p(1) * p(1);

    // double k1, k2, k3, p1, p2;
    // k1 = config.k1;
    // k2 = config.k2;
    // k3 = config.k3;
    // p1 = config.p1;
    // p2 = config.p2;

    // double x_distorted, y_distorted;
    // x_distorted =
    //     p(0) * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * p(0) * p(1) + p2 * (r2 + 2 * p(0) * p(0));
    // y_distorted =
    //     p(1) * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p2 * p(0) * p(1) + p1 * (r2 + 2 * p(1) * p(1));

    // p(0) = x_distorted;
    // p(1) = y_distorted;

    int x = int(p(0) * fx + cx);
    int y = int(p(1) * fy + cy);
    if (x >= 0 && x < col && y >= 0 && y < row)
    {
      pcl::PointXYZRGB new_point;
      uchar const *img_ptr = img.ptr<uchar>(y);
      new_point.x = pt->x;
      new_point.y = pt->y;
      new_point.z = pt->z;
      new_point.b = img_ptr[3 * x];
      new_point.g = img_ptr[3 * x + 1];
      new_point.r = img_ptr[3 * x + 2];
      point_cloud_color.push_back(new_point);
      if (depth >= 0)
        depth_map.at<ushort>(y, x) = ushort(depth);
    }
  }
}

void undistortImage(const cv::Mat image_distorted, cv::Mat &image_undistorted)
{
  double fx, cx, fy, cy;
  fx = config.camera_matrix(0, 0);
  cx = config.camera_matrix(0, 2);
  fy = config.camera_matrix(1, 1);
  cy = config.camera_matrix(1, 2);
  double k1, k2, k3, p1, p2;
  k1 = config.k1;
  k2 = config.k2;
  k3 = config.k3;
  p1 = config.p1;
  p2 = config.p2;

  image_undistorted = cv::Mat::zeros(image_distorted.rows, image_distorted.cols, image_distorted.type());
  for (int v = 0; v < image_undistorted.rows; v++)
  {
    for (int u = 0; u < image_undistorted.cols; u++)
    {
      double x, y, x_distorted, y_distorted;
      x = (u - cx) / fx;
      y = (v - cy) / fy;

      double r2 = x * x + y * y;
      x_distorted = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
      y_distorted = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p2 * x * y + p1 * (r2 + 2 * y * y);

      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;

      if (u_distorted >= 0 && u_distorted < image_distorted.cols && v_distorted >= 0 &&
          v_distorted < image_distorted.rows)
      {
        image_undistorted.at<uchar>(v, 3 * u) = image_distorted.at<uchar>(int(v_distorted), 3 * int(u_distorted));
        image_undistorted.at<uchar>(v, 3 * u + 1) =
            image_distorted.at<uchar>(int(v_distorted), 3 * int(u_distorted) + 1);
        image_undistorted.at<uchar>(v, 3 * u + 2) =
            image_distorted.at<uchar>(int(v_distorted), 3 * int(u_distorted) + 2);
      }
    }
  }
}