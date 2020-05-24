#pragma once
#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <ros/package.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
using namespace std;

// config util
struct ConfigSetting
{
  Eigen::Matrix4f extrinsic_matrix, camera_matrix, projection_matrix;
  double k1, k2, k3, p1, p2;
  double max_cor_dis, trans_eps;
  int iter_num;
  void print()
  {
    std::cout << "Extrinsic matrix: \n" << extrinsic_matrix << std::endl;
    std::cout << "Camera matrix: \n" << camera_matrix << std::endl;
    std::cout << "Projection matrix: \n" << projection_matrix << std::endl;
    std::cout << "Distortion coeff: \n" << k1 << " " << k2 << " " << k3 << " " << p1 << " " << p2 << std::endl;
  }
} config;

void readConfig()
{
  std::string pkg_loc = ros::package::getPath("pointcloud_fusion");
  std::ifstream infile(pkg_loc + "/cfg/config.txt");

  config.extrinsic_matrix.setIdentity(4, 4);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      infile >> config.extrinsic_matrix(i, j);

  config.camera_matrix.setIdentity(4, 4);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      infile >> config.camera_matrix(i, j);

  config.projection_matrix = config.camera_matrix * config.extrinsic_matrix;

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

void paintPointCloud(pcl::PointCloud<pcl::PointXYZI> point_cloud, const cv::Mat img,
                     pcl::PointCloud<pcl::PointXYZRGB> &point_cloud_color, cv::Mat &depth_map)
{
  int row = img.rows;
  int col = img.cols;
  depth_map = cv::Mat::zeros(row, col, CV_16UC1);
  Eigen::Vector4f p;
  for (pcl::PointCloud<pcl::PointXYZI>::iterator pt = point_cloud.points.begin(); pt < point_cloud.points.end(); pt++)
  {
    p << 0, 0, 0, 1;
    p(0) = pt->x;
    p(1) = pt->y;
    p(2) = pt->z;
    if (p(2) == 0)
      continue;
    p = config.extrinsic_matrix * p;
    float depth = p(2) * 10000;
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
    p = config.camera_matrix * p;

    int x = int(p(0));
    int y = int(p(1));
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