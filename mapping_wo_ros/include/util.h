#pragma once
#include <chrono>
#include <iostream>
#include <vector>

#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>       //统计滤波器头文件
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
  int filter_meanK = 100;
  float filter_std_threshold = 1.0;

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

  infile >> config.filter_meanK;
  infile >> config.filter_std_threshold;

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

// Point cloud depth filter functions
// param threshold = error/mean
void ransacStatisticalFilter(pcl::PointCloud<pcl::PointXYZRGB> input, pcl::PointCloud<pcl::PointXYZRGB> &output,
                             float threshold = 0.01, const int max_iter = 10)
{
  if (input.size() <= 2)
  {
    output = input;
    return;
  }

  pcl::PointCloud<pcl::PointWithRange> input_xyzrange;
  input_xyzrange.resize(input.size());
  for (int i = 0; i < input.size(); i++)
  {
    input_xyzrange[i].x = input[i].x;
    input_xyzrange[i].y = input[i].y;
    input_xyzrange[i].z = input[i].z;
    input_xyzrange[i].range =
        sqrt(input_xyzrange[i].x * input_xyzrange[i].x + input_xyzrange[i].y * input_xyzrange[i].y +
             input_xyzrange[i].z * input_xyzrange[i].z);
  }

  int num_inlier = 0;
  // RANSAC
  vector<int> index_final;
  pcl::PointCloud<pcl::PointWithRange> inlier_final;
  while (inlier_final.size() < 2)
  {
    for (int iter = 0; iter < max_iter; iter++)
    {
      int rand_index = rand() % input.size();
      pcl::PointCloud<pcl::PointWithRange> inlier_tmp;
      vector<int> index_tmp;
      float depth = input_xyzrange[rand_index].range;
      for (int i = 0; i < input_xyzrange.size(); i++)
      {
        if (abs(input_xyzrange[i].range - depth) / (depth + 1e-9) < threshold)
        {
          inlier_tmp.push_back(input_xyzrange[i]);
          index_tmp.push_back(i);
        }
      }
      if (inlier_tmp.size() > num_inlier)
      {
        inlier_final = inlier_tmp;
        num_inlier = inlier_tmp.size();
        index_final = index_tmp;
      }
      if (inlier_final.size() / input.size() * 1.0 > 0.5)
        break;
    }
    threshold *= 2;
  }

  for (int i = 0; i < index_final.size(); i++)
    output.push_back(input[index_final[i]]);
}

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

  void filter(int meanK, float std_threshold)
  {
    // point cloud preprocess
    cout << "Start filtering the point cloud!" << endl;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statisticalFilter;
    statisticalFilter.setMeanK(meanK);  // TODO hardcode in here
    statisticalFilter.setStddevMulThresh(std_threshold);
    // statisticalFilter.setKeepOrganized(true);
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radiusFilter;  //创建滤波器对象

    // radiusFilter.setRadiusSearch(std_threshold);  // 设置搜索半径
    // radiusFilter.setMinNeighborsInRadius(meanK);  // 设置一个内点最少的邻居数目

    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_filtered)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
      statisticalFilter.setInputCloud(pc);
      statisticalFilter.filter(*pc);
      // radiusFilter.setInputCloud(pc);
      // radiusFilter.filter(*pc);  //滤波结果存储到cloud_filtered

      // pc_filtered.push_back(cloud_filtered);
    }
    cout << "Filtering finished!" << endl;
  }

  void depthFilter()
  {
    // point cloud preprocess
    cout << "Start filtering the point cloud!" << endl;
    timer t;
    t.tic();
    vector<vector<long int>> index(1080 / 10, vector<long int>(1440 / 10, -1));
    // long int index[1080 / 10][1440 / 10];
    // for (int i = 0; i < 1080 / 10; i++)
    //   for (int j = 0; j < 1440 / 10; j++)
    //     index[i][j] = -1;
    vector<pcl::PointCloud<pcl::PointXYZRGB>> point_cloud_for_process;
    long int count = 0;
    float fx = config.camera_matrix(0, 0);
    float fy = config.camera_matrix(1, 1);
    float cx = config.camera_matrix(0, 2);
    float cy = config.camera_matrix(1, 2);
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_origin)
    {
      // cout << "fuck1" << endl;
      // deal with a set of point cloud
      for (pcl::PointXYZRGB &p : *pc)
      {
        // cout << "fuck2" << endl;
        // Vector3f point;
        // if (p.x != 0)
        //   point << -p.y / p.x, -p.z / p.x, 1.0;
        // else
        //   continue;
        // point = config.camera_matrix * point / 10.0;

        // cout << "fuck3" << endl;
        int u, v;
        u = int((-p.y / p.x * fx + cx) / 10.0);
        v = int((-p.z / p.x * fy + cy) / 10.0);
        if (u >= 0 && u < 1440 / 10 && v >= 0 && v < 1080 / 10)
        {
          // cout << u << " " << v << endl;
          if (index[v][u] == -1)
          {
            index[v][u] = count;
            count++;
            // cout << count << endl;

            pcl::PointCloud<pcl::PointXYZRGB> point_cloud_tmp;
            point_cloud_tmp.push_back(p);
            point_cloud_for_process.push_back(point_cloud_tmp);
          }
          else
            point_cloud_for_process[index[v][u]].push_back(p);
        }
        else
        {
          cout << u << " " << v << endl;
          continue;
        }
        // cout << "fuck4" << endl;
      }
      cout << "Getting index take " << t.toc() << " seconds" << endl;
      cout << "Original point cloud have " << pc->points.size() << " points" << endl;
      t.tic();
      pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
      // cout << "fuck5" << endl;
      for (int i = 0; i < point_cloud_for_process.size(); i++)
      {
        // cout << i << endl;
        pcl::PointCloud<pcl::PointXYZRGB> one_point_cloud;
        ransacStatisticalFilter(point_cloud_for_process[i], one_point_cloud);
        // cout << one_point.x << " " << one_point.y << " " << one_point.z << endl;
        cloud_filtered += one_point_cloud;
      }
      // cout << "fuck6" << endl;
      pc_filtered.push_back(cloud_filtered.makeShared());
      cout << "Filtering pc take " << t.toc() << " seconds" << endl;
      cout << "Filtered point cloud have " << cloud_filtered.size() << " points" << endl;
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