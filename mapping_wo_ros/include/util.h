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

  imageType(vector<Mat> imgs, vector<Mat> depths, vector<Mat> descriptors, vector<vector<KeyPoint>> keypoints)
  {
    this->imgs = imgs;
    this->depths = depths;
    this->descriptors = descriptors;
    this->keypoints = keypoints;
  }

  imageType()
  {
  }

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
  }

  struct imageType copy_by_index(vector<int> index)
  {
    vector<Mat> imgs_tmp, depths_tmp, descriptors_tmp;
    vector<vector<KeyPoint>> keypoints_tmp;
    for (int i = 0; i < index.size(); i++)
    {
      imgs_tmp.push_back(this->imgs[index[i]]);
      depths_tmp.push_back(this->depths[index[i]]);
      descriptors_tmp.push_back(this->descriptors[index[i]]);
      keypoints_tmp.push_back(this->keypoints[index[i]]);
    }
    struct imageType new_image_data(imgs_tmp, depths_tmp, depths_tmp, keypoints_tmp);
    return new_image_data;
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
  int filter_meanK = 100;
  float filter_std_threshold = 1.0;
  float uncertainty_translation = 0.1;  // m
  float uncertainty_degree = 0.5;       // deg

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

  void standardFilter(int meanK, float std_threshold)
  {
    // point cloud preprocess
    timer t;
    cout << "Standard Filter: Start filtering the point cloud!" << endl;
    t.tic();
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statisticalFilter;
    statisticalFilter.setMeanK(meanK);  // TODO hardcode in here
    statisticalFilter.setStddevMulThresh(std_threshold);
    // statisticalFilter.setKeepOrganized(true);
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radiusFilter;  //创建滤波器对象

    // radiusFilter.setRadiusSearch(std_threshold);  // 设置搜索半径
    // radiusFilter.setMinNeighborsInRadius(meanK);  // 设置一个内点最少的邻居数目
    if (pc_filtered.size() != 0)
    {
      for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_filtered)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
        statisticalFilter.setInputCloud(pc);
        statisticalFilter.filter(*pc);
        // radiusFilter.setInputCloud(pc);
        // radiusFilter.filter(*pc);  //滤波结果存储到cloud_filtered

        // pc_filtered.push_back(cloud_filtered);
      }
    }
    else
    {
      for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_origin)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
        statisticalFilter.setInputCloud(pc);
        statisticalFilter.filter(*pc);
        // radiusFilter.setInputCloud(pc);
        // radiusFilter.filter(*pc);  //滤波结果存储到cloud_filtered

        // pc_filtered.push_back(cloud_filtered);
      }
    }
    cout << "Standard filter: Filtering pc takes " << t.toc() << " seconds" << endl;
    cout << "Standard Filter: Filtering finished!" << endl;
  }

  // can only be used as first step of point cloud preprocess
  // @param ratio: Smaller value for large scale dataset, larger value for small scale dataset,
  void depthFilter(float ratio = 10.0)
  {
    if (pc_origin.size() == 0)
    {
      cout << "Error in depth filter! No point cloud exists!" << endl;
      return;
    }
    const int box_size_x = 1000;
    const int box_size_y = 1000;
    timer t;
    // Get the size of the 2D array ()
    t.tic();
    cout << "Depth filter: Getting mapping size..." << endl;
    float x_max, x_min, y_max, y_min;
    float r_x, r_y, c_x, c_y;
    x_max = y_max = -10000.0;
    x_min = y_min = 10000.0;
    for (pcl::PointXYZRGB &p : *pc_origin[0])
    {
      float x = -p.y / p.x;
      float y = -p.z / p.x;
      if (x > x_max)
        x_max = x;
      if (x < x_min)
        x_min = x;
      if (y > y_max)
        y_max = y;
      if (y < y_min)
        y_min = y;
    }
    r_x = x_max - x_min;
    r_y = y_max - y_min;
    c_x = (x_max + x_min) / 2.0;
    c_y = (y_max + y_min) / 2.0;
    cout << "Depth filter: diameter of x: " << r_x << endl;
    cout << "Depth filter: diameter of y: " << r_y << endl;
    cout << "Depth filter: center of x: " << c_x << endl;
    cout << "Depth filter: center of y: " << c_y << endl;
    const float safety_ratio = 1.2;
    float fx, fy;
    fx = box_size_x * 1.0 / (r_x * safety_ratio);
    fy = box_size_y * 1.0 / (r_y * safety_ratio);

    float cx = c_x * fx + box_size_x / 2.0;
    float cy = c_y * fy + box_size_y / 2.0;
    cout << "Depth filter: fx: " << fx << endl;
    cout << "Depth filter: fy: " << fy << endl;
    cout << "Depth filter: cx: " << cx << endl;
    cout << "Depth filter: cy: " << cy << endl;
    cout << "Depth filter: Getting suitable size takes " << t.toc() << " seconds" << endl;

    t.tic();
    cout << "Depth filter: Start filtering the point cloud!" << endl;
    vector<vector<long int>> index(box_size_y / ratio, vector<long int>(box_size_x / ratio, -1));
    vector<pcl::PointCloud<pcl::PointXYZRGB>> point_cloud_for_process;
    long int count = 0;

    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_origin)
    {
      for (pcl::PointXYZRGB &p : *pc)
      {
        int u, v;
        u = int((-p.y / p.x * fx + cx) / ratio);
        v = int((-p.z / p.x * fy + cy) / ratio);
        if (u >= 0 && u < box_size_x / ratio && v >= 0 && v < box_size_y / ratio)
        {
          if (index[v][u] == -1)
          {
            index[v][u] = count;
            count++;

            pcl::PointCloud<pcl::PointXYZRGB> point_cloud_tmp;
            point_cloud_tmp.push_back(p);
            point_cloud_for_process.push_back(point_cloud_tmp);
          }
          else
            point_cloud_for_process[index[v][u]].push_back(p);
        }
        else
        {
          // cout << u << " " << v << endl;
          cout << "Depth filter: WARN:Initial boxing size is not suitable!!!" << endl;
          abort();
        }
      }
      cout << "Depth filter: Getting index takes " << t.toc() << " seconds" << endl;

      t.tic();
      pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
      for (int i = 0; i < point_cloud_for_process.size(); i++)
      {
        pcl::PointCloud<pcl::PointXYZRGB> one_point_cloud;
        ransacStatisticalFilter(point_cloud_for_process[i], one_point_cloud);
        cloud_filtered += one_point_cloud;
      }
      pc_filtered.push_back(cloud_filtered.makeShared());
      cout << "Depth filter: Filtering pc takes " << t.toc() << " seconds" << endl;
      cout << "Depth filter: Original point cloud have " << pc->points.size() << " points" << endl;
      cout << "Depth filter: Filtered point cloud (depth filter) have " << cloud_filtered.size() << " points" << endl;
    }
    cout << "Depth filter: Filtering finished!" << endl;
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