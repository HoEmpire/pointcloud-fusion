#pragma once
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>       //统计滤波器头文件
#include <pcl/filters/statistical_outlier_removal.h>  //统计滤波器头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include "util.h"

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace Eigen;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

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
    struct imageType new_image_data(imgs_tmp, depths_tmp, descriptors_tmp, keypoints_tmp);
    return new_image_data;
  }
};

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

    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc : pc_origin)
    {
      vector<vector<long int>> index(box_size_y / ratio, vector<long int>(box_size_x / ratio, -1));
      vector<pcl::PointCloud<pcl::PointXYZRGB>> point_cloud_for_process;
      long int count = 0;
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

  vector<cv::Mat> paintPointCloud(const vector<cv::Mat> imgs)
  {
    vector<cv::Mat> depths;
    if (imgs.size() < 1)
    {
      cout << "Paint Point Cloud: The length of the image vector is 0!" << endl;
      return depths;
    }

    timer t;
    t.tic();
    int row = imgs[0].rows;
    int col = imgs[0].cols;

    // cout << "fuck1" << endl;
    Eigen::Vector4f p;
    float fx, fy, cx, cy;
    fx = config.camera_matrix(0, 0);
    fy = config.camera_matrix(1, 1);
    cx = config.camera_matrix(0, 2);
    cy = config.camera_matrix(1, 2);

    // cv::Mat depth_map;
    for (int i = 0; i < pc_filtered.size(); i++)
    {
      cv::Mat depth_map = cv::Mat::zeros(row, col, CV_16UC1);
      for (pcl::PointCloud<pcl::PointXYZRGB>::iterator pt = pc_filtered[i]->points.begin();
           pt < pc_filtered[i]->points.end(); pt++)
      {
        // cout << "fuck2" << endl;
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

        int x = int(p(0) / p(2) * fx + cx);
        int y = int(p(1) / p(2) * fy + cy);
        if (x >= 0 && x < col && y >= 0 && y < row)
        {
          uchar const *img_ptr = imgs[i].ptr<uchar>(y);
          pt->b = img_ptr[3 * x];
          pt->g = img_ptr[3 * x + 1];
          pt->r = img_ptr[3 * x + 2];
          if (depth >= 0)
          {
            depth_map.at<ushort>(y, x) = ushort(depth);
          }
        }
        // cout << "fuck3" << endl;
      }
      depths.push_back(depth_map);
    }

    cout << "Paint Point Cloud: Painting pc and getting depth maps takes " << t.toc() << " seconds" << endl;
    return depths;
  }
};