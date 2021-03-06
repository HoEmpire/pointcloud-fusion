#pragma once
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>  //统计滤波器头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ndt.h>
#include <boost/make_shared.hpp>

#include "type.h"
#include "util.h"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
using namespace std;

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation<PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
  MyPointRepresentation()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray(const PointNormalT &p, float *out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

void icpNonlinearWithNormal(vector<PointCloud::Ptr> clouds, vector<Eigen::Matrix4f> init_T,
                            vector<Eigen::Matrix4f> &result_T)
{
  // point cloud preprocess
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statisticalFilter;  //创建滤波器对象
  statisticalFilter.setMeanK(50);                                      //取平均值的临近点数
  statisticalFilter.setStddevMulThresh(1);  //超过平均距离一个标准差以上，该点记为离群点，将其移除

  for (PointCloud::Ptr &pc : clouds)
  {
    statisticalFilter.setInputCloud(pc);  //设置待滤波的点云
    statisticalFilter.filter(*pc);        //执行滤波处理，保存内点到cloud_after_StatisticalRemoval
  }

  vector<PointCloud::Ptr> clouds_resample;
  for (PointCloud::Ptr &pc : clouds)
  {
    pcl::VoxelGrid<PointT> grid;

    grid.setLeafSize(0.05, 0.05, 0.05);  // TODO hardcode in here
    grid.setInputCloud(pc);
    PointCloud::Ptr cloud_resample(new PointCloud);
    grid.filter(*cloud_resample);
    clouds_resample.push_back(cloud_resample);
  }

  Eigen::Matrix4f final_T;
  final_T.setIdentity(4, 4);
  pcl::PointCloud<pcl::PointXYZRGB> origin = *clouds[0];
  // PCL_INFO("Fusing point clouds");

  for (int i = 0; i < clouds.size() - 1; i++)
  {
    cout << "*****fusing point cloud set " << i << " *****" << endl;
    PointCloud::Ptr src(new PointCloud);
    PointCloud::Ptr tgt(new PointCloud);
    src = clouds_resample[i + 1];
    tgt = clouds_resample[i];

    // Compute surface normals and curvature
    PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
    PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

    pcl::NormalEstimation<pcl::PointXYZRGB, PointNormalT> norm_est;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(30);

    norm_est.setInputCloud(src);
    norm_est.compute(*points_with_normals_src);
    pcl::copyPointCloud(*src, *points_with_normals_src);

    norm_est.setInputCloud(tgt);
    norm_est.compute(*points_with_normals_tgt);
    pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

    // Instantiate our custom point representation (defined above) ...
    MyPointRepresentation point_representation;
    // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
    float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
    point_representation.setRescaleValues(alpha);

    pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> icp;

    pcl::transformPointCloud(*points_with_normals_src, *points_with_normals_src, init_T[i]);
    icp.setInputSource(points_with_normals_src);
    icp.setInputTarget(points_with_normals_tgt);

    icp.setMaxCorrespondenceDistance(config.max_cor_dis);  // 0.10
    icp.setTransformationEpsilon(config.trans_eps);        // 1e-10

    icp.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
    // Run the same optimization in a loop and visualize the results

    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
    PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
    icp.setMaximumIterations(1);
    for (int j = 0; j < config.iter_num; ++j)
    {
      if (j % 10 == 0)
        PCL_INFO("Iteration Nr. %d.\n", j);

      // save cloud for visualization purpose
      points_with_normals_src = reg_result;

      // Estimate
      icp.setInputSource(points_with_normals_src);
      icp.align(*reg_result);

      // accumulate transformation between each Iteration
      Ti = icp.getFinalTransformation() * Ti;

      // if the difference between this transformation and the previous one
      // is smaller than the threshold, refine the process by reducing
      // the maximal correspondence distance
      if (abs((icp.getLastIncrementalTransformation() - prev).sum()) < icp.getTransformationEpsilon())
        icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() - 0.001);

      prev = icp.getLastIncrementalTransformation();
    }

    // ROS_INFO_STREAM("ICP has converged?: " << icp.hasConverged());
    // ROS_INFO_STREAM("Fitness Score: " << icp.getFitnessScore());
    // final_T = final_T * icp.getFinalTransformation();
    final_T = final_T * Ti * init_T[i];

    cout << "Final Transformation: " << endl << final_T << endl;
    cout << "***************************" << endl;
    pcl::PointCloud<pcl::PointXYZRGB> new_cloud;

    pcl::transformPointCloud(*clouds[i + 1], new_cloud, final_T);
    origin += new_cloud;
    result_T.push_back((Ti * init_T[i]).inverse());
  }
  if (clouds.size() > 2)
    pcl::io::savePCDFile("/home/tim/icp.pcd", origin);
  else
    pcl::io::savePCDFile("/home/tim/icp_two_frame.pcd", origin);
  PCL_INFO("Fusion Complete!!");
}

void ndtRegistration(struct pointcloudType point_cloud_data, vector<Eigen::Matrix4f> init_T,
                     vector<Eigen::Matrix4f> &result_T)
{
  Eigen::Matrix4f final_T;
  final_T.setIdentity(4, 4);
  pcl::PointCloud<pcl::PointXYZRGB> origin = *point_cloud_data.pc_filtered[0];
  // PCL_INFO("Fusing point clouds");
  timer t;
  for (int i = 0; i < point_cloud_data.pc_filtered.size() - 1; i++)
  {
    if (point_cloud_data.pc_filtered.size() != 2)
      cout << "*****fusing point cloud set " << i << " *****" << endl;
    t.tic();
    PointCloud::Ptr src(new PointCloud);
    PointCloud::Ptr tgt(new PointCloud);
    src = point_cloud_data.pc_resample[i + 1];
    tgt = point_cloud_data.pc_resample[i];

    //初始化正态分布变换（NDT）
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    //设置依赖尺度NDT参数
    //为终止条件设置最小转换差异
    ndt.setTransformationEpsilon(config.c1);  // 0.01
    //为More-Thuente线搜索设置最大步长
    ndt.setStepSize(config.c2);  // 0.1
    //设置NDT网格结构的分辨率（VoxelGridCovariance）
    ndt.setResolution(config.c3);  // 1.0
    //设置匹配迭代的最大次数
    ndt.setMaximumIterations(config.iter_num);
    // 设置要配准的点云
    ndt.setInputSource(src);
    //设置点云配准目标
    ndt.setInputTarget(tgt);

    //计算需要的刚体变换以便将输入的点云匹配到目标点云
    pcl::PointCloud<PointT>::Ptr output_cloud(new pcl::PointCloud<PointT>);
    ndt.align(*output_cloud, init_T[i]);
    cout << "Normal Distributions Transform has converged:" << ndt.hasConverged() << " score: " << ndt.getFitnessScore()
         << endl;

    final_T = final_T * ndt.getFinalTransformation();
    cout << "Final Transformation: " << endl << ndt.getFinalTransformation() << endl;
    cout << "Registraing point cloud " << i + 1 << " takes " << t.toc() << " seconds." << endl;
    cout << "***************************" << endl << endl;
    pcl::PointCloud<pcl::PointXYZRGB> new_cloud;

    pcl::transformPointCloud(*point_cloud_data.pc_filtered[i + 1], new_cloud, final_T);
    origin += new_cloud;
    result_T.push_back(ndt.getFinalTransformation());
  }
  if (point_cloud_data.pc_filtered.size() > 2)
    // TODO If the size of the point cloud is small than 2, it won't be saved, Better move IO in new function
    pcl::io::savePCDFile("/home/tim/ndt.pcd", origin);
  // else
  //   pcl::io::savePCDFile("/home/tim/ndt_two_frame.pcd", origin);
  PCL_INFO("Fusion Complete!!");
  cout << endl;
}

void calLoopsTransform(struct pointcloudType point_cloud_data, vector<vector<int>> loops, vector<Eigen::Matrix4f> T_vo,
                       vector<Eigen::Matrix4f> T_vertex, vector<vector<int>> &loops_good,
                       vector<Eigen::Matrix4f> &T_result)
{
  for (int i = 0; i < loops.size(); i++)
  {
    vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs_two_origin, pcs_two_filtered, pcs_two_resample;

    pcs_two_origin.push_back(point_cloud_data.pc_origin[loops[i][0]]);
    pcs_two_origin.push_back(point_cloud_data.pc_origin[loops[i][1]]);
    pcs_two_filtered.push_back(point_cloud_data.pc_filtered[loops[i][0]]);
    pcs_two_filtered.push_back(point_cloud_data.pc_filtered[loops[i][1]]);
    pcs_two_resample.push_back(point_cloud_data.pc_resample[loops[i][0]]);
    pcs_two_resample.push_back(point_cloud_data.pc_resample[loops[i][1]]);
    struct pointcloudType pc_data_two(pcs_two_origin, pcs_two_filtered, pcs_two_resample);

    vector<Eigen::Matrix4f> T_result_two, T_init_two;
    T_init_two.push_back(T_vo[i]);

    cout << "*****fusing point " << loops[i][0] << " and " << loops[i][1] << " *****" << endl;
    ndtRegistration(pc_data_two, T_init_two, T_result_two);

    Matrix3f rotation_matrix = T_result_two[0].topLeftCorner(3, 3);

    // uncertainty criterion
    Matrix4f T_edge = T_vertex[loops[i][1]].inverse() * T_vertex[loops[i][0]];

    Matrix4f T_error = T_edge * T_result_two[0];
    Vector3f euler_angle = rotationMatrixToEulerAngles(T_error.topLeftCorner(3, 3)) * 180 / PI;
    cout << "Error information between vertex " << loops[i][0] << " and vertex " << loops[i][1] << endl;
    cout << "error in euler anles (deg): " << euler_angle.transpose() << endl;
    cout << "error in translation (m): " << T_error.topRightCorner(3, 1).transpose() << endl;
    cout << "error sum in angles (deg): " << euler_angle.norm() << endl;
    cout << "error sum in translation (m): " << T_error.topRightCorner(3, 1).norm() << endl;
    int distance = loops[i][0] - loops[i][1];
    if (T_error.topRightCorner(3, 1).norm() < config.uncertainty_translation * distance ||
        euler_angle.norm() < config.uncertainty_degree * distance)
    {
      cout << "This pair is good loop!" << endl;
      T_result.push_back(T_result_two[0]);
      loops_good.push_back(loops[i]);
    }
    else
      cout << "Error is too large, this pair is rejected!" << endl;
    cout << endl;
  }
}