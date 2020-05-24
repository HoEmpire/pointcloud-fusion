#pragma once
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <Eigen/Core>
#include <boost/make_shared.hpp>
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

void icpNonlinearWithNormal(vector<PointCloud::Ptr> clouds, vector<Eigen::Matrix4f> init_T)
{
  Eigen::Matrix4f final_T;
  final_T.setIdentity(4, 4);
  pcl::PointCloud<pcl::PointXYZRGB> origin = *clouds[0];
  pcl::PointCloud<pcl::PointXYZRGB> wo_icp = *clouds[0];
  // PCL_INFO("Fusing point clouds");
  for (int i = 0; i < clouds.size() - 1; i++)
  {
    std::cout << "*****fusing point cloud set " << i << " *****" << std::endl;
    PointCloud::Ptr src(new PointCloud);
    PointCloud::Ptr tgt(new PointCloud);
    pcl::VoxelGrid<PointT> grid;

    grid.setLeafSize(0.01, 0.01, 0.01);
    grid.setInputCloud(clouds[i + 1]);
    grid.filter(*src);

    grid.setInputCloud(clouds[i]);
    grid.filter(*tgt);

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
      if (std::abs((icp.getLastIncrementalTransformation() - prev).sum()) < icp.getTransformationEpsilon())
        icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() - 0.001);

      prev = icp.getLastIncrementalTransformation();
    }

    // ROS_INFO_STREAM("ICP has converged?: " << icp.hasConverged());
    // ROS_INFO_STREAM("Fitness Score: " << icp.getFitnessScore());
    // final_T = final_T * icp.getFinalTransformation();
    final_T = final_T * Ti * init_T[i];

    std::cout << "Final Transformation: " << std::endl << final_T << std::endl;
    std::cout << "***************************" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB> new_cloud;

    pcl::transformPointCloud(*clouds[i + 1], new_cloud, final_T);
    origin += new_cloud;
    wo_icp += *clouds[i + 1];
  }
  pcl::io::savePCDFile("/home/tim/icp.pcd", origin);
  pcl::io::savePCDFile("/home/tim/wo_icp.pcd", wo_icp);
  PCL_INFO("Fusion Complete!!");
}