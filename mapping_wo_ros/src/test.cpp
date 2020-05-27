// An example showing TEASER++ registration with the Stanford bunny model
#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include "util.h"

// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.02
#define N_OUTLIERS 1700
#define OUTLIER_TRANSLATION_LB 5
#define OUTLIER_TRANSLATION_UB 10

using namespace std;

inline double getAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est)
{
  return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}

void addNoiseAndOutliers(Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt)
{
  // Add uniform noise
  Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, tgt.cols()) * NOISE_BOUND;
  NOISE_BOUND / 2;
  tgt = tgt + noise;

  // Add outliers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis2(0, tgt.cols() - 1);                               // pos of outliers
  std::uniform_int_distribution<> dis3(OUTLIER_TRANSLATION_LB, OUTLIER_TRANSLATION_UB);  // random translation
  std::vector<bool> expected_outlier_mask(tgt.cols(), false);
  for (int i = 0; i < N_OUTLIERS; ++i)
  {
    int c_outlier_idx = dis2(gen);
    assert(c_outlier_idx < expected_outlier_mask.size());
    expected_outlier_mask[c_outlier_idx] = true;
    tgt.col(c_outlier_idx).array() += dis3(gen);  // random translation
  }
}

int main()
{
  // // Load the .ply file
  // teaser::PLYReader reader;
  // teaser::PointCloud src_cloud;
  // auto status = reader.read("./example_data/bun_zipper_res3.ply", src_cloud);
  // int N = src_cloud.size();

  // // Convert the point cloud to Eigen
  // Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N);
  // for (size_t i = 0; i < N; ++i)
  // {
  //   src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
  // }

  // // Homogeneous coordinates
  // Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  // src_h.resize(4, src.cols());
  // src_h.topRows(3) = src;
  // src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // // Apply an arbitrary SE(3) transformation
  // Eigen::Matrix4d T;
  // // clang-format off
  // T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
  //     -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
  //     4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
  //     0,              0,                0,              1;
  // // clang-format on

  // // Apply transformation
  // Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  // Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);

  // // Add some noise & outliers
  // addNoiseAndOutliers(tgt);

  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  pcl::PointCloud<pcl::PointXYZRGB> src_cloud;
  pcl::PointCloud<pcl::PointXYZRGB> tgt_cloud;
  vector<cv::Mat> imgs, depths;
  readConfig();
  readData(pcs, imgs, depths);

  cout << "fuck1" << endl;
  // tgt_cloud = *pcs[0];
  // src_cloud = *pcs[1];

  pcl::VoxelGrid<pcl::PointXYZRGB> grid;

  grid.setLeafSize(0.05, 0.05, 0.05);
  grid.setInputCloud(pcs[1]);
  grid.filter(src_cloud);

  grid.setInputCloud(pcs[0]);
  grid.filter(tgt_cloud);

  int N_tgt = tgt_cloud.size();
  int N_src = src_cloud.size();
  Eigen::Matrix3Xd src(3, N_src);
  Eigen::Matrix3Xd tgt(3, N_tgt);
  cout << "fuck2" << endl;
  // Eigen::MatrixXd tgt(3, N_tgt);
  // Eigen::MatrixXd src(3, N_src);

  for (int i = 0; i < N_tgt; i++)
  {
    tgt(0, i) = tgt_cloud[i].x;
    tgt(1, i) = tgt_cloud[i].y;
    tgt(2, i) = tgt_cloud[i].z;
  }
  cout << "fuck3" << endl;

  for (int i = 0; i < N_src; i++)
  {
    src(0, i) = src_cloud[i].x;
    src(1, i) = src_cloud[i].y;
    src(2, i) = src_cloud[i].z;
  }
  cout << "fuck4" << endl;
  cout << N_src << endl;
  cout << N_tgt << endl;
  pcl::io::savePCDFile("/home/tim/gg.pcd", src_cloud);

  // Run TEASER++ registration
  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = NOISE_BOUND;
  params.cbar2 = config.c1;  // 1
  params.estimate_scaling = false;
  params.rotation_max_iterations = 1000;
  params.rotation_gnc_factor = config.c2;  // 1.4
  params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = config.c3;  // 0.005
  cout << "fuck5" << endl;
  // Solve with TEASER++
  teaser::RobustRegistrationSolver solver(params);
  cout << "fuck5.5" << endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  solver.solve(src, tgt);
  cout << "fuck6" << endl;
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  auto solution = solver.getSolution();
  cout << "fuck7" << endl;
  // Compare results
  std::cout << "=====================================" << std::endl;
  std::cout << "          TEASER++ Results           " << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "Estimated rotation: " << std::endl;
  std::cout << solution.rotation << std::endl;
  std::cout << "Estimated translation: " << std::endl;
  std::cout << solution.translation << std::endl;
  std::cout << std::endl;
  std::cout << "Time taken (s): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB> new_cloud;
  Eigen::Matrix4d T;
  T.setIdentity(4, 4);
  T.topLeftCorner(3, 3) = solution.rotation;
  T.topRightCorner(3, 1) << solution.translation(0), solution.translation(1), solution.translation(2);
  cout << T << endl;
  pcl::transformPointCloud(*pcs[1], new_cloud, T);
  pcl::PointCloud<pcl::PointXYZRGB> origin = *pcs[0];
  origin += new_cloud;
  pcl::io::savePCDFile("/home/tim/icp.pcd", origin);
}