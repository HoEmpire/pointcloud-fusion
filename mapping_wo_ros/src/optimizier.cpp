#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/robust_kernel.h>
// #include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include "data_io.h"
#include "icp.h"
#include "util.h"

using namespace std;

int main(int argv, char **argc)
{
  vector<g2o::EdgeSE3 *> edges;

  // 设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;  // 图模型
  optimizer.setAlgorithm(solver);  // 设置求解器
  optimizer.setVerbose(true);      // 打开调试输出

  int vertexCnt = 0, edgeCnt = 0;  // 顶点和边的数量

  ifstream fin("./result.txt");
  while (!fin.eof())
  {
    string name;
    fin >> name;
    if (name == "VERTEX:")
    {
      // SE3 顶点
      g2o::VertexSE3 *v = new g2o::VertexSE3();
      int index = 0;
      fin >> index;
      v->setId(index);

      double data[7];
      for (double &d : data)
        fin >> d;
      Quaterniond q(data[6], data[3], data[4], data[5]);
      q.normalize();
      v->setEstimate(g2o::SE3Quat(q, Vector3f(data[0], data[1], data[2]).matrix()));

      optimizer.addVertex(v);
      vertexCnt++;
      if (index == 0)
        v->setFixed(true);
    }
    else if (name == "EDGE:")
    {
      // SE3-SE3 边
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      int idx1, idx2;  // 关联的两个顶点
      fin >> idx1 >> idx2;
      e->setId(edgeCnt++);
      e->setVertex(0, optimizer.vertices()[idx1]);
      e->setVertex(1, optimizer.vertices()[idx2]);

      double data[7];
      for (double &d : data)
        fin >> d;
      Quaterniond q(data[6], data[3], data[4], data[5]);
      q.normalize();
      e->setMeasurement(g2o::SE3Quat(q, Vector3f(data[0], data[1], data[2])));
      Eigen::MatrixXd information_matrix(6, 6);
      information_matrix.setIdentity(6, 6);
      if (idx1 < idx2)
        e->setInformation(information_matrix);
      else
        e->setInformation(information_matrix);
      // e->setRobustKernel(new g2o::RobustKernelHuber());
      edges.push_back(e);
      optimizer.addEdge(e);
    }
    if (!fin.good())
      break;
  }
  optimizer.initializeOptimization();

  cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

  for (int i = 0; i < edges.size(); i++)
  {
    cout << edges[i]->id() << endl << edges[i]->error().matrix() << endl;
    edges[i]->computeError();
    cout << edges[i]->id() << endl << edges[i]->error().matrix() << endl;
  }

  cout << "optimizing ..." << endl;

  optimizer.optimize(30);

  // const int num_iteration = 10;

  // for (int iter = 0; iter < num_iteration; iter++)
  // {
  //   // adjust weight of bad edges
  //   vector<double> errors;
  //   vector<int> edge_inliers_id_tmp;
  //   vector<int> edge_inliers_id_best;
  //   for (int i = 0; i < edges.size(); i++)
  //   {
  //     cout << edges[i]->id() << " " << edges[i]->computeError() << endl;
  //     errors.push_back(edges[i]->chi2());
  //   }

  //   for (int iter_ransac = 0; iter_ransac < num_iteration; iter_ransac++)
  //   {
  //     int rand_index = rand() % edges.size();
  //     vector<int>().swap(edge_inliers_id_tmp);
  //     for (int i = 0; i < edges.size(); i++)
  //     {
  //       if (abs(errors[i] - errors[rand_index]) / (abs(errors[i]) + 1e-9) < 0.5)
  //         edge_inliers_id_tmp.push_back(i);
  //     }
  //     if (iter_ransac == 0 || edge_inliers_id_best.size() < edge_inliers_id_tmp.size())
  //       edge_inliers_id_best.assign(edge_inliers_id_tmp.begin(), edge_inliers_id_tmp.end());
  //   }

  //   optimizer.clear();

  //   cout << "optimizing ..." << endl;
  //   optimizer.initializeOptimization();
  //   optimizer.optimize(30);
  // }

  cout << "saving optimization results ..." << endl;
  optimizer.save("result.g2o");

  readConfig();
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  struct imageType image_data;
  struct timer t;

  t.tic();
  readData(pcs, image_data);
  cout << "Reading data takes " << t.toc() << " seconds." << endl;

  t.tic();
  struct pointcloudType pc_data(pcs);
  pc_data.standardFilter(config.filter_std_threshold, config.frame_distance_threshold);
  pc_data.resample(config.point_cloud_resolution);
  cout << "Preprocessing point clouds takes " << t.toc() << " seconds." << endl;

  pcl::PointCloud<pcl::PointXYZRGB> origin;
  for (int i = 0; i < pc_data.pc_filtered.size(); i++)
  {
    Eigen::Matrix4f final_T;
    g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    final_T = v->estimate().matrix();
    cout << "Pose=" << endl << final_T << endl;
    pcl::PointCloud<pcl::PointXYZRGB> tmp;
    pcl::transformPointCloud(*pc_data.pc_filtered[i], tmp, final_T);
    origin += tmp;
  }

  pcl::io::savePCDFile("/home/tim/ndt_lc.pcd", origin);

  // Eigen::Matrix4f final_T_edge;
  // final_T_edge.setIdentity(4, 4);
  // pcl::PointCloud<pcl::PointXYZRGB> origin_edge;
  // origin_edge = *pcs[0];
  // for (int i = 1; i < data_len; i++)
  // {
  //   final_T_edge = final_T_edge * edges[i - 1]->measurement().matrix().inverse();
  //   cout << "Pose=" << endl << final_T_edge << endl;
  //   pcl::PointCloud<pcl::PointXYZRGB> tmp;
  //   pcl::transformPointCloud(*pcs[i], tmp, final_T_edge);
  //   origin_edge += tmp;
  // }
  // pcl::io::savePCDFile("/home/tim/ndt_edge.pcd", origin_edge);
}