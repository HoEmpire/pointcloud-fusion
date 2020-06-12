#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/robust_kernel.h>
// #include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include "icp.h"
#include "util.h"

using namespace std;

int main(int argv, char **argc)
{
  ifstream fin("./result.txt");

  // 设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;  // 图模型
  optimizer.setAlgorithm(solver);  // 设置求解器
  optimizer.setVerbose(true);      // 打开调试输出

  int vertexCnt = 0, edgeCnt = 0;  // 顶点和边的数量
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
      v->setEstimateFromSE3Quat(g2o::SE3Quat(q, Vector3d(data[0], data[1], data[2])));

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
      e->setMeasurement(g2o::SE3Quat(q, Vector3d(data[0], data[1], data[2])));
      if (idx1 < idx2)
        e->setInformation(Eigen::MatrixXd::Identity(6, 6));
      else
        e->setInformation(Eigen::MatrixXd::Identity(6, 6));
      // e->setRobustKernel(new g2o::RobustKernelHuber());
      optimizer.addEdge(e);
    }
    if (!fin.good())
      break;
  }

  cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

  readConfig();
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
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

  cout << "optimizing ..." << endl;
  optimizer.initializeOptimization();
  optimizer.optimize(30);

  cout << "saving optimization results ..." << endl;
  optimizer.save("result.g2o");

  for (int i = 0; i < data_len; i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + to_string(i) + ".pcd", *tmp);
    pcs.push_back(tmp);
  }
  cout << "Loading point clouds finished!" << endl;

  pcl::PointCloud<pcl::PointXYZRGB> origin;
  for (int i = 0; i < data_len; i++)
  {
    Eigen::Matrix4d final_T;
    g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    final_T = v->estimate().matrix().inverse();
    cout << "Pose=" << endl << final_T << endl;

    pcl::transformPointCloud(*pcs[i], *pcs[i], final_T);
    origin += *pcs[i];
  }

  pcl::io::savePCDFile("/home/tim/ndt_lc.pcd", origin);
}