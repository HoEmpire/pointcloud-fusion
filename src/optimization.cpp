#include "pointcloud_fusion/optimization.h"

optimization::optimization()
{
  // 设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  optimizer.setAlgorithm(solver);  // 设置求解器
  optimizer.setVerbose(true);      // 打开调试输出
  num_vertex = 0;
  num_edge = 0;
}

void optimization::addVertexs(vector<Matrix4f> T_vertexs)
{
  for (int i = 0; i < T_vertexs.size(); i++)
  {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(i);

    Matrix3d rotation = T_vertexs[i].topLeftCorner(3, 3).cast<double>();
    Vector3d translation = T_vertexs[i].topRightCorner(3, 1).cast<double>();
    v->setEstimate(g2o::SE3Quat(rotation, translation));

    optimizer.addVertex(v);
    num_vertex++;
    if (i == 0)
      v->setFixed(true);
  }
}

void optimization::addEdges(vector<Matrix4f> T_edges)
{
  for (int i = 0; i < T_edges.size(); i++)
  {
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setId(num_edge++);
    e->setVertex(0, optimizer.vertices()[i]);
    e->setVertex(1, optimizer.vertices()[i + 1]);

    Matrix3d rotation = T_edges[i].topLeftCorner(3, 3).cast<double>();
    Vector3d translation = T_edges[i].topRightCorner(3, 1).cast<double>();
    e->setMeasurement(g2o::SE3Quat(rotation, translation));

    Eigen::MatrixXd information_matrix(6, 6);
    information_matrix.setIdentity(6, 6);
    e->setInformation(information_matrix);
    // e->setRobustKernel(new g2o::RobustKernelHuber());
    edges.push_back(e);
    optimizer.addEdge(e);
  }
}

void optimization::addLoops(vector<Matrix4f> T_loops, vector<vector<int>> loops)
{
  for (int i = 0; i < T_loops.size(); i++)
  {
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setId(num_edge++);
    e->setVertex(0, optimizer.vertices()[loops[i][0]]);
    e->setVertex(1, optimizer.vertices()[loops[i][1]]);

    Matrix3d rotation = T_loops[i].topLeftCorner(3, 3).cast<double>();
    Vector3d translation = T_loops[i].topRightCorner(3, 1).cast<double>();
    e->setMeasurement(g2o::SE3Quat(rotation, translation));

    Eigen::MatrixXd information_matrix(6, 6);
    information_matrix.setIdentity(6, 6);
    e->setInformation(information_matrix);
    // e->setRobustKernel(new g2o::RobustKernelHuber());
    edges.push_back(e);
    optimizer.addEdge(e);
  }
}

void optimization::optimize(int num_iteration)
{
  cout << "read total " << num_vertex << " vertices, " << num_edge << " edges." << endl;
  optimizer.initializeOptimization();
  cout << "optimizing ..." << endl;
  // cout << num_iteration << endl;
  // for (int i = 0; i < edges.size(); i++)
  // {
  //   cout << edges[i]->id() << endl << edges[i]->error().matrix() << endl;
  //   edges[i]->computeError();
  //   cout << edges[i]->id() << endl << edges[i]->error().matrix() << endl;
  // }
  optimizer.optimize(num_iteration);
  cout << "saving optimization results ..." << endl;
  optimizer.save("result.g2o");
}

vector<Matrix4f> optimization::getResult()
{
  vector<Eigen::Matrix4f> T_result;
  for (int i = 0; i < num_vertex; i++)
  {
    g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    T_result.push_back(v->estimate().matrix().cast<float>());
  }
  return T_result;
}