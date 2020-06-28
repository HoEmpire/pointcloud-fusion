#include "optimization.h"

optimization::optimization()
{
  // 设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;  // 图模型
  optimizer.setAlgorithm(solver);  // 设置求解器
  optimizer.setVerbose(true);      // 打开调试输出
  num_vertex = 0;
  num_edge = 0;
}

void optimization::addVertexs(vector<Matrix4d> T_vertexs)
{
  for (int i = 0; i < T_vertexs.size(); i++)
  {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(i);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
    T.rotate(T_vertexs[i]);                               // 按照rotation_vector进行旋转
    T.pretranslate(T_vertexs[i]);                         // 把平移向量设成(1,3,4)
    v->setEstimate(T);
    optimizer.addVertex(v);
    num_vertex++;
    if (i == 0)
      v->setFixed(true);
  }
}

void optimization::addEdges(vector<Matrix4d> T_edges)
{
  for (int i = 0; i < T_edges.size(); i++)
  {
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setId(num_edge++);
    e->setVertex(0, optimizer.vertices()[i]);
    e->setVertex(1, optimizer.vertices()[i + 1]);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
    T.rotate(T_edges[i]);                                 // 按照rotation_vector进行旋转
    T.pretranslate(T_edges[i]);                           // 把平移向量设成(1,3,4)
    e->setMeasurement(T);
    Eigen::MatrixXd information_matrix(6, 6);
    information_matrix.setIdentity(6, 6);
    e->setInformation(information_matrix);
    // e->setRobustKernel(new g2o::RobustKernelHuber());
    edges.push_back(e);
    optimizer.addEdge(e);
  }
}

void optimization::addLoops(vector<Matrix4d> T_loops, vector<vector<int>> loops)
{
  for (int i = 0; i < T_loops.size(); i++)
  {
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setId(num_edge++);
    e->setVertex(0, optimizer.vertices()[loops[i][0]]);
    e->setVertex(1, optimizer.vertices()[loops[i][1]]);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
    T.rotate(T_loops[i]);                                 // 按照rotation_vector进行旋转
    T.pretranslate(T_loops[i]);                           // 把平移向量设成(1,3,4)
    e->setMeasurement(T);
    Eigen::MatrixXd information_matrix(6, 6);
    information_matrix.setIdentity(6, 6);
    e->setInformation(information_matrix);
    // e->setRobustKernel(new g2o::RobustKernelHuber());
    edges.push_back(e);
    optimizer.addEdge(e);
  }
}

void optimization::optimize(int num_iteration = 30)
{
  cout << "read total " << num_vertex << " vertices, " << num_edge << " edges." << endl;
  optimizer.initializeOptimization();
  cout << "optimizing ..." << endl;
  optimizer.optimize(num_iteration);
  cout << "saving optimization results ..." << endl;
  optimizer.save("result.g2o");
}

vector<Matrix4d> optimization::getResult()
{
  vector<Eigen::Matrix4d> T_result;
  for (int i = 0; i < num_vertex; i++)
  {
    g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    T_result.push_back(v->estimate().matrix());
  }
  return T_result;
}