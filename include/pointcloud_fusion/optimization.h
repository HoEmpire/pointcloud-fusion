#pragma once
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/robust_kernel.h>
// #include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

using namespace std;
using namespace Eigen;

class optimization
{
public:
  optimization();

  ~optimization() = default;

  void addVertexs(vector<Matrix4f> T_vertexs);

  void addEdges(vector<Matrix4f> T_edges);

  void addLoops(vector<Matrix4f> T_loops, vector<vector<int>> loops);

  void optimize(int num_iteration = 30);

  vector<Matrix4f> getResult();

  g2o::SparseOptimizer optimizer;

  int num_vertex, num_edge;

  vector<g2o::EdgeSE3 *> edges;
};
