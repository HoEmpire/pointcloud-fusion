#pragma once
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/robust_kernel.h>
// #include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include "util.h"

using namespace std;

class optimization
{
public:
  optimization();

  ~optimization() = default;

  void addVertexs(vector<Matrix4d> T_vertexs);

  void addEdges(vector<Matrix4d> T_edges);

  void addLoops(vector<Matrix4d> T_loops, vector<vector<int>> loops);

  void optimize(int num_iteration);

  vector<Matrix4d> getResult();

  g2o::SparseOptimizer optimizer;

  int num_vertex, num_edge;

  vector<g2o::EdgeSE3 *> edges;
};
