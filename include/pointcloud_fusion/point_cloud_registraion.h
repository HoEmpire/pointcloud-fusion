#include "data_io.h"
#include "icp.h"
#include "optimization.h"
#include "type.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void pointCloudRegistration(struct imageType image_data, struct pointcloudType pc_data,
                            string path = "/home/tim/test.pcd")
{
  struct timer t;

  t.tic();
  image_data.init();
  cout << "Extracting features takes " << t.toc() << " seconds." << endl;

  t.tic();

  pc_data.depthFilter(config.depth_filter_ratio);
  pc_data.standardFilter(config.filter_std_threshold, config.frame_distance_threshold);
  image_data.depths = pc_data.paintPointCloud(image_data.imgs);
  pc_data.resample(config.point_cloud_resolution);

  cout << "Preprocessing point clouds and get depth maps takes " << t.toc() << " seconds." << endl;

  t.tic();
  vector<Matrix4f> T_init = calVisualOdometry(image_data);
  cout << "Calculating visual odometry takes " << t.toc() << " seconds." << endl;

  t.tic();
  // icpNonlinearWithNormal(pcs, T_init);
  vector<Matrix4f> T_pc;  // transformation between two near frame
  ndtRegistration(pc_data, T_init, T_pc);

  Matrix4f tmp_pos;
  tmp_pos.setIdentity(4, 4);
  vector<Matrix4f> T_vertex;  // transformation between current frame and world frame
  T_vertex.push_back(tmp_pos);
  for (int i = 0; i < T_pc.size(); i++)
  {
    tmp_pos = tmp_pos * T_pc[i];
    T_vertex.push_back(tmp_pos);
  }
  cout << "Registing point cloud by NDT takes " << t.toc() << " seconds." << endl;

  vector<vector<int>> loops, loops_good;
  vector<Matrix4f> T_loops, T_vo, T_result, T_final;
  loopClosure(image_data, loops, T_vo);
  calLoopsTransform(pc_data, loops, T_vo, T_vertex, loops_good, T_loops);

  optimization opt;
  opt.addVertexs(T_vertex);
  opt.addEdges(T_pc);
  opt.addLoops(T_loops, loops_good);
  opt.optimize();
  T_final = opt.getResult();
  savePointCloud(pc_data, T_final, path);
}
