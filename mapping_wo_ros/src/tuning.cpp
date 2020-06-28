#include "icp.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
int main(int argv, char **argc)
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  struct imageType image_data;
  struct timer t;

  readConfig();
  t.tic();
  readDataWithID(pcs, image_data);
  cout << "Reading data takes " << t.toc() << " seconds." << endl;

  t.tic();
  image_data.init();
  cout << "Extracting features takes " << t.toc() << " seconds." << endl;

  t.tic();
  struct pointcloudType pc_data(pcs);
  pc_data.filter(config.filter_std_threshold, config.frame_distance_threshold);
  pc_data.resample(config.point_cloud_resolution);
  cout << "Preprocessing point clouds takes " << t.toc() << " seconds." << endl;

  t.tic();
  vector<Matrix4d> T_init = calVisualOdometry(image_data);
  cout << "Calculating visual odometry takes " << t.toc() << " seconds." << endl;

  t.tic();
  // icpNonlinearWithNormal(pcs, T_init);
  vector<Matrix4d> T_result;
  ndtRegistration(pc_data, T_init, T_result);
  cout << "Registing point cloud by NDT takes " << t.toc() << " seconds." << endl;

  vector<vector<int>> loops;
  loop_closure(image_data, loops);
}
