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
  struct pointcloudType pc_data(pcs);
  // pc_data.filter(config.filter_std_threshold, config.frame_distance_threshold);
  pc_data.depthFilter();
  // pc_data.resample(config.point_cloud_resolution);
  pc_data.pc_filtered[0]->width = pc_data.pc_filtered[0]->size();
  pc_data.pc_filtered[0]->height = 1;
  pc_data.standardFilter(config.filter_meanK, config.filter_std_threshold);
  cout << "Preprocessing point clouds takes " << t.toc() << " seconds." << endl;

  pcl::io::savePCDFile("/home/tim/filtered_pc.pcd", *pc_data.pc_filtered[0]);
}
