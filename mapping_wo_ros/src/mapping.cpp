#include <iostream>

#include "icp.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
int main(int argv, char **argc)
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  vector<Mat> imgs, depths;
  readConfig();
  readData(pcs, imgs, depths);

  vector<Matrix4f> T_init = calVisualOdometry(imgs, depths);
  // icpNonlinearWithNormal(pcs, T_init);
  ndtRegistration(pcs, T_init);
}