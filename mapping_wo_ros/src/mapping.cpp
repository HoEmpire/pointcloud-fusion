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
  vector<Mat> imgs;
  readConfig();
  readData(pcs, imgs);

  vector<Matrix4f> T_init = calVisualOdometry(imgs);
  icpNonlinearWithNormal(pcs, T_init);
}