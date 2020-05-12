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

  vector<Matrix4f> T_init = cal_visual_odometry(imgs);
  icp_nonlinear_with_normal(pcs, T_init);
}