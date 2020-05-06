#include <iostream>
#include "util.h"
int main(int argv, char **argc)
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  std::vector<cv::Mat> imgs;
  readConfig();
  readData(pcs, imgs);
}