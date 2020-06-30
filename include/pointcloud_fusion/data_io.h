#pragma once
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include "type.h"
#include "util.h"
using namespace std;

void savePointCloud(struct pointcloudType pc_data, vector<Matrix4f> T_final, string filepath)
{
  pcl::PointCloud<pcl::PointXYZRGB> origin;
  for (int i = 0; i < pc_data.pc_filtered.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB> tmp;
    pcl::transformPointCloud(*pc_data.pc_filtered[i], tmp, T_final[i]);
    origin += tmp;
  }
  pcl::io::savePCDFile(filepath, origin);
}