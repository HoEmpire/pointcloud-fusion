#include <iostream>

#include <pcl/filters/statistical_outlier_removal.h>  //统计滤波器头文件
#include <pcl/visualization/cloud_viewer.h>
#include "icp.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)  //设置背景颜色
{
  viewer.setBackgroundColor(1.0f, 0.7f, 1.0f);
}

int main(int argv, char** argc)
{
  readConfig();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + "10.pcd", *pc);

  ///////****************************************************////////////////////
  /*方法三：统计滤波器滤波*/
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after_StatisticalRemoval(new pcl::PointCloud<pcl::PointXYZRGB>);  //

  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> Statistical;  //创建滤波器对象
  Statistical.setInputCloud(pc);                                 //设置待滤波的点云
  Statistical.setMeanK(50);                                      //取平均值的临近点数
  Statistical.setStddevMulThresh(1);  //设置是否为离群点的阈值：超过平均距离一个标准差以上，该点记为离群点，将其移除
  Statistical.filter(*pc);  //执行滤波处理，保存内点到cloud_after_StatisticalRemoval

  std::cout << "统计分析滤波后点云数据点数：" << cloud_after_StatisticalRemoval->points.size() << std::endl;
  ///////****************************************************////////////////////

  pcl::visualization::CloudViewer viewer("cloud_after_Radius");
  viewer.showCloud(pc);
  viewer.runOnVisualizationThreadOnce(viewerOneOff);  //该注册函数在渲染输出是每次都调用

  while (!viewer.wasStopped())
  {
  }
}