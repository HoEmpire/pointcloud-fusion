#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "DBoW3/DBoW3.h"

#include "icp.h"
#include "util.h"
#include "visual_odometry.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main(int argc, char** argv)
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  struct imageType image_data;
  readConfig();
  readData(pcs, image_data);
  image_data.init();

  // create vocabulary
  cout << "creating vocabulary ... " << endl;
  DBoW3::Vocabulary vocab;
  vocab.create(image_data.descriptors);
  cout << "vocabulary info: " << vocab << endl;
  vocab.save("vocab_sift.yml.gz");
  cout << "done" << endl;

  return 0;
}