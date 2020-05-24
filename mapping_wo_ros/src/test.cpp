#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
  cv::Mat depth_map = cv::Mat::zeros(2, 2, CV_16UC1);
  float depth = 35000.0;
  depth_map.at<ushort>(1, 1) = ushort(depth);
  cv::imwrite("/home/tim/test.png", depth_map);
}