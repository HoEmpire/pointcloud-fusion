#pragma once
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/eigen.hpp"
#include "util.h"
using namespace std;
using namespace cv;
using namespace Eigen;

void findFeatureMatches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1,
                        std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches)
{
  //-- 初始化
  Mat descriptors_1, descriptors_2;

  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++)
  {
    double dist = match[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  //   printf("-- Max dist : %f \n", max_dist);
  //   printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= max(2 * min_dist, 30.0))
    {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
  return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void poseEstimation2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                        std::vector<DMatch> matches, Mat &R, Mat &t)
{
  // 相机内参,TUM Freiburg2
  Mat K;
  Matrix3f camera_matrix = config.camera_matrix.topLeftCorner(3, 3);
  eigen2cv(camera_matrix, K);

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++)
  {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算本质矩阵
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, K, FM_LMEDS);

  //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  recoverPose(essential_matrix, points1, points2, K, R, t);
}

vector<Matrix4f> calVisualOdometry(vector<Mat> imgs, vector<Mat> depths)
{
  vector<Matrix4f> T_between_frames;
  if (imgs.size() <= 1)
    cout << "WARNING:The numebr of images is less than 1 !" << endl;
  for (int i = 1; i < imgs.size(); i++)
  {
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    findFeatureMatches(imgs[i - 1], imgs[i], keypoints_1, keypoints_2, matches);
    cout << "*****计算第" << i << "组图片*****" << endl;
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //对极几何估计两张图像间运动
    Mat R, t;
    poseEstimation2d2d(keypoints_1, keypoints_2, matches, R, t);

    int count = 0;
    for (vector<DMatch>::iterator m = matches.begin(); m != matches.end(); m++)
    {
      KeyPoint kp1, kp2;
      kp1 = keypoints_1[m->queryIdx];
      kp2 = keypoints_2[m->trainIdx];
      if (depths[i - 1].at<ushort>(kp1.pt.y, kp1.pt.x) != 0 || depths[i].at<ushort>(kp2.pt.y, kp2.pt.x) != 0)
        count++;
    }
    cout << "能够恢复绝对尺度的点数: " << count << endl;

    Matrix3f R_eigen;
    Vector3f t_eigen;
    Matrix4f T;
    cv2eigen(R, R_eigen);
    cv2eigen(t, t_eigen);
    T.setIdentity(4, 4);
    T.topLeftCorner(3, 3) = R_eigen;
    // T.topRightCorner(3,1) = t_eigen;
    T = config.extrinsic_matrix.inverse().cast<float>() * T.inverse().cast<float>() *
        config.extrinsic_matrix.cast<float>();

    // show in euler angle
    cout << "T:" << endl << T << endl;
    Vector3f euler_angle = rotationMatrixToEulerAngles(T.topLeftCorner(3, 3)) * 180 / PI;
    cout << "euler anles = " << euler_angle.transpose() << endl;
    cout << "*******************" << endl;
    T_between_frames.push_back(T);
  }
  return T_between_frames;
}
