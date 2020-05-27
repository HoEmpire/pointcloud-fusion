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

void findFeatureMatches(const Mat &img_1, const Mat &img_2, vector<KeyPoint> &keypoints_1,
                        vector<KeyPoint> &keypoints_2, vector<DMatch> &matches)
{
  //-- 初始化
  Mat descriptors_1, descriptors_2;

  Ptr<FeatureDetector> detector = ORB::create(1000);
  Ptr<DescriptorExtractor> descriptor = ORB::create(1000);
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
  return Point2d((p.x - K.at<float>(0, 2)) / K.at<float>(0, 0), (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1));
}

void poseEstimation2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat &R,
                        Mat &t, vector<DMatch> &inliers)
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
  Mat mask;
  inliers.clear();
  essential_matrix = findEssentialMat(points1, points2, K, FM_RANSAC, 0.999, 0.8, mask);
  for (int i = 0; i < mask.rows; i++)
  {
    if (mask.at<uchar>(i, 0) == 1)
      inliers.push_back(matches[i]);
  }

  //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  recoverPose(essential_matrix, points1, points2, K, R, t);
}

inline bool isZero(float d)
{
  return abs(d) < 1e-6;
}

float computeScaleFromPoint(Vector3f x1, Vector3f x2, float d1, float d2, Matrix3f R, Vector3f t)
{
  MatrixXf A(3, 1);
  Vector3f b;
  int flag = -1;
  if (!isZero(d1) && isZero(d2))
  {
    flag = 1;
    A = hat(x2) * t;
    b = d1 * hat(x2) * R * x1;
  }
  else if (isZero(d1) && !isZero(d2))
  {
    flag = 2;
    A = hat(x1) * R.inverse() * t;
    b = hat(x1) * R.inverse() * x2 * d2;
  }
  else if (!isZero(d1) && !isZero(d2))
  {
    flag = 3;
    A = t;
    b = d2 * x2 - d1 * R * x1;
  }
  cout << "Recover Svale::Point type: " << flag << endl;
  // cout << d1 << endl;
  // cout << d2 << endl;
  // cout << x1 << endl;
  // cout << x2 << endl;
  // cout << R << endl;
  // cout << t << endl;
  // cout << A << endl;
  // cout << b << endl;
  return leastSquareMethod(A, b)(0, 0);
}

float recoverScale(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> inliers_with_depth,
                   vector<float> depth1, vector<float> depth2, Matrix3f R, Vector3f t, Mat K)
{
  vector<float> scale;
  for (int i = 0; i < inliers_with_depth.size(); i++)
  {
    Vector3f x1, x2;
    Point2d p1, p2;
    p1 = pixel2cam(keypoints_1[inliers_with_depth[i].queryIdx].pt, K);
    p2 = pixel2cam(keypoints_2[inliers_with_depth[i].trainIdx].pt, K);
    x1 << float(p1.x), float(p1.y), 1;
    x2 << float(p2.x), float(p2.y), 1;
    float tmp_scale = computeScaleFromPoint(x1, x2, depth1[i], depth2[i], R, t);
    cout << "Recover Svale::Scale: " << tmp_scale << endl;
    scale.push_back(tmp_scale);
  }
  float average_scale = std::accumulate(scale.begin(), scale.end(), 0.0) / scale.size();
  cout << "Recover Svale::Finish recover scale!!!! " << endl;
  cout << "Recover Svale::Average Scale: " << average_scale << endl;
}

vector<Matrix4f> calVisualOdometry(vector<Mat> imgs, vector<Mat> depths)
{
  vector<Matrix4f> T_between_frames;
  if (imgs.size() <= 1)
    cout << "WARNING:The numebr of images is less than 1 !" << endl;
  for (int i = 1; i < imgs.size(); i++)
  {
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches, inliers;
    findFeatureMatches(imgs[i - 1], imgs[i], keypoints_1, keypoints_2, matches);
    cout << "*****计算第" << i << "组图片*****" << endl;
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //对极几何估计两张图像间运动
    Mat R, t;
    poseEstimation2d2d(keypoints_1, keypoints_2, matches, R, t, inliers);

    int count = 0;
    vector<DMatch> inliers_with_depth;
    vector<float> depth1, depth2;
    for (vector<DMatch>::iterator m = inliers.begin(); m != inliers.end(); m++)
    {
      KeyPoint kp1, kp2;
      kp1 = keypoints_1[m->queryIdx];
      kp2 = keypoints_2[m->trainIdx];
      ushort d1, d2;
      d1 = depths[i - 1].at<ushort>(kp1.pt.y, kp1.pt.x);
      d2 = depths[i].at<ushort>(kp2.pt.y, kp2.pt.x);
      if (d1 != 0 || d2 != 0)
      {
        count++;
        inliers_with_depth.push_back(*m);
        depth1.push_back(float(d1 / 10000.0));
        depth2.push_back(float(d2 / 10000.0));
      }
    }
    cout << "能够恢复绝对尺度的点数: " << count << endl;

    Matrix3f R_eigen;
    Vector3f t_eigen;
    Matrix4f T;
    cv2eigen(R, R_eigen);
    cv2eigen(t, t_eigen);

    Mat K;
    Matrix3f camera_matrix = config.camera_matrix.topLeftCorner(3, 3);
    eigen2cv(camera_matrix, K);
    float scale = recoverScale(keypoints_1, keypoints_2, inliers_with_depth, depth1, depth2, R_eigen, t_eigen, K);

    T.setIdentity(4, 4);
    T.topLeftCorner(3, 3) = R_eigen;
    T.topRightCorner(3, 1) = t_eigen * scale;
    T = config.extrinsic_matrix.inverse() * T.inverse() * config.extrinsic_matrix;

    // show in euler angle
    cout << "T:" << endl << T << endl;
    Vector3f euler_angle = rotationMatrixToEulerAngles(T.topLeftCorner(3, 3)) * 180 / PI;
    cout << "euler anles = " << euler_angle.transpose() << endl;
    float angles =
        sqrt(euler_angle[0] * euler_angle[0] + euler_angle[0] * euler_angle[0] + euler_angle[0] * euler_angle[0]);
    if (angles > 30)  // TODO hardcode in here
    {
      cout << "WARNING:Visual odometry degenerate!!" << endl;
      T.setIdentity(4, 4);
    }

    cout << "*******************" << endl;
    T_between_frames.push_back(T);
  }
  return T_between_frames;
}
