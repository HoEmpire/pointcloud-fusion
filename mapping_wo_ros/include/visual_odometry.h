#pragma once
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "DBoW3/DBoW3.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "type.h"
#include "util.h"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace Eigen;

void findFeatureMatches(const Mat &descriptors_1, const Mat &descriptors_2, vector<DMatch> &matches)
{
  //-- 初始化

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

  vector<vector<DMatch>> knn_matches;
  matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;

  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      matches.push_back(knn_matches[i][0]);
    }
  }
}

Point2f pixel2cam(const Point2f &p, const Mat &K)
{
  return Point2f((p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
                 (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1));  // without unit
}

void poseEstimation2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat &R,
                        Mat &t, vector<DMatch> &inliers)
{
  // 相机内参,TUM Freiburg2
  Mat K;
  Matrix3d camera_matrix = config.camera_matrix.topLeftCorner(3, 3);
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

float computeScaleFromPoint(Vector3d x1, Vector3d x2, float d1, float d2, Matrix3d R, Vector3d t)
{
  MatrixXd A(3, 1);
  Vector3d b;
  int flag = -1;
  if (!isZero(d1) && isZero(d2))
  {
    flag = 1;
    A = hat(x2) * t;
    b = -d1 * hat(x2) * R * x1;
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
    A = hat(x2) * t;
    b = -d1 * hat(x2) * R * x1;
  }
  // cout << endl;
  // cout << "Recover Svale::Point type: " << flag << endl;
  // cout << "Recover Svale::x1,x2: " << endl;
  // cout << d1 << endl;
  // cout << d2 << endl;
  // cout << x1 << endl;
  // cout << x2 << endl;
  return leastSquareMethod(A, b)(0, 0);
}

float recoverScale(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> inliers_with_depth,
                   vector<float> depth1, vector<float> depth2, Matrix3d R, Vector3d t, Mat K, const int max_iter = 20,
                   const float threshold = 0.5)
{
  vector<float> scale, scale_inliers_best, scale_inliers_tmp;
  for (int i = 0; i < inliers_with_depth.size(); i++)
  {
    Vector3d x1, x2;
    Point2f p1, p2;
    p1 = pixel2cam(keypoints_1[inliers_with_depth[i].queryIdx].pt, K);
    p2 = pixel2cam(keypoints_2[inliers_with_depth[i].trainIdx].pt, K);
    x1 << float(p1.x), float(p1.y), 1;
    x2 << float(p2.x), float(p2.y), 1;
    float tmp_scale = computeScaleFromPoint(x1, x2, depth1[i], depth2[i], R, t);
    // cout << "Recover Svale::Scale: " << tmp_scale << endl;
    scale.push_back(tmp_scale);
  }

  for (int iter = 0; iter < max_iter; iter++)
  {
    int rand_index = rand() % scale.size();
    vector<float>().swap(scale_inliers_tmp);
    for (int i = 0; i < scale.size(); i++)
    {
      if (abs(scale[i] - scale[rand_index]) / (abs(scale[i]) + 1e-9) < threshold)
        scale_inliers_tmp.push_back(scale[i]);
    }
    if (iter == 0 || scale_inliers_best.size() < scale_inliers_tmp.size())
      scale_inliers_best.assign(scale_inliers_tmp.begin(), scale_inliers_tmp.end());
  }

  cout << "Recover Svale::After RANSAC " << scale_inliers_best.size() << " sets of matches left" << endl;
  float average_scale;
  if (scale_inliers_best.size() >= 10 || 1.0 * scale_inliers_best.size() / (scale.size() + 0.1) > 0.4)
    average_scale =
        std::accumulate(scale_inliers_best.begin(), scale_inliers_best.end(), 0.0) / scale_inliers_best.size();
  else
    average_scale = 0.0;
  cout << "Recover Svale::Finish recover scale!!!! " << endl;
  cout << "Recover Svale::Average Scale: " << average_scale << endl;
  return average_scale;
}

vector<Matrix4d> calVisualOdometry(struct imageType image_data)
{
  vector<Matrix4d> T_between_frames;
  if (image_data.imgs.size() <= 1)
    cout << "WARNING:The numebr of images is less than 1 !" << endl;
  for (int i = 1; i < image_data.imgs.size(); i++)
  {
    vector<DMatch> matches, inliers;
    findFeatureMatches(image_data.descriptors[i - 1], image_data.descriptors[i], matches);
    cout << "*****计算第" << i << "组图片*****" << endl;
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat K;
    Matrix3d camera_matrix = config.camera_matrix.topLeftCorner(3, 3);
    eigen2cv(camera_matrix, K);

    int count = 0;
    int count_PnP = 0;
    vector<DMatch> inliers_with_depth;
    vector<float> depth1, depth2;
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    Mat R, t;
    poseEstimation2d2d(image_data.keypoints[i - 1], image_data.keypoints[i], matches, R, t, inliers);

    for (vector<DMatch>::iterator m = inliers.begin(); m != inliers.end(); m++)
    {
      KeyPoint kp1, kp2;
      kp1 = image_data.keypoints[i - 1][m->queryIdx];
      kp2 = image_data.keypoints[i][m->trainIdx];
      ushort d1, d2;
      d1 = image_data.depths[i - 1].at<ushort>(kp1.pt.y, kp1.pt.x);
      d2 = image_data.depths[i].at<ushort>(kp2.pt.y, kp2.pt.x);
      if (d1 != 0 || d2 != 0)
      {
        count++;
        inliers_with_depth.push_back(*m);
        depth1.push_back(float(d1 / 1000.0));
        depth2.push_back(float(d2 / 1000.0));
      }

      if (d1 != 0)
      {
        count_PnP++;
        float dd = float(d1 / 1000.0);
        Point2f p1 = pixel2cam(kp1.pt, K);
        // cout << Point3f(p1.x * dd, p1.y * dd, dd).x << " " << Point3f(p1.x * dd, p1.y * dd, dd).y << " "
        //      << Point3f(p1.x * dd, p1.y * dd, dd).z << endl;
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(kp2.pt);
      }
    }
    cout << "能够恢复绝对尺度的点数: " << count << endl;
    cout << "能够用PnP绝对尺度的点数: " << count_PnP << endl;

    float scale;

    Matrix3d R_eigen;
    Vector3d t_eigen;

    if (count_PnP > 15)  // threshold for choosing PnP or epipolar search for scale recovery
    {
      // PnP估计两张图像间运动
      Mat R_PnP, t_PnP, r_PnP;
      solvePnPRansac(pts_3d, pts_2d, K, Mat(), r_PnP, t_PnP);
      Rodrigues(r_PnP, R_PnP);
      // cout << R_PnP << endl;
      // cout << t_PnP << endl;
      cv2eigen(R_PnP, R_eigen);
      cv2eigen(t_PnP, t_eigen);
      cout << "finish pose estimation by PnP" << endl;
      scale = 1.0;
    }
    else
    {
      //对极几何估计两张图像间运动
      cv2eigen(R, R_eigen);
      // cout << t << endl;
      cv2eigen(t, t_eigen);
      // cout << t_eigen << endl;
      if (count != 0)
      {
        scale = recoverScale(image_data.keypoints[i - 1], image_data.keypoints[i], inliers_with_depth, depth1, depth2,
                             R_eigen, t_eigen, K);
      }
      else
        scale = 0;
      cout << "finish pose estimation by epipolar search" << endl;
    }

    Matrix4d T;
    T.setIdentity(4, 4);
    T.topLeftCorner(3, 3) = R_eigen;
    // cout << scale << endl;
    T.topRightCorner(3, 1) = t_eigen * scale;
    // cout << T << endl;
    T = config.extrinsic_matrix.inverse() * T.inverse() * config.extrinsic_matrix;

    // show in euler angle
    cout << "T:" << endl << T << endl;
    Vector3d euler_angle = rotationMatrixToEulerAngles(T.topLeftCorner(3, 3)) * 180 / PI;
    cout << "euler anles = " << euler_angle.transpose() << endl;
    float angles =
        sqrt(euler_angle[0] * euler_angle[0] + euler_angle[1] * euler_angle[1] + euler_angle[2] * euler_angle[2]);
    if (angles > 30)  // TODO hardcode in here
    {
      cout << "WARNING:Visual odometry degenerate!!" << endl;
      T.setIdentity(4, 4);
    }
    // cout << t << endl;

    cout << "*******************" << endl;
    T_between_frames.push_back(T);
  }
  return T_between_frames;
}

void loop_closure(struct imageType image_data, vector<vector<int>> &loops, vector<Matrix4d> &T_vo)
{
  // read the images and database
  cout << "LOOP CLOSURE::reading database" << endl;
  DBoW3::Vocabulary vocab("./vocab_sift.yml.gz");
  if (vocab.empty())
  {
    cerr << "LOOP CLOSURE::Vocabulary does not exist." << endl;
    return;
  }
  cout << "LOOP CLOSURE::comparing images with database " << endl;
  DBoW3::Database db(vocab, false, 0);
  for (int i = 0; i < image_data.descriptors.size(); i++)
    db.add(image_data.descriptors[i]);
  for (int i = 0; i < image_data.descriptors.size(); i++)
  {
    DBoW3::QueryResults ret;
    const int num_of_result = config.num_of_result;
    const int frame_distance_threshold = config.frame_distance_threshold;
    const float threshold = config.score_threshold;

    db.query(image_data.descriptors[i], ret, num_of_result);  // max result=4
    for (int j = 0; j < num_of_result; j++)
      if (ret[j].Score > threshold && i - int(ret[j].Id) >= frame_distance_threshold)
      {
        cout << "LOOP CLOSURE::Found loop! image " << i << " and " << ret[j].Id << " is a loop!" << endl;
        vector<int> index_tmp;
        index_tmp.push_back(i);
        index_tmp.push_back(int(ret[j].Id));
        loops.push_back(index_tmp);

        struct imageType image_data_tmp = image_data.copy_by_index(index_tmp);
        T_vo.push_back(calVisualOdometry(image_data_tmp)[0]);
      }
  }
}
