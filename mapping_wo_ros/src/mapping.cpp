#include "data_io.h"
#include "icp.h"
#include "type.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
int main(int argv, char **argc)
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  struct imageType image_data;
  struct timer t;

  readConfig();
  t.tic();
  readData(pcs, image_data);
  cout << "Reading data takes " << t.toc() << " seconds." << endl;

  t.tic();
  image_data.init();
  cout << "Extracting features takes " << t.toc() << " seconds." << endl;

  t.tic();
  struct pointcloudType pc_data(pcs);
  pc_data.depthFilter();
  pc_data.standardFilter(config.filter_std_threshold, config.frame_distance_threshold);
  pc_data.resample(config.point_cloud_resolution);
  cout << "Preprocessing point clouds takes " << t.toc() << " seconds." << endl;

  t.tic();
  vector<Matrix4d> T_init = calVisualOdometry(image_data);
  cout << "Calculating visual odometry takes " << t.toc() << " seconds." << endl;

  t.tic();
  // icpNonlinearWithNormal(pcs, T_init);
  vector<Matrix4d> T_result;
  ndtRegistration(pc_data, T_init, T_result);
  cout << "Registing point cloud by NDT takes " << t.toc() << " seconds." << endl;

  vector<vector<int>> loops;
  vector<Matrix4d> T_loops;
  loop_closure(image_data, loops, T_loops);

  ofstream outfile("./result.txt");
  Matrix4d tmp_pos;
  tmp_pos.setIdentity(4, 4);
  vector<Matrix4d> T_vertex;
  outfile << "VERTEX: 0 0.0 0.0 0.0 0.0 0.0 0.0 1.0" << endl;
  T_vertex.push_back(tmp_pos);
  for (int i = 0; i < T_result.size(); i++)
  {
    tmp_pos = tmp_pos * T_result[i];
    T_vertex.push_back(tmp_pos);
    Matrix3d rotation_matrix = tmp_pos.topLeftCorner(3, 3).cast<double>();
    Quaterniond q(rotation_matrix);
    outfile << "VERTEX: " << i + 1 << " " << tmp_pos.topRightCorner(3, 1).transpose() << " " << q.x() << " " << q.y()
            << " " << q.z() << " " << q.w() << endl;
  }

  for (int i = 0; i < T_result.size(); i++)
  {
    Matrix3d rotation_matrix = T_result[i].topLeftCorner(3, 3).cast<double>();
    Quaterniond q(rotation_matrix);
    outfile << "EDGE: " << i << " " << i + 1 << " " << T_result[i].topRightCorner(3, 1).transpose() << " " << q.x()
            << " " << q.y() << " " << q.z() << " " << q.w() << endl;
  }

  for (int i = 0; i < loops.size(); i++)
  {
    vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs_two_origin, pcs_two_filtered, pcs_two_resample;

    pcs_two_origin.push_back(pc_data.pc_origin[loops[i][0]]);
    pcs_two_origin.push_back(pc_data.pc_origin[loops[i][1]]);
    pcs_two_filtered.push_back(pc_data.pc_filtered[loops[i][0]]);
    pcs_two_filtered.push_back(pc_data.pc_filtered[loops[i][1]]);
    pcs_two_resample.push_back(pc_data.pc_resample[loops[i][0]]);
    pcs_two_resample.push_back(pc_data.pc_resample[loops[i][1]]);
    struct pointcloudType pc_data_two(pcs_two_origin, pcs_two_filtered, pcs_two_resample);

    vector<Mat> imgs, depths;
    imgs.push_back(image_data.imgs[loops[i][0]]);
    imgs.push_back(image_data.imgs[loops[i][1]]);
    depths.push_back(image_data.depths[loops[i][0]]);
    depths.push_back(image_data.depths[loops[i][1]]);

    struct imageType image_data_two;
    image_data_two.imgs = imgs;
    image_data_two.depths = depths;
    image_data_two.init();
    vector<Matrix4d> T_init_two = calVisualOdometry(image_data_two);
    vector<Matrix4d> T_result_two;
    ndtRegistration(pc_data_two, T_init_two, T_result_two);

    Matrix3d rotation_matrix = T_result_two[0].topLeftCorner(3, 3).cast<double>();
    Quaterniond q(rotation_matrix);

    // uncertainty criterion
    Matrix4d T_edge = T_vertex[loops[i][1]].inverse() * T_vertex[loops[i][0]];

    Matrix4d T_error = T_edge * T_result_two[0];
    // cout << "T edge: " << endl << T_edge << endl;
    // cout << "T_result: " << endl << T_result_two[0] << endl;
    // cout << "T_error: " << endl << T_error << endl;
    Vector3d euler_angle = rotationMatrixToEulerAngles(T_error.topLeftCorner(3, 3)) * 180 / PI;
    cout << "error in euler anles (deg): " << euler_angle.transpose() << endl;
    cout << "error in translation (m): " << T_error.topRightCorner(3, 1).transpose() << endl;
    cout << "error sum in angles (deg): " << euler_angle.norm() << endl;
    cout << "error sum in translation (m): " << T_error.topRightCorner(3, 1).norm() << endl;
    cout << endl << endl;
    // write the data
    outfile << "EDGE: " << loops[i][0] << " " << loops[i][1] << " " << T_result_two[0].topRightCorner(3, 1).transpose()
            << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
  }
  outfile.close();
}
