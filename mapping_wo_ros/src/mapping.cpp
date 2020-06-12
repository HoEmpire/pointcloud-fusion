#include "icp.h"
#include "util.h"
#include "visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
int main(int argv, char **argc)
{
  vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs;
  struct imageType image_data;
  readConfig();
  readData(pcs, image_data);
  image_data.init();

  vector<Matrix4f> T_init = calVisualOdometry(image_data);
  // icpNonlinearWithNormal(pcs, T_init);
  vector<Matrix4f> T_result;
  ndtRegistration(pcs, T_init, T_result);

  vector<vector<int>> loops;
  loop_closure(image_data, loops);
  ofstream outfile("./result.txt");

  Matrix4f tmp_pos;
  tmp_pos.setIdentity(4, 4);
  outfile << "VERTEX: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0" << endl;
  for (int i = 0; i < T_result.size(); i++)
  {
    tmp_pos = T_result[i] * tmp_pos;
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
    vector<Mat> imgs, depths;
    vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcs_two;

    pcs_two.push_back(pcs[loops[i][0]]);
    pcs_two.push_back(pcs[loops[i][1]]);

    imgs.push_back(image_data.imgs[loops[i][0]]);
    imgs.push_back(image_data.imgs[loops[i][1]]);
    depths.push_back(image_data.depths[loops[i][0]]);
    depths.push_back(image_data.depths[loops[i][1]]);

    struct imageType image_data_two;
    image_data_two.imgs = imgs;
    image_data_two.depths = depths;
    image_data_two.init();
    vector<Matrix4f> T_init_two = calVisualOdometry(image_data_two);
    vector<Matrix4f> T_result_two;
    ndtRegistration(pcs_two, T_init_two, T_result_two);

    Matrix3d rotation_matrix = T_result_two[0].topLeftCorner(3, 3).cast<double>();
    Quaterniond q(rotation_matrix);
    outfile << "EDGE: " << loops[i][0] << " " << loops[i][1] << " " << T_result_two[0].topRightCorner(3, 1).transpose()
            << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    ;
  }
  outfile.close();
}
