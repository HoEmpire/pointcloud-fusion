#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace Eigen;

MatrixXf leastSquareMethod(MatrixXf A, VectorXf b)
{
  return (A.transpose() * A).inverse() * A.transpose() * b;
}

int main(int argc, char** argv)
{
  MatrixXf A(6, 1);
  A << 1, 2, 3, 4, 5, 6;
  VectorXf b(6, 1);
  b << 1, 2, 3, 1, 2, 3;
  VectorXf out = leastSquareMethod(A, b);
  cout << A.transpose() * A << endl;
  cout << out << endl;
  // cout << A.inverse() << endl;
}