#pragma once
#include "util.h"
#include "type.h"
using namespace std;

void readData(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &pcs, struct imageType &image_data)
{
    int data_len;
    ifstream infile(config.data_path + "description.txt");
    infile >> data_len;
    if (!data_len)
        cout << "\n NO data to read!" << endl;
    else
    {
        cout << "The length of the data is: " << data_len << endl;
        cout << "Reading data..." << endl;
    }

    infile.close();

    for (int i = 0; i < data_len; i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + to_string(i) + ".pcd", *tmp);
        pcs.push_back(tmp);
        image_data.imgs.push_back(cv::imread(config.data_path + to_string(i) + ".jpg"));
        image_data.depths.push_back(cv::imread(config.data_path + to_string(i) + ".png", CV_16UC1));
    }
}

void readDataWithID(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &pcs, struct imageType &image_data)
{
    vector<int> ids;
    int tmp;
    ifstream infile(config.data_path + "id.txt");
    while (1)
    {
        infile >> tmp;
        if (infile.eof())
            break;
        cout << "read id: " << tmp << endl;
        ids.push_back(tmp);
    }

    infile.close();

    for (int i = 0; i < ids.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::io::loadPCDFile<pcl::PointXYZRGB>(config.data_path + to_string(ids[i]) + ".pcd", *tmp);
        pcs.push_back(tmp);
        image_data.imgs.push_back(cv::imread(config.data_path + to_string(ids[i]) + ".jpg"));
        image_data.depths.push_back(cv::imread(config.data_path + to_string(ids[i]) + ".png", CV_16UC1));
    }
}
