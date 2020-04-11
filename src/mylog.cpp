#include "pointcloud_fusion/mylog.h"
mylog::mylog()
{
}

mylog::~mylog()
{
    outfile_pointcloud.open(save_path + "/points.txt", std::ios_base::trunc);
    outfile_img_time.open(save_path + "/image_time.txt", std::ios_base::trunc);
    int num_point = 0;
    for (int i = 0; i < pc_array.size(); i++)
    {
        double time_pointcloud = pc_array[i]->header.stamp.toSec();
        pcl::PointCloud<pcl::PointXYZI> point_cloud;
        pcl::fromROSMsg(*pc_array[i], point_cloud);
        for (pcl::PointCloud<pcl::PointXYZI>::iterator pt =
                 point_cloud.points.begin();
             pt < point_cloud.points.end(); pt++)
        {
            if (pt->x != 0)
            {
                outfile_pointcloud << std::fixed << std::setprecision(9) << time_pointcloud << " ";
                outfile_pointcloud << pt->x << " ";
                outfile_pointcloud << pt->y << " ";
                outfile_pointcloud << pt->z << " ";
                outfile_pointcloud << pt->intensity << std::endl;
                num_point++;
            }
        }
    }
    std::cout << "saving " << num_point << " points done" << std::endl;
    for (int i = 0; i < img_array.size(); i++)
    {
        double time_img = img_array[i]->header.stamp.toSec();
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(img_array[i], sensor_msgs::image_encodings::BGR8);
        cv::imwrite(save_path + "/" + std::to_string(i) + ".jpg", cv_ptr->image);
        std::cout << save_path + "/" + std::to_string(i) + ".jpg" << std::endl;
        outfile_img_time << i << " ";
        outfile_img_time << std::fixed << std::setprecision(9) << time_img << std::endl;
    }
    std::cout << "saving " << img_array.size() << " images done" << std::endl;
    outfile_pointcloud.close();
    outfile_img_time.close();
}
