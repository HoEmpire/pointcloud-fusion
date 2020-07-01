# Colorize the lidar point cloud and mapping

<div align=center><img width="780" height="395" src="images/result.png"/></div>

## Content

- [Introduction](#Introduction)
- [Dependence](#Dependence)
- [Usage](#Usage)

## Introduction

In this project, we use a camera and a livox lidar to realize the function of a RGBD camera.

We project the lidar point cloud to the image pixel by pre-known extrinsic calibration value, so as to colorize the lidar point cloud. Then we use a sequence of colorized pointclouds to recover a dense colorful map of an area.

The advantages of this sensor set are longer detection range and better performance in outdoor scenario.

### Data preprocess Pipeline:

<div align=center><img width="490" height="157" src="images/data_preprocess_pipeline.png"/></div>

### Registration Pipeline:

<div align=center><img width="536" height="281" src="images/registration_pipeline.png"/></div>

## Dependence

- PCL >= 1.7
- ROS-kinetic
- OpenCV >= 4.2
- Eigen >= 3.4
- g2o
- DBoW3
- ros-keyboard(`/pointcloud_fusion/dependence/ros-keyboard`)

## Usage

### 1. Build the files

1. Build a workspace

```shell
mkdir ws/src
```

2. Clone this repository in `/ws/src`

```shell
cd ws/src
git clone https://github.com/HoEmpire/pointcloud-fusion.git
```

3. Copy the ros-keyboard in '/wsr/src'

```shell
copy -r pointcloud/dependence/ros-keyboard .
```

4. Build the files

```shell
catkin_make
```

### 2. Set the configs

We need to set two config files. The first one is in `/pointcloud_fusion/cfg/config.txt` and the other is in `/pointcloud_fusion/cfg/pointcloud_fusion.yaml`

#### config.txt

```txt
-0.00332734 -0.999773 0.0210511 0.00796863
-0.0123546 -0.0210085 -0.999703 0.0592725
0.999918 -0.00358643 -0.0122819 0.0644411
1211.827006 0.000000 710.858487
0.000000 1209.968892 552.339491
0.000000 0.000000 1.000000
-0.103090 0.109216 0.002118 0.000661 0.000000
```

```txt
Rotation Matrix[3x3] Translation Vector[3x1]
Camera Matrix[3x3]
k1 k2 p1 p2 k3
```

#### pointcloud_fusion.yaml

```yaml
pointcloud_fusion:
  lidar_topic: /livox/lidar # subscribed lidar topic
  camera_topic: /camera_array/cam0/image_raw # subscribed camera topic
  save_path: /home/tim/dataset/test2 # save path of the logging mode

icp_nolinear:
  max_correspondence_distance: 0.10
  transformation_epsilon: 0.0001

ndt:
  num_iteration: 50
  transformation_epsilon: 0.01
  step_size: 0.1
  resolution: 0.04

point_cloud_preprocess:
  resample_resolution: 0.02
  statistical_filter_meanK: 30
  statistical_filter_std: 2
  depth_filter_ratio: 5.0 #smaller value for large-scale scenario

loop_closure:
  num_of_result: 4 #number of the return result of the potential loops
  frame_distance_threshold: 2 #the minimum frame distance between two loops
  score_threshold: 0.1 #the matching score threshold for a correct loop
  translation_uncertainty: 0.1 #translation uncertainty between two closest frame
  rotation_uncertainty: 0.5 #rotation uncertainty between two closest frame

io:
  point_cloud_save_path: /home/tim/test.pcd #save path of the point cloud
```

### 3. Run the code

```shell
roslaunch pointcloud_fusion pointcloud_fusion
rosrun keyboard keyboard
```

Then press the keyboard in keyboard window to enter different command.

1-add a new set of data ( a pair of pointcloud and image)

2-map with all the pointclouds and save the result, then kill ros after finishing this process

3-save a set of colorized pointcloud in `.pcd`, RGB images in `.jpg`, depth map in `.png` in
`save_path`

4-end logging and generate a description of the datas, then kill ros

Finally after mapping, the code will generate a mapping result in `point_cloud_save_path` set in `pointcloud_fusion.yaml`. You can view the result by

```shell
pcl_viewer test.pcd #the save file is test.pcd for example
```
