The code found here is based on the code developed for running the Hesai QT128 LiDAR using ROS, the original repo can be found here:
https://github.com/HesaiTechnology/HesaiLidar_Swift_ROS

**Requirements**
1. Ubuntu 20.04
2. ROS Noetic
3. Hesai QT128 LiDAR 

**Explination of the code**

The code shall be installed (pulled) into a manually to be created folder called "rosworkspace". 
To find to the code which is added for denoising the point cloud go to:
"rosworkspace/src/HesaiLidar_Swift_ROS/pandar_pointcloud/src/conversions/live_denoiser.py"

To find to the code which is added for generating the ground surface point cloud there are two options (scripts).
One in C++ code (more advanced, works better) one simpler Python version, I recommend running the C++ script.
1. ground_reconstruction_node.cpp (C++) 
2. fast_seg.py (Python)

When adjusting the code and after pulling the code make sure to enter the "catkin_make" command once you are inside to rosworkspace folder using the terminal. 

**OFFLINE USE: Running a Rosbag file**

To do so, go to the "rosworkspace/src/HesaiLidar_Swift_ROS/pandar_pointcloud/launch" folder and run the brutus_node.launch
Make sure the SanDisk USB stick called "510F-1275" inserted into the Jetson Orin Nano.

**ONLINE USE: Using the actual Hesai QT128 LiDAR**

----------------- ROS LiDAR setup ----------------------------

First run the command below to find the LiDAR's name:
ifconfig

Secondly run the following command to configure the correct adress (you may need to run this multiple times until it works!):
sudo ifconfig ethCORRECTNAMEHERE 192.168.1.100

To validate the LiDAR connection open firefox and run:
192.168.1.201

----------------- ROS view live data ------------------------

Run the following commnd just once:
﻿source /opt/ros/noetic/setup.bash

Navigate to the ROS workspace and run the following command in **all the terminals you open**: 
source devel/setup.bash

In termninal one run: 
roscore

In termnal two run: 
rviz

In terminal three run: 
roslaunch pandar_pointcloud PandarSwift_points.launch

Next do the following in RIVZ (other wise you won't see anything):
Set the fixed frame to: PandarSwift
Set the topic of the PointCloud you want to visual to: /hesai/pandar_points

----------------- ROS view processed data --------------------

To enable and view the point cloud denoiser:
Open a new terminal and enter: rosrun pandar_pointcloud live_denoiser.py
In RVIZ change the topic of the PointCloud you want to visualise to /denoised_cloud

To enable and view the reconstructed ground surface point cloud:
Open a new terminal and enter: rosrun pandar_pointcloud ground_reconstruction_node
In RVIZ change the topic of the PointCloud you want to visualise to /filtered_cloud







