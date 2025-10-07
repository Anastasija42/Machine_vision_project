# Machine_vision_project
Master's course project for detecting wooden planks

## Making a ROS package

```bash
catkin_create_pkg vision_processor rospy std_msgs geometry_msgs

cd ~/catkin_ws
catkin_make

source devel/setup.bash

cd ~/catkin_ws/src/vision_processor
mkdir scripts
cd scripts

gedit point_publisher.py

chmod +x point_subscriber.py

```




