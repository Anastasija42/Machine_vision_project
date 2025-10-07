#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
import random

def point_publisher():
    rospy.init_node('vision_publisher_node', anonymous=True)
    pub = rospy.Publisher('/vision/detected_point', Point, queue_size=10)
    rate = rospy.Rate(10)
    rospy.loginfo("Vision point publisher node started...")

    while not rospy.is_shutdown():

        detected_x = 150.0 + random.uniform(-10, 10)
        detected_y = 200.0 + random.uniform(-10, 10)
        detected_z = 0.0

        point_msg = Point()
        point_msg.x = detected_x
        point_msg.y = detected_y
        point_msg.z = detected_z

        rospy.loginfo(f"Publishing point: x={point_msg.x:.2f}, y={point_msg.y:.2f}, z={point_msg.z:.2f}")

        pub.publish(point_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        point_publisher()
    except rospy.ROSInterruptException:
        pass