#!/usr/bin/env python

import rospy
from pose_retargeting.mapper import Mapper
from dl_pose_estimation.msg import JointsPosition


def run():
    node_name = 'pose_mapping_vrep'
    rospy.init_node(node_name)
    mapper = Mapper(node_name)
    rospy.Subscriber("/dl_pose_estimation/joints_position", JointsPosition, mapper.callback)

    mapper.execute()


if __name__ == '__main__':
    run()
