#!/usr/bin/env python

import vrep
import rospy


class JointHandlesDict:
    def __init__(self, clientID):
        joint_handle_names = ['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint', 'ITIP_tip',
                              'MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint', 'MTIP_tip',
                              'RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint', 'RTIP_tip',
                              'metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                              'PDIP_joint', 'PTIP_tip', 'TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint', 'TPIP_front_joint',
                              'TDIP_joint', 'TTIP_tip', 'ShadowRobot_base_target', 'ShadowRobot_base_tip']
        self.joint_handles_dict = {}
        for handle_name in joint_handle_names:
            result, handle = vrep.simxGetObjectHandle(clientID, handle_name, vrep.simx_opmode_blocking)
            if result != vrep.simx_return_ok:
                rospy.logerr("Handle %s does not exist! Exiting.", handle_name)
                exit(1)
            self.joint_handles_dict[handle_name] = handle

    def getHandle(self, handle_name):
        return self.joint_handles_dict[handle_name]
