#!/usr/bin/env python

import numpy as np
from pose_retargeting.error_calculation import ErrorCalculation
from pose_retargeting.jacobians.configuration_type import ConfigurationType
from pose_retargeting.hand_part import HandPart


class Hand:
    def __init__(self, simulator, calculate_error=False):
        self.simulator = simulator
        self.calculate_error = calculate_error

        index_finger = HandPart(['IMCP_side_joint', 'IMCP_front_joint', 'IPIP_joint', 'IDIP_joint'],
                                'ITIP_tip',
                                [['ITIP_tip', 'IPIP_joint'], ['IMCP_side_joint', 'IMCP_side_joint'], [11, 9]],
                                [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'index',
                                self.simulator)
        middle_finger = HandPart(['MMCP_side_joint', 'MMCP_front_joint', 'MPIP_joint', 'MDIP_joint'],
                                 'MTIP_tip',
                                 [['MTIP_tip', 'MPIP_joint'], ['MMCP_side_joint', 'MMCP_side_joint'], [14, 12]],
                                 [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'middle',
                                 self.simulator)
        ring_finger = HandPart(['RMCP_side_joint', 'RMCP_front_joint', 'RPIP_joint', 'RDIP_joint'],
                               'RTIP_tip',
                               [['RTIP_tip', 'RPIP_joint'], ['RMCP_side_joint', 'RMCP_side_joint'], [17, 15]],
                               [[10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.finger, 'ring',
                               self.simulator)
        pinkie_finger = HandPart(['metacarpal_joint', 'PMCP_side_joint', 'PMCP_front_joint', 'PPIP_joint',
                                  'PDIP_joint'], 'PTIP_tip',
                                 [['PTIP_tip', 'PPIP_joint'], ['metacarpal_joint', 'metacarpal_joint'], [20, 18]],
                                 [[45., 0.], [10., -10.], [100., 0.], [90., 0.], [90., 0.]], ConfigurationType.pinkie,
                                 'pinkie', self.simulator)
        thumb_finger = HandPart(['TMCP_rotation_joint', 'TMCP_front_joint', 'TPIP_side_joint', 'TPIP_front_joint',
                                 'TDIP_joint'], 'TTIP_tip',
                                [['TTIP_tip', 'TPIP_front_joint'], ['TMCP_rotation_joint', 'TMCP_rotation_joint'],
                                 [8, 6]],
                                [[60., -60.], [70., 0.], [30., -30.], [12., -12.], [90, 0]], ConfigurationType.thumb,
                                'thumb', self.simulator)

        self.hand_parts_list = (index_finger, middle_finger, ring_finger, pinkie_finger, thumb_finger)
        if calculate_error:
            self.error_calculation = ErrorCalculation(list(self.hand_parts_list),
                                                      [['IPIP_joint', 'IDIP_joint', 'ITIP_tip'],
                                                       ['MPIP_joint', 'MDIP_joint', 'MTIP_tip'],
                                                       ['RPIP_joint', 'RDIP_joint', 'RTIP_tip'],
                                                       ['PPIP_joint', 'PDIP_joint', 'PTIP_tip'],
                                                       ['TPIP_front_joint', 'TDIP_joint', 'TTIP_tip']],
                                                      [[9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20],
                                                       [6, 7, 8]], 10, self.simulator)

    def controlOnce(self):
        for hand_part in self.hand_parts_list:
            hand_part.executeControl()
        if self.calculate_error:
            self.error_calculation.calculateError()

    def getControlOnce(self):
        action_dict = {}
        for hand_part in self.hand_parts_list:
            action_dict.update(hand_part.executeControl())  # given as velocities
        # clamp velocities between -3.5 and 3.5
        for key, value in action_dict.items():
            action_dict[key] = min(max(-3.5, value), 3.5)
        # for key, value in action_dict.items():
        #     action_dict[key] = value * self.simulator.env.dt  # integrate the velocity

        complete_action_vector = self.simulator.getHandBaseAction()
        complete_action_vector = np.pad(complete_action_vector, (0, self.simulator.getNumberOfJoints() -
                                                                 complete_action_vector.size), 'constant',
                                        constant_values=0)
        for k, v in action_dict.items():
            complete_action_vector[k] = v  #+ self.simulator.getJointIndexPosition(k)  # add position step to current
        return complete_action_vector

    def newPositionFromHPE(self, new_data):
        for hand_part in self.hand_parts_list:
            hand_part.newPositionFromHPE(new_data)
        if self.calculate_error:
            self.error_calculation.newPositionFromHPE(new_data)
