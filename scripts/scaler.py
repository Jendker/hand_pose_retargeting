#!/usr/bin/env python

import numpy as np
import vrep


class Scaler:
    def __init__(self):
        self.fingers = [[[2, 9, 10, 11], 'finger'], [[3, 12, 13, 14], 'finger'], [[4, 15, 16, 17], 'finger'],
                        [[5, 18, 19, 20], 'finger'], [[1, 6, 7, 8], 'thumb']]
        self.vrep_points_knuckles = [np.array([0.033, -.0099, 0.352]), np.array([0.011, -0.0099, .356]),
                                     np.array([-.011, -.0099, .352]),
                                     np.array([-0.033, -.0099, .3436]), np.array([0.033, -0.0189, 0.286])]
        self.parts_lengths = {'finger': [0.045, 0.025, 0.026], 'thumb': [0.038, 0.032, 0.0275]}

    def scalePoints(self, original_points_to_scale):
        new_points = original_points_to_scale[:]  # copy
        for finger_index, finger in enumerate(self.fingers):
            finger_indices = finger[0]
            finger_type = finger[1]
            for this_index, point_index in enumerate(finger_indices):
                if this_index == 0:
                    new_points[point_index] = self.vrep_points_knuckles[finger_index]
                else:
                    previous_index = finger_indices[this_index - 1]
                    vector = original_points_to_scale[point_index] - original_points_to_scale[previous_index]
                    new_points[point_index] = new_points[previous_index] + vector / np.linalg.norm(vector) * \
                                              self.parts_lengths[finger_type][this_index - 1]

        return new_points
