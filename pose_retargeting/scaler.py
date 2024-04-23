#!/usr/bin/env python

import numpy as np


class Scaler:
    def __init__(self, simulator):
        self.fingers = [[[2, 9, 10, 11], 'finger'], [[3, 12, 13, 14], 'finger'], [[4, 15, 16, 17], 'finger'],
                        [[5, 18, 19, 20], 'finger'], [[1, 6, 7, 8], 'thumb']]
        self.simulator_points_knuckles = simulator.scaling_points_knuckles[:]
        self.parts_lengths = {'finger': [0.045, 0.025, 0.026], 'thumb': [0.038, 0.032, 0.0275]}

    def scalePoints(self, original_points_to_scale):
        new_points = original_points_to_scale[:]  # copy
        for finger_index, finger in enumerate(self.fingers):
            finger_indices = finger[0]
            finger_type = finger[1]
            for this_index, point_index in enumerate(finger_indices):
                if this_index == 0:
                    new_points[point_index] = self.simulator_points_knuckles[finger_index]
                else:
                    previous_index = finger_indices[this_index - 1]
                    vector = original_points_to_scale[point_index] - original_points_to_scale[previous_index]
                    new_points[point_index] = new_points[previous_index] + vector / np.linalg.norm(vector) * \
                                              self.parts_lengths[finger_type][this_index - 1]

        return new_points
