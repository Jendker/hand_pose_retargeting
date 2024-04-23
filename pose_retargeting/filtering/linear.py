import numpy as np
from pose_retargeting.filtering.common import packHandPoints, flattenHandPoints


def slerp(q0, q1, h):
    theta = np.arccos(np.dot(q0/np.linalg.norm(q0), q1/np.linalg.norm(q1)))
    sin_theta = np.sin(theta)
    return np.sin((1-h) * theta) / sin_theta * q0 + np.sin(h * theta)/sin_theta * q1


def lerp(q0, q1, h):
    return q0 * h + q1 * (1 - h)


class LinearHandFiltering:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last_flattened_hand_data = None

    def __lerp_data(self, q0, q1):
        ret = []
        new_q0 = np.copy(q0)
        new_q1 = np.copy(q1)
        for q0, q1 in zip(new_q0, new_q1):
            ret.append(q0 * self.alpha + q1 * (1 - self.alpha))
        return np.array(ret)

    def __lerp_hand_data(self, hand_data, old_flattened_hand_data):
        flattened_data = flattenHandPoints(hand_data)
        filtered_flat_data = self.__lerp_data(flattened_data, old_flattened_hand_data)
        return packHandPoints(hand_data, filtered_flat_data), filtered_flat_data

    def filter(self, data):
        if self.last_flattened_hand_data is None:
            self.last_flattened_hand_data = flattenHandPoints(data)
        else:
            data, self.last_flattened_hand_data = self.__lerp_hand_data(data, self.last_flattened_hand_data)
        return data
