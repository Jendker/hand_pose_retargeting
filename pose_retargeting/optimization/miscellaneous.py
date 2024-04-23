import numpy as np
import re


class Weights:
    def __init__(self, bodies_for_hand_pose_energy_position, mujoco_env, minimum_distance=None, maximum_distance=None):
        self.weight_hand_pose_energy_position = None
        self.weight_hand_pose_energy_angle = None
        self.max_weight_hand_pose_energy = 1
        self.min_weight_hand_pose_energy = 0.2
        self.pose_weights = np.ones(len(bodies_for_hand_pose_energy_position))
        self.sum_of_hand_pose_weights = np.sum(self.pose_weights)
        # weights task energy
        self.weight_task_energy = None

        self.max_weight_task_energy = 0.8
        self.min_weight_task_energy = 0

        self.palm_weight = 3
        self.thumb_weight = 3
        thumb_geom_count, other_fingers_geom_count = self.get_finger_geoms_count(mujoco_env)
        self.sum_of_task_energy_weights = self.palm_weight + thumb_geom_count * self.thumb_weight + \
                                          other_fingers_geom_count

        self.minimum_distance = minimum_distance if minimum_distance is not None else 0.08
        self.maximum_distance = maximum_distance if maximum_distance is not None else 0.2

    def update_weights(self, distance_between_hand_and_object):
        distance = np.clip(distance_between_hand_and_object, self.minimum_distance, self.maximum_distance)
        # alpha = (0, 1) with 0 meaning we are very close to the object
        alpha = (distance - self.minimum_distance) / (self.maximum_distance - self.minimum_distance)
        self.weight_task_energy = self.min_weight_task_energy * alpha + self.max_weight_task_energy * (1 - alpha)
        weight_pose_energy = self.max_weight_hand_pose_energy * alpha + self.min_weight_hand_pose_energy * (1 - alpha)
        self.weight_hand_pose_energy_position = weight_pose_energy * 1
        self.weight_hand_pose_energy_angle = weight_pose_energy * 0

    @staticmethod
    def get_finger_geoms_count(mujoco_env):
        geom1 = mujoco_env.env.model.pair_geom1
        geom2 = mujoco_env.env.model.pair_geom2
        all_geoms = set().union(geom1, geom2)

        thumb_geom_count = 0
        other_fingers_geom_count = 0
        for geom_id in all_geoms:
            geom_name = mujoco_env.env.model.geom_id2name(geom_id)
            if re.match('TH\d_collision', geom_name):
                thumb_geom_count += 1
            if re.match('.F\d_collision', geom_name):
                other_fingers_geom_count += 1
        return thumb_geom_count, other_fingers_geom_count


class Targets:
    def __init__(self):
        self.target_joints_pose = None
        self.hand_target_position = None
        self.hand_target_orientation = None


class ConstantData:
    def __init__(self, mujoco_env):
        self.HPE_indices_for_hand_pose_energy_position = range(2, 21)
        self.bodies_for_hand_pose_energy_position = [mujoco_env.getHPEIndexEquivalentBody(x)
                                                     for x in self.HPE_indices_for_hand_pose_energy_position]
        self.HPE_indices_for_hand_pose_energy_angle = range(0, 21)
        self.bodies_for_hand_pose_energy_angles = [mujoco_env.getHPEIndexEquivalentBody(x)
                                                   for x in self.HPE_indices_for_hand_pose_energy_angle]
        self.palm_max_index, self.palm_min_index = self.getMaxMinPalmGeomIndices(mujoco_env)
        self.thumb_geom_indices = self.getThumbGeomIndices(mujoco_env)

    @staticmethod
    def getMaxMinPalmGeomIndices(mujoco_env):
        palm_geom_indices = []
        for geom_name in mujoco_env.env.model.geom_names:
            if 'palm_collision_' in geom_name:
                palm_geom_indices.append(mujoco_env.env.model.geom_name2id(geom_name))
        return max(palm_geom_indices), min(palm_geom_indices)

    @staticmethod
    def getThumbGeomIndices(mujoco_env):
        thumb_geom_indices = []
        for geom_name in mujoco_env.env.model.geom_names:
            if re.match('TH\d_collision', geom_name):
                thumb_geom_indices.append(mujoco_env.env.model.geom_name2id(geom_name))
        return thumb_geom_indices
