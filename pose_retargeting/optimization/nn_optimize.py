import os
import re
import pickle
import numpy as np


def extract_int(x):
    ints = re.findall(r'\d+', x)
    if ints:
        return ints[0]
    else:
        return ''


def biggest_int(x):
    return max([int(extract_int(s)) for s in x if extract_int(s)])


class NN_optimize:
    def __init__(self, path=None, policy=None):
        job_name = 'relocate_demo_init_dapg_sphere'
        self.weight_nn = 0.25

        if policy is None:
            if path is None:
                script_path = os.path.realpath(__file__)
                path = os.path.dirname(script_path) + "/../../../mt_src/mt_src/training/Runs"
            path = path + "/" + job_name + "/iterations"
            if not os.path.exists(path):
                print("NN optimize. Path with policy:\n", path, "\nDoes not exist! Exiting.")
                exit(1)
            try:
                file_list = os.listdir(path)
            except NotADirectoryError:
                print("No policy files found for NN optimize in path:\n" + path + "\nExiting.")
                exit(1)
            file_list = sorted(file_list, key=extract_int)
            if file_list:
                max_iteration_number = biggest_int(file_list)
                if policy is None:
                    policy = pickle.load(
                        open(path + '/checkpoint_' + str(max_iteration_number) + '.pickle', 'rb'))[0]
                print("NN optimize starting from iteration no. " + str(max_iteration_number))
            else:
                print("No policy files found for NN optimize in path:\n" + path + "\nExiting.")
                exit(1)
        self.policy = policy
        self.obj_body_index = None
        self.grasp_site_index = None

    def getDistanceBetweenObjectAndHand(self, mujoco_env):
        if self.obj_body_index is None:
            self.obj_body_index = mujoco_env.env.model.body_name2id('Object')
            self.grasp_site_index = mujoco_env.env.model.site_name2id('S_grasp')

        obj_pos = mujoco_env.env.data.body_xpos[self.obj_body_index].ravel()
        palm_pos = mujoco_env.env.data.site_xpos[self.grasp_site_index].ravel()
        return np.linalg.norm(obj_pos - palm_pos)

    def optimize(self, observation, action, sim_env):
        distance_between_object_and_hand = self.getDistanceBetweenObjectAndHand(sim_env)
        if distance_between_object_and_hand > 0.075:
            return action
        nn_actions = self.policy.get_action(observation)[1]['evaluation']
        new_action = self.weight_nn * nn_actions + (1 - self.weight_nn) * action
        new_action[:6] = action[:6]  # don't optimize the position and orientation with NN
        return new_action
