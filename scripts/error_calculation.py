#!/usr/bin/env python

import time
import scipy.io
import os
import rospy
import vrep
import numpy as np


class ErrorCalculation:
    def __init__(self, clientID, hand_parts, joint_handle_names_for_errors, indices_of_joints_from_hpe,
                 frequency, alpha, joint_handles_dict):
        self.clientID = clientID
        self.alpha = alpha
        self.indices_of_joints_from_hpe = indices_of_joints_from_hpe
        self.running = False
        self.hand_parts = hand_parts
        self.joint_handles_for_errors = []
        for joint_handle_group in joint_handle_names_for_errors:
            these_handles = []
            for joint_handle_name in joint_handle_group:
                these_handles.append(joint_handles_dict.getHandle(joint_handle_name))
            self.joint_handles_for_errors.append(these_handles)
        self.frequency = frequency
        self.errors = []
        self.start_time = 0
        self.execution_times = 0
        self.last_hpe_update = 0
        self.all_joint_handles = [item for sublist in self.joint_handles_for_errors for item in sublist]
        self.per_finger_errors = {}
        for hand_part in self.hand_parts:
            self.per_finger_errors[hand_part.getName()] = []
        streaming = []
        hand_base_handle = joint_handles_dict.getHandle('ShadowRobot_base_tip')
        for handle in self.all_joint_handles:  # initialize streaming
            result, _ = vrep.simxGetObjectPosition(self.clientID, handle, hand_base_handle, vrep.simx_opmode_streaming)
            streaming.append(result)
        while not all(result == vrep.simx_return_ok for result in streaming):
            streaming = []
            for handle in self.all_joint_handles:  # initialize streaming
                result, _ = vrep.simxGetObjectPosition(self.clientID, handle, hand_base_handle, vrep.simx_opmode_buffer)
                streaming.append(result)
            time.sleep(0.01)
        self.last_human_hand_pose = []
        for index, hand_part in enumerate(self.hand_parts):
            self.last_human_hand_pose.append(hand_part.simulationObjectsPose(self.joint_handles_for_errors[index]))

    def __del__(self):
        self.saveResults()

    def handPartError(self, index):
        error = 0.
        for handle_index, handle in enumerate(self.joint_handles_for_errors[index]):
            current_pose = self.hand_parts[index].simulationObjectsPose([handle])
            error += np.linalg.norm(self.last_human_hand_pose[index][handle_index * 3:handle_index * 3 + 3] - current_pose)
        return error

    def calculateError(self):
        if self.running and self.start_time + 1. / self.frequency * self.execution_times < time.time():
            self.execution_times += 1
            whole_error = 0.
            for index, hand_part in enumerate(self.hand_parts):
                this_error = self.handPartError(index)
                whole_error += this_error
                self.per_finger_errors[hand_part.getName()].append(this_error)
            self.errors.append(whole_error)
            if self.last_hpe_update + 2. < time.time():
                self.stop()

    def newPositionFromHPE(self, new_data):
        for hand_part_index, indices_group in enumerate(self.indices_of_joints_from_hpe):
            hand_part_poses = []
            for index in indices_group:
                hand_part_poses.append(new_data.joints_position[index])
            HPE_hand_part_poses = np.concatenate(hand_part_poses)
            self.last_human_hand_pose[hand_part_index] = HPE_hand_part_poses * self.alpha + self.last_human_hand_pose[hand_part_index] * (1 - self.alpha)
        self.last_hpe_update = time.time()
        if not self.running:
            self.running = True
            self.start_time = time.time()
            rospy.loginfo("Starting calculation of errors for finger joints.")

    def stop(self):
        self.running = False
        self.saveResults()
        rospy.loginfo("Finished calculation of errors for finger joints.")

    def saveResults(self):
        folder_path = '/media/psf/Dropbox/Forschungspraxis/error_results'  # laptop
        if not os.path.isdir('/media/psf/Dropbox/Forschungspraxis'):
            folder_path = '/home/jedrzej/Dropbox/Forschungspraxis/error_results'  # university computer
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        file_name = ''
        for hand_part in self.hand_parts:
            file_name += hand_part.getName() + str(hand_part.taskDescriptorsCount())
        dict_to_save = self.per_finger_errors
        dict_to_save['whole_error'] = self.errors
        whole_file_name = folder_path + '/' + file_name + '.mat'
        if os.path.exists(whole_file_name):
            os.remove(whole_file_name)
        scipy.io.savemat(whole_file_name, mdict=dict_to_save)

