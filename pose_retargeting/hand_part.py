#!/usr/bin/env python

from pose_retargeting.vrep_types import VRepMode
import numpy as np
try:
    import rospy
except ImportError:
    pass
from pose_retargeting.simulator.simulator import SimulatorType
import math
import time


def degToRad(angle):
    return angle / 180.0 * math.pi


class HandPart:
    def __init__(self, list_joint_handles_names, tip_handle_name, task_descriptor_base_handles_and_indices,
                 joints_limits, configuration_type, name, simulator):
        self.initialized = False
        self.task_prioritization = True
        self.simulator = simulator
        self.hand_base_handle = self.simulator.getHandle('ShadowRobot_base_tip')
        self.name = name
        self.tip_handle = self.simulator.getHandle(tip_handle_name)
        self.list_joints_handles = []
        for joint_handle_name in list_joint_handles_names:
            handle = self.simulator.getHandle(joint_handle_name)
            self.list_joints_handles.append(handle)

        handle_name_dict = dict(zip(list_joint_handles_names, self.list_joints_handles))
        handle_name_dict[tip_handle_name] = self.tip_handle
        self.task_descriptor_handles = [handle_name_dict[joint_handle_name] for joint_handle_name in
                                        task_descriptor_base_handles_and_indices[0]]
        self.base_handles = [handle_name_dict[joint_handle_name] for joint_handle_name in
                             task_descriptor_base_handles_and_indices[1]]
        self.task_descriptor_equivalent_hpe_indices = task_descriptor_base_handles_and_indices[2]
        self.DOF_count = len(self.list_joints_handles)
        self.tasks_count = len(self.task_descriptor_handles)

        if self.task_prioritization:
            self.K_matrix = np.identity(3) * 15  # for prioritization we use just single error
        else:
            self.K_matrix = np.identity(3 * len(self.task_descriptor_handles)) * 15
        self.weight_matrix_inv = np.identity(self.DOF_count)

        self.human_hand_vel = np.zeros(3 * self.tasks_count)
        self.last_human_hand_part_pose = self.simulator.simulationObjectsPose(
            self.task_descriptor_handles, mode=VRepMode.BLOCKING)
        all_handles_for_jacobian_calc = self.list_joints_handles[:]
        all_handles_for_jacobian_calc.append(self.tip_handle)
        self.jacobian_calculation = self.simulator.jacobianCalculation(
            all_handles_for_jacobian_calc, zip(self.task_descriptor_handles, self.base_handles),
            self.simulator, configuration_type=configuration_type)

        for joint_handle in self.list_joints_handles:  # initialize streaming
            simulator.getJointPosition(joint_handle, mode=VRepMode.STREAMING)
        for handle in self.task_descriptor_handles:
            simulator.getObjectPosition(handle, self.hand_base_handle, mode=VRepMode.STREAMING)

        self.joint_velocity = np.zeros(self.DOF_count)
        self.joints_limits = []
        if self.simulator.type == SimulatorType.MUJOCO:
            for joint_handle in self.list_joints_handles:
                self.joints_limits.append(self.simulator.getJointLimits(joint_handle))
        elif self.simulator.type == SimulatorType.VREP:
            for joint_limits in joints_limits:
                max_angle, min_angle = joint_limits
                self.joints_limits.append([degToRad(max_angle), degToRad(min_angle)])
        else:
            raise ValueError
        self.dummy_targets_handles = self.__createTargetDummies()
        self.first_inverse_calculation = True
        self.errors_in_connection = 0
        self.last_callback_time = 0  # 0 means no callback yet
        self.initialized = True
        self.visualisation_last_poses = self.last_human_hand_part_pose.reshape(2, 3)

    def __del__(self):
        zero_velocities = np.zeros(np.shape(self.list_joints_handles))
        # TODO: Here take care of mujoco as well (get velocities from setJointTargetVelocities and set them before exit
        self.__setJointsTargetVelocity(zero_velocities)
        if self.initialized:
            for dummy_handle in self.dummy_targets_handles:
                self.simulator.removeObject(dummy_handle)

    def __createTargetDummies(self):
        dummy_targets = []
        for i in range(0, self.tasks_count):
            dummy_target = self.simulator.createDummy(0.02,
                                                      [255 * (i % 3), 255 * ((i + 1) % 3), 255 * ((i + 2) % 3), 255])
            dummy_targets.append(dummy_target)
        return dummy_targets

    def __getDampingMatrix(self, jacobian):
        if not self.task_prioritization:
            return np.identity(3 * self.tasks_count) * 0.00001

        lambda_max = 0.01
        epsilon = 0.001
        _, s, _ = np.linalg.svd(jacobian)
        smallest_sigma = s[jacobian.shape[0] - 1]
        if smallest_sigma >= epsilon:
            output_lambda = 0
        else:
            output_lambda = (1 - (smallest_sigma / epsilon) ** 2.) * lambda_max
        return np.identity(3) * output_lambda

    def __updateWeightMatrixInverse(self):
        weight_matrix = np.identity(self.DOF_count)
        for index, joint_handle in enumerate(self.list_joints_handles):
            result, joint_position = self.simulator.getJointPosition(joint_handle)
            if not result:  # failed
                continue
            joint_velocity = self.joint_velocity[index]
            joint_max, joint_min = self.joints_limits[index]
            joint_middle = (joint_max + joint_min) / 2.0
            going_away = bool((joint_position > joint_middle and joint_velocity < 0) or
                              (joint_position < joint_middle and joint_velocity > 0))
            if going_away:
                w = 1.0
            else:
                performance_gradient = (((joint_max - joint_min) ** 2) * (2.0 * joint_position - joint_max - joint_min)
                                        ) / float(
                    4.0 * ((joint_max - joint_position) ** 2) * ((joint_position - joint_min) ** 2)
                    + 0.0000001)
                w = 1.0 + abs(performance_gradient)
            weight_matrix[index, index] = w
        self.weight_matrix_inv = np.linalg.inv(weight_matrix)

    def __updateTargetDummiesPoses(self):
        if self.simulator.type != SimulatorType.VREP:
            all_poses = []
        for index, dummy_handle in enumerate(self.dummy_targets_handles):
            start_index = index * 3
            end_index = start_index + 3
            dummy_position = self.last_human_hand_part_pose[start_index:end_index]
            if self.simulator.type == SimulatorType.VREP:
                self.simulator.setObjectPosition(dummy_handle, self.hand_base_handle, dummy_position.tolist())
            else:
                all_poses.append(dummy_position)
        if self.simulator.type != SimulatorType.VREP:
            self.__updateVisualisationPoses(all_poses)

    def __setJointsTargetVelocity(self, joints_velocities):
        if self.simulator.type == SimulatorType.MUJOCO:
            return joints_velocities
        elif self.simulator.type == SimulatorType.VREP:
            for index, velocity in enumerate(joints_velocities):
                self.simulator.setJointTargetVelocity(self.list_joints_handles[index], velocity,
                                                      self.first_inverse_calculation)
            return None
        else:
            raise ValueError

    def __getError(self, index=None):
        if index is None:
            current_pose = self.simulator.simulationObjectsPose(self.task_descriptor_handles)
            return self.last_human_hand_part_pose - current_pose
        else:
            current_pose = self.simulator.simulationObjectsPose([self.task_descriptor_handles[index]])
            return self.last_human_hand_part_pose[index * 3:index * 3 + 3] - current_pose

    def __updateVisualisationPoses(self, new_poses):
        self.visualisation_last_poses = new_poses

    def getAllTaskDescriptorsErrors(self):
        error = 0.
        for index, _ in enumerate(self.task_descriptor_handles):
            error = error + np.linalg.norm(self.__getError(index))
        return error

    def taskPrioritization(self):
        self.__updateWeightMatrixInverse()
        pseudo_inverse_jacobians, jacobians = self.__getPseudoInverseForTaskPrioritization()
        q_vel = np.zeros(self.DOF_count)
        multiplier = np.identity(self.DOF_count)
        rotation_matrix = self.simulator.getTransformationMatrixToBase()[0:3, 0:3]
        for index, task_handle in enumerate(self.task_descriptor_handles):
            error = self.__getError(index)
            q_vel = q_vel + np.dot(np.dot(multiplier, pseudo_inverse_jacobians[index]),
                                   rotation_matrix @ (self.human_hand_vel[index * 3:index * 3 + 3]
                                                      + np.dot(self.K_matrix, error)))
            multiplier = np.dot(multiplier,
                                np.identity(self.DOF_count) - np.dot(pseudo_inverse_jacobians[index], jacobians[index]))
        self.joint_velocity = q_vel
        return self.__setJointsTargetVelocity(self.joint_velocity)

    def taskAugmentation(self):
        # TODO: the same thing as in taskPrioritization to do here
        error = self.__getError()
        self.__updateWeightMatrixInverse()
        pseudo_inverse_jacobian = self.__getPseudoInverseForTaskAugmentation()
        rotation_matrix = self.simulator.getTransformationMatrixToBase()[0:3, 0:3]
        stacked_rotation_matrix = np.block([[rotation_matrix, np.zeros((3, 3))], [np.zeros((3, 3)), rotation_matrix]])
        q_vel = np.dot(pseudo_inverse_jacobian, stacked_rotation_matrix @ (self.human_hand_vel + np.dot(self.K_matrix, error)))
        self.joint_velocity = q_vel
        return self.__setJointsTargetVelocity(self.joint_velocity)

    def __getPseudoInverseForTaskPrioritization(self):
        whole_jacobian = self.jacobian_calculation.getJacobian()
        jacobians = []
        pseudo_jacobian_inverses = []
        for task_index, _ in enumerate(self.task_descriptor_handles):
            this_jacobian = whole_jacobian[..., task_index * 3:task_index * 3 + 3].T
            jacobians.append(this_jacobian)
            damping_matrix = self.__getDampingMatrix(this_jacobian)
            this_pseudo_jacobian_inverse = np.linalg.multi_dot([self.weight_matrix_inv, this_jacobian.T, np.linalg.inv(
                np.linalg.multi_dot([this_jacobian, self.weight_matrix_inv, this_jacobian.T]) + damping_matrix)])
            pseudo_jacobian_inverses.append(this_pseudo_jacobian_inverse)
        return pseudo_jacobian_inverses, jacobians

    def __getPseudoInverseForTaskAugmentation(self):
        if len(self.task_descriptor_handles) != 2:
            rospy.logerr("Task augmentation works currently only with 2 target handles. Current count: %d. Exiting.",
                         len(self.task_descriptor_handles))
            exit(1)
        jacobian = self.jacobian_calculation.getJacobian()
        jacobian = np.concatenate((jacobian[..., 0:3].T, jacobian[..., 3:6].T), axis=0)
        damping_matrix = self.__getDampingMatrix(jacobian)
        return np.linalg.multi_dot([self.weight_matrix_inv, jacobian.T, np.linalg.inv(
            np.linalg.multi_dot([jacobian, self.weight_matrix_inv, jacobian.T]) + damping_matrix)])

    def executeControl(self):
        if self.task_prioritization:
            joints_velocities = self.taskPrioritization()
        else:
            joints_velocities = self.taskAugmentation()
        self.first_inverse_calculation = False
        if self.simulator.type == SimulatorType.VREP:
            return None
        elif self.simulator.type == SimulatorType.MUJOCO:
            self.simulator.visualisePose(self.visualisation_last_poses)
            joint_velocity_dict = {}
            for index, velocity in enumerate(joints_velocities):
                joint_velocity_dict[self.simulator.getJointIndex(self.list_joints_handles[index])] = velocity
            return joint_velocity_dict
        else:
            raise ValueError

    def newPositionFromHPE(self, new_data):
        current_time = time.time()
        hand_part_poses = []
        for index in self.task_descriptor_equivalent_hpe_indices:
            hand_part_poses.append(new_data[index, :])
        new_HPE_hand_pose = np.concatenate(hand_part_poses)
        if self.last_callback_time != 0:  # TODO: maybe here we can make better with calculating with first iteration
            self.human_hand_vel = (new_HPE_hand_pose - self.last_human_hand_part_pose) / (
                    current_time - self.last_callback_time)
            self.last_callback_time = current_time
        else:
            self.last_callback_time = current_time
        self.last_human_hand_part_pose = new_HPE_hand_pose
        self.__updateTargetDummiesPoses()

    def getName(self):
        return self.name

    def taskDescriptorsCount(self):
        return len(self.task_descriptor_handles)
