import vrep
from enum import Enum
import sympy as sp
import numpy as np
import time
import rospy
import pickle
import os
import rospkg


class ConfigurationType(Enum):
    finger = 1
    thumb = 2
    pinkie = 3


class JacobianCalculation:
    def __init__(self, clientID, transformation_handles, task_objects_handles_and_bases, configuration_type, joint_handles_dict):
        self.clientID = clientID
        self.joint_handles = transformation_handles[:-1]
        self.task_object_handles_and_bases = task_objects_handles_and_bases
        self.all_handles = transformation_handles[:]  # contains also finger tip
        self.q = sp.symbols('q_0:{}'.format(len(self.joint_handles)))
        self.joint_handle_q_map = dict(zip(self.joint_handles, self.q))
        for joint_handle in self.joint_handles:  # initialize streaming
            result, _ = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_streaming)
        while result != vrep.simx_return_ok:
            result, _ = vrep.simxGetJointPosition(self.clientID, self.joint_handles[0], vrep.simx_opmode_buffer)
            time.sleep(0.01)
        self.jacobian, self.Ts = self.calculateJacobian(configuration_type, joint_handles_dict)

    def updateClientID(self, clientID):
        self.clientID = clientID

    def calculateJacobian(self, configuration_type, joint_handles_dict):
        rospack = rospkg.RosPack()
        jacobians_folder_path = rospack.get_path('pose_mapping_vrep') + "/scripts/jacobians/"
        if not os.path.isdir(jacobians_folder_path):
            os.mkdir(jacobians_folder_path)
        jacobian_configuration = self.task_object_handles_and_bases
        conf_string = np.array_str(np.array(jacobian_configuration)).replace("\n", "")
        jacobians_filename = jacobians_folder_path + conf_string + configuration_type.name + ".dat"

        already_calculated = False
        if os.path.isfile(jacobians_filename):
            with open(jacobians_filename, 'rb') as handle:
                calculated_configuration = pickle.load(handle)
                Ts = calculated_configuration[0]
                jacobian = calculated_configuration[1]
                already_calculated = True

        if not already_calculated:
            rospy.loginfo("Calculating Jacobian for hand part. Please wait")
            hand_base_handle = joint_handles_dict.getHandle('ShadowRobot_base_tip')
            objects_positions = {}
            for joint_handle in self.all_handles:
                _, this_object_position = vrep.simxGetObjectPosition(self.clientID, joint_handle, hand_base_handle,
                                                                     vrep.simx_opmode_blocking)
                objects_positions[joint_handle] = np.array(this_object_position)

            jacobian = sp.zeros(len(self.joint_handles), 3 * len(self.task_object_handles_and_bases))
            Ts = []
            for task_index, [target_handle, base_handle] in enumerate(self.task_object_handles_and_bases):
                transformations = []
                transformation_handles = []
                for handle in self.all_handles:
                    if not transformation_handles:  # if empty
                        if handle == base_handle:
                            transformation_handles.append(handle)
                    else:
                        transformation_handles.append(handle)
                        if handle == target_handle:
                            break

                if not transformation_handles:
                    str = "No base handle found for transformation %d. Skipping." % task_index
                    rospy.logwarn(str)
                    continue

                if configuration_type == ConfigurationType.finger:
                    for index, joint_handle in enumerate(transformation_handles):
                        if joint_handle == target_handle:
                            last_translation = np.linalg.norm(objects_positions[transformation_handles[index]] -
                                                              objects_positions[transformation_handles[index - 1]])
                            transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 1, last_translation],
                                                              [0, 0, 0, 1]]))
                            continue

                        angle = self.joint_handle_q_map[joint_handle]
                        if index == 0:
                            vector_this_object_position = objects_positions[joint_handle]
                            transformations.append(sp.Matrix([[1, 0, 0, vector_this_object_position[0]],
                                                              [0, 1, 0, vector_this_object_position[1]],
                                                              [0, 0, 1, vector_this_object_position[2]],
                                                              [0, 0, 0, 1]]))
                            transformations.append(sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), 0],
                                                              [0, 1, 0, 0],
                                                              [-sp.sin(angle), 0, sp.cos(angle), 0],
                                                              [0, 0, 0, 1]]))
                            continue

                        length = np.linalg.norm(objects_positions[transformation_handles[index]] - objects_positions[
                            transformation_handles[index - 1]])
                        # if length < 0.001:
                        #     length = 0.
                        transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                          [0, sp.cos(angle), -sp.sin(angle), 0],
                                                          [0, sp.sin(angle), sp.cos(angle), length],
                                                          [0, 0, 0, 1]]))

                elif configuration_type == ConfigurationType.thumb:
                    for index, joint_handle in enumerate(transformation_handles):
                        vector_this_object_position = objects_positions[joint_handle]

                        if joint_handle == target_handle:
                            last_translation = np.linalg.norm(objects_positions[transformation_handles[index]] -
                                                              objects_positions[transformation_handles[index - 1]])
                            transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 1, last_translation],
                                                              [0, 0, 0, 1]]))
                            continue

                        angle = self.joint_handle_q_map[joint_handle]  # -angle, because the joints turn clockwise
                        if index == 0:
                            transformations.append(sp.Matrix([[1, 0, 0, vector_this_object_position[0]],
                                                              [0, 1, 0, vector_this_object_position[1]],
                                                              [0, 0, 1, vector_this_object_position[2]],
                                                              [0, 0, 0, 1]]))
                            deg_to_rad = sp.pi / 4.0
                            transformations.append(sp.Matrix([[sp.cos(deg_to_rad), 0, sp.sin(deg_to_rad), 0],
                                                              [0, 1, 0, 0],
                                                              [-sp.sin(deg_to_rad), 0, sp.cos(deg_to_rad), 0],
                                                              [0, 0, 0, 1]]))
                            transformations.append(sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0, 0],
                                                              [sp.sin(angle), sp.cos(angle), 0, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]]))
                            continue

                        length = np.linalg.norm(objects_positions[transformation_handles[index]] - objects_positions[
                            transformation_handles[index - 1]])
                        # if length < 0.001:
                        #     length = 0.
                        if index == 1 or index == 3:
                            transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                              [0, sp.cos(angle), -sp.sin(angle), 0],
                                                              [0, sp.sin(angle), sp.cos(angle), length],
                                                              [0, 0, 0, 1]]))
                            continue
                        if index == 2 or index == 4:
                            angle = -angle  # these two joints are rotated by 180* in VREP
                            transformations.append(
                                sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), 0],
                                           [0, 1, 0, 0],
                                           [-sp.sin(angle), 0, sp.cos(angle), length],
                                           [0, 0, 0, 1]]))
                            continue
                        rospy.logerr("Transformation index not defined")
                        exit(1)

                elif configuration_type == ConfigurationType.pinkie:
                    for index, joint_handle in enumerate(transformation_handles):
                        if joint_handle == target_handle:
                            last_translation = np.linalg.norm(objects_positions[transformation_handles[index]] -
                                                              objects_positions[transformation_handles[index - 1]])
                            transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 1, last_translation],
                                                              [0, 0, 0, 1]]))
                            continue

                        angle = self.joint_handle_q_map[joint_handle]
                        if index == 0:
                            vector_this_object_position = objects_positions[joint_handle]
                            angle_55_deg = 55. / 180. * sp.pi
                            transformations.append(sp.Matrix([[1, 0, 0, vector_this_object_position[0]],
                                                              [0, 1, 0, vector_this_object_position[1]],
                                                              [0, 0, 1, vector_this_object_position[2]],
                                                              [0, 0, 0, 1]]))
                            transformations.append(sp.Matrix([[sp.cos(-angle_55_deg), 0, sp.sin(-angle_55_deg), 0],
                                                              [0, 1, 0, 0],
                                                              [-sp.sin(-angle_55_deg), 0, sp.cos(-angle_55_deg), 0],
                                                              [0, 0, 0, 1]]))
                            transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                              [0, sp.cos(angle), -sp.sin(angle), 0],
                                                              [0, sp.sin(angle), sp.cos(angle), 0],
                                                              [0, 0, 0, 1]]))
                            transformations.append(sp.Matrix([[sp.cos(angle_55_deg), 0, sp.sin(angle_55_deg), 0],
                                                              [0, 1, 0, 0],
                                                              [-sp.sin(angle_55_deg), 0, sp.cos(angle_55_deg), 0],
                                                              [0, 0, 0, 1]]))
                            vector_previous_object_position = vector_this_object_position.copy()
                            continue
                        length = np.linalg.norm(objects_positions[transformation_handles[index]] - objects_positions[
                            transformation_handles[index - 1]])
                        if index == 1:
                            vector_this_object_position = objects_positions[joint_handle]
                            vector_shift = vector_this_object_position - vector_previous_object_position
                            transformations.append(sp.Matrix([[sp.cos(angle), 0, sp.sin(angle), vector_shift[0]],
                                                              [0, 1, 0, vector_shift[1]],
                                                              [-sp.sin(angle), 0, sp.cos(angle), vector_shift[2]],
                                                              [0, 0, 0, 1]]))
                            continue
                        # if length < 0.001:
                        #     length = 0.
                        transformations.append(sp.Matrix([[1, 0, 0, 0],
                                                          [0, sp.cos(angle), -sp.sin(angle), 0],
                                                          [0, sp.sin(angle), sp.cos(angle), length],
                                                          [0, 0, 0, 1]]))

                else:
                    print("Configuration not defined")
                    exit(1)

                T = sp.eye(4)
                for index, transformation in enumerate(transformations):
                    T = T * transformation
                T = sp.simplify(T)
                Ts.append(T)
                this_jacobian = T[0:3, 3].T.jacobian(sp.Matrix(self.q)).T
                jacobian[:, task_index * 3: task_index * 3 + 3] = this_jacobian

            jacobian = sp.simplify(jacobian)
            calculated_configuration = (Ts, jacobian)
            with open(jacobians_filename, 'wb') as handle:
                pickle.dump(calculated_configuration, handle, protocol=pickle.HIGHEST_PROTOCOL)
            rospy.loginfo("Finished calculating Jacobian for hand part")

        lamb_jacobian = sp.lambdify(self.q, jacobian)
        lamb_Ts = [sp.lambdify(self.q, T) for T in Ts]
        return lamb_jacobian, lamb_Ts

    def getJacobian(self):
        joint_positions = []
        for joint_handle in self.joint_handles:
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_buffer)
            joint_positions.append(joint_position)

        # self.printTransformation(joint_positions=joint_positions)
        result = self.jacobian(*joint_positions)
        return result

    def printTransformation(self, joint_positions=None):
        if joint_positions is None:
            joint_positions = []
            for joint_handle in self.joint_handles:
                _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_buffer)
                joint_positions.append(joint_position)
        print(self.Ts[0](*joint_positions)[0:3, 3])
