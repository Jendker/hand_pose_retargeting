from pose_retargeting.simulator.simulator import Simulator, SimulatorType
from pose_retargeting.vrep_types import VRepMode
import numpy as np
import pose_retargeting.rotations_mujoco as rotations
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_mujoco import JacobianCalculationMujoco
import sched
import time
import threading
from mujoco_py import const


def euclideanTransformation(rotation_matrix, transformation_vector):
    if len(transformation_vector.shape) < 2:
        transformation_vector = transformation_vector[:, np.newaxis]
    top = np.concatenate((rotation_matrix, transformation_vector), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class Mujoco(Simulator):
    def __init__(self, env, env_name=None, visualisation=False):
        super().__init__()
        self.type = SimulatorType.MUJOCO
        self.visualisation = visualisation
        try:
            self.env = env.env.env
        except AttributeError:
            self.env = env.env
        self.last_observations = []
        if env_name is None:
            self.env_name = env.env.unwrapped.spec.id
        else:
            self.env_name = env_name

        self.limits_hand_orientation = ((-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14))

        self.joint_handles_dict = JointHandlesDict(self)
        self.hand_base_name = self.getHandle('ShadowRobot_base_tip')
        self.hand_base_index = self.env.model.body_names.index(self.hand_base_name)
        self.hand_target_position = np.array([0, 0, 0])
        self.hand_target_orientation = np.array([0, 0, 0])  # here euler because we set action angle with euler
        self.scaling_points_knuckles = self.__getKnucklesPositions()
        self.transformation_hand_points = [self.scaling_points_knuckles[0], self.scaling_points_knuckles[1],
                                           self.scaling_points_knuckles[2], np.array([-0.011, -0.005, 0.271])]
        if self.visualisation:
            self.scheduler = sched.scheduler(time.time, time.sleep)
            setup_viewer_thread = threading.Thread(target=self.__setupViewer)
            setup_viewer_thread.start()
        self.act_mid = np.mean(self.env.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.env.model.actuator_ctrlrange[:,1]-self.env.model.actuator_ctrlrange[:,0])

    def __setupViewer(self):
        try:
            self.viewer = self.env.viewer
        except AttributeError:
            self.scheduler.enter(0.1, 1, self.__setupViewer)
            self.scheduler.run()

    def __getKnucklesPositions(self):  # only to run at startup, because metacarpal angle may change
        ret = []
        knuckles_handles = self.getHandles(['IMCP_side_joint', 'MMCP_side_joint', 'RMCP_side_joint', 'PMCP_side_joint',
                                            'TMCP_rotation_joint'])
        for knuckle_handle in knuckles_handles:
            ret.append(self.getObjectPosition(knuckle_handle, self.hand_base_name))
        return ret

    def __getInverseTransformationMatrixToBase(self):
        """Returns transformation from world to hand base coordinate system"""
        rotation_matrix = self.env.data.body_xmat[self.hand_base_index].reshape((3, 3))
        translation = self.env.data.body_xpos[self.hand_base_index]
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    def getTransformationMatrixToBase(self):
        """Returns transformation from hand base coordinate system to world coordinate system"""
        rotation_matrix = self.env.data.body_xmat[self.hand_base_index].reshape((3, 3))
        translation = self.env.data.body_xpos[self.hand_base_index]
        return euclideanTransformation(rotation_matrix, translation)

    def __getInverseTransformationMatrix(self, handle):
        idx = self.env.model.body_names.index(handle)
        rotation_matrix = self.env.data.body_xmat[idx].reshape((3, 3))
        translation = self.env.data.body_xpos[idx]
        return euclideanTransformation(rotation_matrix.T, np.dot(-rotation_matrix.T, translation))

    def clampActions(self, actions):
        return (actions - self.act_mid) / self.act_rng

    def unclampActions(self, actions):
        return actions * self.act_rng + self.act_mid

    @staticmethod
    def quat2euler(quat):
        return rotations.quat2euler(quat)

    @staticmethod
    def euler2quat(euler):
        return rotations.euler2quat(euler)

    @staticmethod
    def mat2quat(matrix):
        if matrix.shape == (4, 4):
            matrix = matrix[:3, :3]
        return rotations.mat2quat(matrix)

    def jacobianCalculation(self, *argv, **kwargs):
        return JacobianCalculationMujoco(*argv, **kwargs)

    def simulationObjectsPose(self, body_names, mode=VRepMode.BUFFER):
        """
        Returns single vector of bodies pose in hand base coordinate system
        :param body_names:
        :param mode: defined for backward compability with VRep
        :return:
        """
        if mode != VRepMode.BUFFER and mode != VRepMode.BLOCKING:
            return
        return np.concatenate(self.simulationObjectsPoseList(body_names), axis=None)

    def simulationObjectsPoseList(self, body_names, mode=VRepMode.BUFFER):
        """
        Returns list of bodies pose in hand base coordinate system
        :param body_names:
        :param mode: defined for backward compability with VRep
        :return:
        """
        if mode != VRepMode.BUFFER and mode != VRepMode.BLOCKING:
            return
        current_pos = []
        inverse_transformation_matrix = self.__getInverseTransformationMatrixToBase()
        for body_name in body_names:
            idx = self.env.model.body_names.index(body_name)
            this_current_pos = self.env.data.body_xpos[idx]
            current_pos.append(np.dot(inverse_transformation_matrix, np.append(this_current_pos, [1]))[0:3])
        return current_pos

    def getJointPosition(self, body_name, mode=VRepMode.BUFFER):
        if mode != VRepMode.BUFFER and mode != VRepMode.BLOCKING:
            return
        idx = self.env.model.joint_names.index(self.getBodyJointName(body_name))
        return [True, self.env.data.qpos[idx]]

    def getJointIndexPosition(self, index):
        assert (len(self.env.data.qpos) == len(self.env.data.qvel))  # need to make sure, for some envs this is not the same
        return self.env.data.qpos[index]

    def getJointNamePosition(self, joint_name):
        idx = self.env.model.joint_names.index(joint_name)
        return self.env.data.qpos[idx]

    def getObjectPosition(self, body_name, parent_handle, **kwargs):
        idx = self.env.model.body_names.index(body_name)
        return self.getObjectIndexPosition(idx, parent_handle)

    def getObjectPositionWithReturn(self, handle, parent_handle, mode=None):
        return [True, self.getObjectPosition(handle, parent_handle, mode=mode)]

    def getObjectIndexPosition(self, index, parent_handle, mode=None):
        if mode == VRepMode.STREAMING:
            return None
        current_pos = self.env.data.body_xpos[index]
        if parent_handle == -1:
            return current_pos
        else:
            transformation_matrix = self.__getInverseTransformationMatrix(parent_handle)
            return np.dot(transformation_matrix, np.append(current_pos, [1]))[0:3]

    def setObjectPosition(self, handle, base_handle, position_to_set):
        raise NotImplementedError  # not needed

    def getObjectQuaternion(self, handle, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        idx = self.env.model.body_names.index(handle)
        return self.env.data.body_xquat[idx]

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        raise NotImplementedError  # not needed

    def getObjectIndexQuaternion(self, index, **kwargs):
        try:
            if kwargs['parent_handle'] != -1:
                raise NotImplementedError
        except KeyError:
            pass
        return self.env.data.body_xquat[index]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        raise NotImplementedError

    def setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = self.quat2euler(target_quaternion)

    def getHandTargetPositionAndQuaternion(self):
        return self.hand_target_position, self.euler2quat(self.hand_target_orientation)

    def removeObject(self, handle):
        pass

    def createDummy(self, size, color):
        return None

    def getJointIndex(self, body_name):
        return self.env.model.joint_names.index(self.getBodyJointName(body_name))

    def getJointNameIndex(self, joint_name):
        return self.env.model.joint_names.index(joint_name)

    def getBodyJointName(self, body_name):
        return self.joint_handles_dict.getBodyJointName(body_name)

    def getHPEIndexEquivalentBody(self, index):
        return self.joint_handles_dict.getHPEIndexEquivalentBody(index)

    def getJacobianFromBodyName(self, body_name):
        return self.env.data.get_body_jacp(body_name).reshape(3, -1)

    def applyLimitsOfOrientation(self, old_angles):
        new_angles = old_angles.copy()
        for i in range(0, 3):
            if new_angles[i] < self.limits_hand_orientation[i][0]:
                new_angles[i] += 3.1416 * 2
            elif new_angles[i] > self.limits_hand_orientation[i][1]:
                new_angles[i] -= 3.1416 * 2
        return new_angles

    @staticmethod
    def inverseUpdateHandPosition(old_position):
        new_position = old_position.copy()
        new_position[0] += 1.2
        return new_position

    def getHandBaseAction(self):
        return np.concatenate((self.hand_target_position,
                               self.applyLimitsOfOrientation(self.hand_target_orientation)))

    def getNumberOfJoints(self):
        return self.env.data.ctrl.size

    def getJointNameVelocity(self, joint_name):
        idx = self.env.model.joint_names.index(joint_name)
        return self.env.data.qvel[idx]

    def getJointIndexVelocity(self, index):
        return self.env.data.qvel[index]

    def getJointLimits(self, body_name):
        idx = self.getJointIndex(body_name)
        return self.env.action_space.high[idx], self.env.action_space.low[idx]

    @staticmethod
    def __transformPoint(point, transformation_matrix):
        return np.dot(transformation_matrix, np.append(point, [1]))[0:3]

    def visualisePose(self, poses):
        if self.visualisation:
            transformation_matrix = self.getTransformationMatrixToBase()
            try:
                for i, pose in enumerate(poses):
                    self.viewer.add_marker(pos=self.__transformPoint(pose, transformation_matrix), type=const.GEOM_SPHERE,
                                           size=np.ones(3) * 0.008, label='',
                                           rgba=np.array([1 * (i % 3), 1 * ((i + 1) % 3), 1 * ((i + 2) % 3), 0.6]))
            except (AttributeError, TypeError):
                pass

    def getHandBaseRotationMatrix(self):
        return self.env.data.body_xmat[self.hand_base_index].reshape((3, 3))

    def getHandBasePosition(self):
        return self.env.data.body_xpos[self.hand_base_index]
