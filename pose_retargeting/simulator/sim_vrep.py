from pose_retargeting.simulator.simulator import Simulator, SimulatorType
from pose_retargeting.vrep_types import VRepReturn, VRepMode
import pose_retargeting.vrep as vrep
import numpy as np
import rospy
import time
from pose_retargeting.joint_handles_dict import JointHandlesDict
from pose_retargeting.jacobians.jacobian_calculation_vrep import JacobianCalculationVRep
import pose_retargeting.rotations_vrep as rotations
import logging
logger = logging.getLogger(__name__)


VRepMode2vrep = {VRepMode.BUFFER: vrep.simx_opmode_buffer, VRepMode.STREAMING: vrep.simx_opmode_streaming,
                 VRepMode.BLOCKING: vrep.simx_opmode_blocking, VRepMode.ONESHOT: vrep.simx_opmode_oneshot}


def euclideanTransformation(rotation_matrix, transformation_vector):
    if len(transformation_vector.shape) < 2:
        transformation_vector = transformation_vector[:, np.newaxis]
    top = np.concatenate((rotation_matrix, transformation_vector), axis=1)
    return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)


class VRep(Simulator):
    def __init__(self):
        super().__init__()
        self.type = SimulatorType.VREP

        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        while self.clientID == -1:
            if rospy.is_shutdown():
                exit(0)
            logger.info("No connection to remote API server, retrying...")
            vrep.simxFinish(-1)
            time.sleep(3)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        logger.info('Connected to remote API server.')

        self.joint_handles_dict = JointHandlesDict(self)
        self.hand_base_handle = self.getHandle('ShadowRobot_base_tip')
        self.hand_base_target_handle = self.getHandle('ShadowRobot_base_target')

        self.hand_target_position = np.array(self.getObjectPosition(self.hand_base_handle, -1,
                                             mode=vrep.simx_opmode_blocking))
        self.hand_target_orientation = np.array(self.getObjectQuaternion(self.hand_base_handle, parent_handle=-1,
                                                mode=vrep.simx_opmode_blocking))  # quaternion

        self.errors_in_connection = 0
        self.shift_translation = np.array([-1.5, 0., 0.25])
        self.scaling_points_knuckles = [np.array([0.033, -.0099, 0.352]), np.array([0.011, -0.0099, .356]),
                                        np.array([-.011, -.0099, .352]), np.array([-0.033, -.0099, .3436]),
                                        np.array([0.033, -0.0189, 0.286])]
        self.transformation_hand_points = [self.scaling_points_knuckles[0], self.scaling_points_knuckles[1],
                                           self.scaling_points_knuckles[2], np.array([-0.011, -0.005, 0.281])]

    def __del__(self):
        logger.info('Closing connection to remote API server.')
        vrep.simxFinish(self.clientID)

    def jacobianCalculation(self, *argv, **kwargs):
        return JacobianCalculationVRep(*argv, **kwargs)

    def simulationObjectsPose(self, handles, mode=VRepMode.BUFFER):
        current_pos = []
        for handle in handles:
            _, this_current_pos = vrep.simxGetObjectPosition(self.clientID, handle, self.hand_base_handle,
                                                             VRepMode2vrep[mode])
            current_pos.extend(this_current_pos)
        return np.array(current_pos)

    def getJointPosition(self, joint_handle, mode=VRepMode.BUFFER):
        result, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, VRepMode2vrep[mode])
        return [result == vrep.simx_return_ok, joint_position]

    def getObjectPosition(self, handle, parent_handle, **kwargs):
        return vrep.simxGetObjectPosition(self.clientID, handle, parent_handle, VRepMode2vrep[kwargs['mode']])[1]

    def getObjectPositionWithReturn(self, handle, parent_handle, mode):
        result, object_position = vrep.simxGetObjectPosition(self.clientID, handle, parent_handle, VRepMode2vrep[mode])
        return [result == vrep.simx_return_ok, object_position]

    def setObjectPosition(self, handle, base_handle, position_to_set):
        vrep.simxSetObjectPosition(self.clientID, handle, base_handle, position_to_set, vrep.simx_opmode_oneshot)

    def getObjectQuaternion(self, handle, **kwargs):
        return vrep.simxGetObjectQuaternion(self.clientID, handle, kwargs['parent_handle'],
                                            VRepMode2vrep[kwargs['mode']])[1]

    def setObjectQuaternion(self, handle, parent_handle, quaternion_to_set):
        vrep.simxSetObjectQuaternion(self.clientID, handle, parent_handle, quaternion_to_set,
                                     vrep.simx_opmode_oneshot)

    def setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.hand_target_position = target_position
        self.hand_target_orientation = target_quaternion
        self.setObjectPosition(self.hand_base_target_handle, -1, target_position.tolist())
        self.setObjectQuaternion(self.hand_base_target_handle, -1, target_quaternion.tolist())

    def getHandTargetPositionAndQuaternion(self):
        return self.hand_target_position + self.shift_translation, self.hand_target_orientation

    def removeObject(self, handle):
        vrep.simxRemoveObject(self.clientID, handle, vrep.simx_opmode_blocking)

    def createDummy(self, size, color):
        return vrep.simxCreateDummy(self.clientID, size, color, vrep.simx_opmode_blocking)[1]

    def setJointTargetVelocity(self, handle, velocity, disable_warning_on_no_connection):
        result = vrep.simxSetJointTargetVelocity(self.clientID, handle, velocity, vrep.simx_opmode_oneshot)
        if result != 0 and not disable_warning_on_no_connection:
            self.errors_in_connection += 1
            if self.errors_in_connection > 10:
                logger.warning("vrep.simxSetJointTargetVelocity return code: %d", result)
                logger.info("Probably no connection with remote API server. Exiting.")
                exit(0)
            else:
                time.sleep(0.2)

    def getObjectHandle(self, handle_name):
        result = vrep.simxGetObjectHandle(self.clientID, handle_name, vrep.simx_opmode_blocking)
        if result == vrep.simx_return_ok:
            return VRepReturn.OK
        else:
            return VRepReturn.ERROR

    def getHandBaseAction(self):
        return None

    def getJointIndexPosition(self, index):
        return 0  # not needed, just return 0

    @staticmethod
    def quat2euler(quat):
        return rotations.euler_from_quaternion(quat)

    @staticmethod
    def euler2quat(euler):
        return rotations.quaternion_from_euler(*euler)

    @staticmethod
    def mat2quat(matrix):
        return rotations.quaternion_from_matrix(matrix)
