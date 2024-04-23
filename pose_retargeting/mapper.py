#!/usr/bin/env python

import numpy as np
import time
try:
    import rospy
    import tf2_ros
    import tf
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import PointCloud
    from visualization_msgs.msg import MarkerArray, Marker
    from dl_pose_estimation.msg import JointsPosition
    import geometry_msgs.msg
except ImportError:
    pass
from pose_retargeting.rotations_vrep import quaternion_from_matrix
from pose_retargeting.hand import Hand
from pose_retargeting.FPS_counter import FPSCounter
from pose_retargeting.scaler import Scaler
from pose_retargeting.filtering.kalman import Kalman
from pose_retargeting.optimization.pso import PSO
from pose_retargeting.optimization.nn_optimize import NN_optimize
from pose_retargeting.simulator.sim_mujoco import euclideanTransformation
import logging
logger = logging.getLogger(__name__)


class Mapper:
    def __init__(self, node_name, simulator=None, use_PSO=False, use_nn_optimize=True, optimization=True):
        self.last_callback_time = 0  # 0 means no callback yet
        self.node_frame_name = "hand_vrep"
        self.camera_frame_name = "camera_link"
        self.last_update = time.time()
        try:
            self.using_left_hand = rospy.get_param('transformation/left_hand', False)
        except NameError:
            self.using_left_hand = False

        self.data_filtering = Kalman()

        try:
            self.marker_pub = rospy.Publisher(node_name + '/transformed_hand', MarkerArray, queue_size=10)
            self.points_pub = rospy.Publisher(node_name + '/in_base', PointCloud, queue_size=10)
            self.tf_listener_ = tf.TransformListener()
        except NameError:
            pass
        self.errors_in_connection = 0
        self.simulator = simulator
        if self.simulator is None:
            from pose_retargeting.simulator.sim_vrep import VRep
            self.simulator = VRep()

        self.hand = Hand(self.simulator)
        self.sampling_time = 0.05

        self.FPSCounter = FPSCounter()
        self.scaler = Scaler(self.simulator)
        logger.info("Pose mapping initialization finished.")

        self.PSO = None
        self.NN_optimize = None
        if optimization:
            if use_PSO:
                self.PSO = PSO(self.simulator)
            elif use_nn_optimize:
                self.NN_optimize = NN_optimize()
        self.start_rotation_base = self.simulator.getHandBaseRotationMatrix().copy()
        self.translation_base = self.simulator.getHandBasePosition().copy()
        self.translation_base /= 4  # moves the hand a bit forward
        self.inverse_start_transformation_base = euclideanTransformation(self.start_rotation_base.T,
                                                                         -self.start_rotation_base.T
                                                                         @ self.translation_base)
        rotate_hand = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
                                [np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
                                [0, 0, 1]])
        self.rotate_hand = self.__euclideanTransformation(rotate_hand, np.zeros(3))  # rotates hand by 90 degrees to match
                                                                                     # the required orientation
        self.transformation_to_origin_from_demo = None

    def __euclideanTransformation(self, rotation_matrix, transformation_vector):
        top = np.concatenate((rotation_matrix, transformation_vector[:, np.newaxis]), axis=1)
        return np.concatenate((top, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)

    def __transformDataWithTransform(self, data, transformation_matrix):
        points_return = []
        for index, _ in enumerate(data.joints_position):
            vector_transformed = np.dot(transformation_matrix,
                                        np.append(self.__getPositionVectorForDataIndex(data, index), [1]))[0:3]
            points_return.append(vector_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = self.node_frame_name
        data_return.header.stamp = rospy.Time.now()
        return data_return

    def __getPositionVectorForDataIndex(self, data, index):
        joint = data.joints_position[index]
        position = [joint.x, joint.y, joint.z]
        return np.array(position)

    def __publishMarkers(self, data, transformation_matrix=None):
        message = MarkerArray()
        lines = [[2, 9, 10, 11], [3, 12, 13, 14], [4, 15, 16, 17], [5, 18, 19, 20], [6, 7, 8], [0, 1, 6, 2, 3, 4, 5, 0]]
        time_now = rospy.Time.now()
        for index, line_points in enumerate(lines):
            line_marker = Marker()
            line_marker.header.frame_id = self.camera_frame_name
            line_marker.header.stamp = time_now
            line_marker.id = index
            line_marker.color.r = 1.0
            # message.color.g = 1.0
            line_marker.color.a = 1.0
            line_marker.scale.x = line_marker.scale.y = line_marker.scale.z = 0.01
            line_marker.type = line_marker.LINE_STRIP

            finger_points = [data.joints_position[x] for x in line_points]

            for point in finger_points:
                if transformation_matrix is not None:
                    point = (transformation_matrix @ np.append(point, 1))[0:3]
                message_point = Point()
                message_point.x = point[0]
                message_point.y = point[1]
                message_point.z = point[2]
                line_marker.points.append(message_point)
            message.markers.append(line_marker)
        self.marker_pub.publish(message)

    def __correspondancePointsTransformation(self, from_set, to_set):
        num = from_set.shape[0]
        dim = from_set.shape[1]
        mean_from_set = from_set.mean(axis=0)
        from_zero_mean_set = from_set - mean_from_set
        mean_to_set = to_set.mean(axis=0)
        to_zero_mean_set = to_set - mean_to_set

        D = np.dot(to_zero_mean_set.T, from_zero_mean_set) / num  # eq. 38

        from_set_variance = from_set.var(axis=0).sum()

        U, Sigma, Vt = np.linalg.svd(D)
        rank = np.linalg.matrix_rank(D)
        S = np.identity(dim, dtype=np.double)
        if np.linalg.det(D) < 0:
            S[dim-1, dim-1] = -1.
        if rank == 0:
            return np.full([dim+1, dim+1], np.nan)
        if rank >= dim-1:
            scaling = 1. / from_set_variance * np.trace(np.dot(np.diag(Sigma), S))
            if rank == 2:
                if np.isclose(np.linalg.det(U) * np.linalg.det(Vt), -1., 0.00001):
                    S[dim-1, dim-1] = -1
                else:
                    S[dim-1, dim-1] = 1
            rotation_matrix = np.linalg.multi_dot([U, S, Vt])
            translation = mean_to_set - scaling * np.dot(rotation_matrix, mean_from_set)
            correct_rotation_matrix = rotation_matrix.copy()
            rotation_matrix *= scaling
            return self.__euclideanTransformation(rotation_matrix, translation),  self.__euclideanTransformation(correct_rotation_matrix, translation)
        else:
            rospy.logfatal("Transformation between the points not defined!")
            exit(1)

    def __getAndPublishTransformation(self, data):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = data.header.frame_id
        t.child_frame_id = self.node_frame_name

        finger_indices = [2, 3, 4, 0]  # index, middle, ring, thumb
        world_points = []
        for index in finger_indices:
            world_points.append(self.__getPositionVectorForDataIndex(data, index))

        simulator_transformation_hand_points = self.simulator.transformation_hand_points

        transformation_matrix, correct_rotation_matrix = self.__correspondancePointsTransformation(
            np.array(world_points), np.array(simulator_transformation_hand_points))

        # transform publishing
        t.transform.translation.x = transformation_matrix[0, 3]
        t.transform.translation.y = transformation_matrix[1, 3]
        t.transform.translation.z = transformation_matrix[2, 3]

        q = quaternion_from_matrix(correct_rotation_matrix)  # here we need to keep rotations_vrep, which comes from ROS
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)
        return transformation_matrix

    def __mirrorData(self, data):
        new_data = data
        for index, joint_position in enumerate(new_data.joints_position):
            new_data.joints_position[index].x = -new_data.joints_position[index].x

        return data

    def publishNewPointCloud(self, data):
        point_cloud = PointCloud()
        point_cloud.header.frame_id = self.camera_frame_name
        point_cloud.header.stamp = data.header.stamp
        for point in data.joints_position:
            pc_point = Point()
            pc_point.x = point[0]
            pc_point.y = point[1]
            pc_point.z = point[2]
            point_cloud.points.append(pc_point)
        self.points_pub.publish(point_cloud)

    def __setHandPosition(self, position, quaternion):
        self.simulator.setHandTargetPositionAndQuaternion(position, quaternion)

    def __transformToCameraLink(self, data):
        target_frame = self.camera_frame_name
        from_frame = data.header.frame_id
        self.tf_listener_.waitForTransform(from_frame, target_frame, rospy.Time(0), rospy.Duration(4))

        point_cloud = PointCloud()
        point_cloud.header.frame_id = from_frame
        for point in data.joints_position:
            point_cloud.points.append(point)

        point_cloud_in_target = self.tf_listener_.transformPointCloud(target_frame, point_cloud)
        return_points = data
        for index, point in enumerate(point_cloud_in_target.points):
            return_points.joints_position[index] = point
        return_points.header.frame_id = target_frame
        return return_points

    def pointToNpArray(self, point):
        return np.array([point.x, point.y, point.z])

    @staticmethod
    def __unPackHandPointsMatrix(data):
        ret = np.empty((21, 3))
        for i in range(0, 21):
            ret[i, :] = data.joints_position[i]
        return ret

    def _getHandBasePosRotFromTransformationMatrix(self, transformation_matrix):
        hand_position = transformation_matrix[0:3, 3]
        hand_quaternion = self.simulator.mat2quat(transformation_matrix)
        return hand_position, hand_quaternion

    def __transformTransformationMatrix(self, transformation_matrix):
        inverse_rotation_matrix = np.linalg.inv(transformation_matrix[0:3, 0:3])
        translation = transformation_matrix[0:3, 3]
        inverse_translation = -np.dot(inverse_rotation_matrix, translation)
        inverse_transformation_matrix = self.__euclideanTransformation(inverse_rotation_matrix, inverse_translation)
        if self.transformation_to_origin_from_demo is None:
            self.transformation_to_origin_from_demo = np.identity(4)
            self.transformation_to_origin_from_demo[0:3, 3] = -inverse_transformation_matrix[0:3, 3]\
                                                              + np.array([0, 0, 0.1])


        new_transformation_matrix = self.inverse_start_transformation_base @ self.rotate_hand @ \
                                    self.transformation_to_origin_from_demo @ \
                                    inverse_transformation_matrix  # in rotated hand coordinates with shift of origin
                                                                   # to place where hand was on the beginning of demo
        return new_transformation_matrix

    def callback(self, data):
        if self.using_left_hand:
            data = self.__mirrorData(data)
        data = self.__transformToCameraLink(data)
        data = self.data_filtering.filter(data)

        transformation_matrix = self.__getAndPublishTransformation(data)  # get to not rotated hand coordinates
        data = self.__transformDataWithTransform(data, transformation_matrix)
        data.joints_position = self.scaler.scalePoints(data.joints_position)  # ready to save after scaling
        new_transformation_matrix = self.__transformTransformationMatrix(transformation_matrix)
        position, quaternion = self._getHandBasePosRotFromTransformationMatrix(new_transformation_matrix)
        self.__setHandPosition(position, quaternion)
        # self.__publishMarkers(data, inverse_transformation_matrix)  # to visualize results
        # self.publishNewPointCloud(data)  # to visualize results
        data = self.__unPackHandPointsMatrix(data)
        self._newPositionFromHPE(data, position, quaternion)

    def newHandPointsData(self, data):
        self.__setHandPosition(position=data['base_pose'][0:3], quaternion=data['base_pose'][3:])
        self._newPositionFromHPE(data['finger_points'], data['base_pose'][0:3], data['base_pose'][3:])

    def _newPositionFromHPE(self, data, position, quaternion):
        self.hand.newPositionFromHPE(data)
        if self.PSO is not None:
            self.PSO.new_taget_pose(data, position, quaternion)

    def getControlOnce(self):
        self.FPSCounter.getAndPrintFPS()
        return self.hand.getControlOnce()

    def getClampedControlOnce(self, observation):
        self.FPSCounter.getAndPrintFPS()
        actions = self.hand.getControlOnce()
        actions = self.simulator.clampActions(actions)
        if self.PSO is not None:
            actions = self.PSO.optimize(actions, self.simulator)
        if self.NN_optimize is not None:
            actions = self.NN_optimize.optimize(observation, actions, self.simulator)
        return actions

    def __executeInverseOnce(self):
        self.hand.controlOnce()

    def execute(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            self.__executeInverseOnce()
            self.FPSCounter.getAndPrintFPS()
            time.sleep(self.sampling_time - ((time.time() - start_time) % self.sampling_time))
            self.last_update = time.time()
