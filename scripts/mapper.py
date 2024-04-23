#!/usr/bin/env python

import vrep
import numpy as np
import time
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import MarkerArray, Marker
import geometry_msgs.msg
import tf_conversions
from hand import Hand
from joint_handles_dict import JointHandlesDict
from FPS_counter import FPSCounter
from scaler import Scaler


class Mapper:
    def __init__(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        while self.clientID == -1:
            if rospy.is_shutdown():
                return
            rospy.loginfo("No connection to remote API server, retrying...")
            vrep.simxFinish(-1)
            time.sleep(3)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
        rospy.loginfo('Connected to remote API server')

        self.last_callback_time = 0  # 0 means no callback yet
        self.last_data = []
        self.node_frame_name = "hand_vrep"
        self.camera_frame_name = "camera_link"
        self.last_update = time.time()
        self.using_left_hand = rospy.get_param('transformation/left_hand')
        self.shift_translation = np.array([-1.5, 0., 0.25])

        self.marker_pub = rospy.Publisher('pose_mapping_vrep/transformed_hand', MarkerArray, queue_size=10)
        self.points_pub = rospy.Publisher('pose_mapping_vrep/in_base', PointCloud, queue_size=10)
        self.tf_listener_ = tf.TransformListener()
        self.errors_in_connection = 0
        self.alpha = 0.2
        self.joint_handles_dict = JointHandlesDict(self.clientID)
        self.hand = Hand(self.clientID, self.alpha, self.joint_handles_dict)
        self.sampling_time = 0.05

        self.hand_base_target_handle = self.joint_handles_dict.getHandle('ShadowRobot_base_target')
        _, self.last_quaternion = vrep.simxGetObjectQuaternion(self.clientID, self.hand_base_target_handle, -1, vrep.simx_opmode_blocking)
        self.last_quaternion = np.array(self.last_quaternion)
        _, self.last_position = vrep.simxGetObjectPosition(self.clientID, self.hand_base_target_handle, -1, vrep.simx_opmode_blocking)
        self.last_position = np.array(self.last_position)

        self.FPSCounter = FPSCounter()
        self.scaler = Scaler()
        rospy.loginfo("Pose mapping initialization finished.")

    def __del__(self):
        del self.hand  # not deleted properly, so executing explicitly
        # Close the connection to V-REP:
        vrep.simxFinish(self.clientID)

    def __euclideanTransformation(self, rotationMatrix, transformationVector):
        top = np.concatenate((rotationMatrix, transformationVector[:, np.newaxis]), axis=1)
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

    def __returnTransformation(self, data, inverse_transformation_matrix):
        points_return = []
        for point in data.joints_position:
            vector_transformed = np.dot(inverse_transformation_matrix, np.append(point, [1]))[0:3]\
                                 + self.shift_translation  # shift to keep hand above surface
            points_return.append(vector_transformed)
        data_return = data
        data_return.joints_position = points_return
        data_return.header.frame_id = self.camera_frame_name
        return data_return

    def __getPositionVectorForDataIndex(self, data, index):
        joint = data.joints_position[index]
        position = [joint.x, joint.y, joint.z]
        return np.array(position)

    def __publishMarkers(self, data):
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

    def __publishTransformation(self, data):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = data.header.frame_id
        t.child_frame_id = self.node_frame_name

        finger_indices = [2, 3, 4, 0]  # index, middle, ring, thumb
        world_points = []
        for index in finger_indices:
            world_points.append(self.__getPositionVectorForDataIndex(data, index))

        vrep_points = [np.array([0.033, -.0099, 0.352]), np.array([0.011, -0.0099, .356]), np.array([-.011, -.0099, .352]),
                       np.array([-0.011, -0.005, 0.281])]

        transformation_matrix, correct_rotation_matrix = self.__correspondancePointsTransformation(np.array(world_points), np.array(vrep_points))

        # transform publishing
        t.transform.translation.x = transformation_matrix[0, 3]
        t.transform.translation.y = transformation_matrix[1, 3]
        t.transform.translation.z = transformation_matrix[2, 3]

        q = tf_conversions.transformations.quaternion_from_matrix(correct_rotation_matrix)
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

    def __setHandPosition(self, transformation_matrix):
        inverse_rotation_matrix = np.linalg.inv(transformation_matrix[0:3, 0:3])
        translation = transformation_matrix[0:3, 3]
        inverse_translation = -np.dot(inverse_rotation_matrix, translation)
        inverse_transformation_matrix = self.__euclideanTransformation(inverse_rotation_matrix, inverse_translation)
        q = tf_conversions.transformations.quaternion_from_matrix(inverse_transformation_matrix)
        whole_translation = inverse_translation + self.shift_translation  # shift to keep hand above surface
        self.last_position = whole_translation * self.alpha + self.last_position * (1. - self.alpha)
        vrep.simxSetObjectPosition(self.clientID, self.hand_base_target_handle, -1, self.last_position.tolist(), vrep.simx_opmode_oneshot)
        q = np.array(q) / np.linalg.norm(q)
        self.last_quaternion = q * self.alpha + self.last_quaternion * (1 - self.alpha)
        self.last_quaternion = self.last_quaternion / np.linalg.norm(self.last_quaternion)
        vrep.simxSetObjectQuaternion(self.clientID, self.hand_base_target_handle, -1, self.last_quaternion.tolist(), vrep.simx_opmode_oneshot)
        return inverse_transformation_matrix

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

    def callback(self, data):
        if self.using_left_hand:
            data = self.__mirrorData(data)
        data = self.__transformToCameraLink(data)
        transformation_matrix = self.__publishTransformation(data)
        data = self.__transformDataWithTransform(data, transformation_matrix)
        # self.__publishMarkers(data)  # to visualize results
        # self.publishNewPointCloud(data)  # to visualize results
        data.joints_position = self.scaler.scalePoints(data.joints_position)  # ready to save after scaling
        self.__setHandPosition(transformation_matrix)
        self.last_data = data

        self.hand.newPositionFromHPE(self.last_data)

    def __executeInverseOnce(self):
        self.hand.controlOnce()

    def execute(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            self.__executeInverseOnce()
            self.FPSCounter.printFPS()
            time.sleep(self.sampling_time - ((time.time() - start_time) % self.sampling_time))
            self.last_update = time.time()
