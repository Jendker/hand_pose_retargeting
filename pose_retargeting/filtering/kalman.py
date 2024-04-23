from pykalman import KalmanFilter
import numpy as np
from pose_retargeting.filtering.measurements.sample_data import SampleData
from pose_retargeting.filtering.common import flattenHandPoints, packHandPoints
import pickle
import os
import time

class Kalman:
    def __init__(self):
        script_path = os.path.realpath(__file__)
        self.model_path = os.path.dirname(script_path) + '/models/kalman_filter.pkl'
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as handle:
                self.kf = pickle.load(handle)
        else:
            self.kf = KalmanFilter(n_dim_obs=63, observation_matrices=self.getObservationMatrix(),
                                   transition_matrices=self.getTransitionMatrix(),
                                   observation_covariance=self.getObservationCovarianceMatrix(),
                                   em_vars=['transition_covariance', 'initial_state_covariance',
                                            'initial_state_mean', 'observation_covariance'])
            sample_data = SampleData()
            measurements = np.asarray(sample_data.getData())
            self.kf.em(measurements, n_iter=5)
            self.kf.observation_covariance *= 30  # decrease confidence in data
            try:
                os.mkdir(os.path.dirname(script_path) + '/models')
            except FileExistsError:
                pass
            with open(self.model_path, 'wb') as handle:
                pickle.dump(self.kf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.last_filtered_state_means = self.kf.initial_state_mean
        self.last_filtered_state_covariances = self.kf.initial_state_covariance
        self.last_time = time.time()
        self.positions = []

    @staticmethod
    def getObservationMatrix():
        return np.identity(63)

    @staticmethod
    def getObservationCovarianceMatrix():
        return np.identity(63)

    @staticmethod
    def getTransitionMatrix():
        return np.identity(63)

    def filter(self, data):
        flattened_data = flattenHandPoints(data)
        self.positions = flattened_data

        self.last_filtered_state_means, self.last_filtered_state_covariances = \
            self.kf.filter_update(self.last_filtered_state_means, self.last_filtered_state_covariances, flattened_data)
        return packHandPoints(data, self.last_filtered_state_means)

