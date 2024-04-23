import os
import pickle


def flattenHandPoints(data):
    flattened_data = []
    for joint_point in data.joints_position:
        flattened_data.extend([joint_point.x, joint_point.y, joint_point.z])
    return flattened_data


class SampleData:
    def __init__(self):
        script_path = os.path.realpath(__file__)
        self.data_path = os.path.dirname(script_path) + '/pose_samples.pkl'
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as handle:
                self.positions_to_save = pickle.load(handle)
        else:
            self.positions_to_save = []
        pass
        self.t = 0
        self.write_frequency = 1
        self.fps = 5
        self.recording_time = 10  # in sec

    def __del__(self):
        with open(self.data_path, 'wb') as handle:
            pickle.dump(self.positions_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def saveData(self, data):
        self.t += 1
        if self.t >= self.write_frequency:
            self.t = 0
            self.positions_to_save.append(flattenHandPoints(data))
            while len(self.positions_to_save) > self.recording_time * self.fps / self.write_frequency:
                self.positions_to_save.pop(0)

    def getData(self):
        return self.positions_to_save
