def flattenHandPoints(data):
    flattened_data = []
    for joint_point in data.joints_position:
        flattened_data.extend([joint_point.x, joint_point.y, joint_point.z])
    return tuple(flattened_data)


def packHandPoints(data, flattened_data):
    for i in range(0, 21):
        data_point = data.joints_position[i]
        data_point.x = flattened_data[i * 3]
        data_point.y = flattened_data[i * 3 + 1]
        data_point.z = flattened_data[i * 3 + 2]
    return data
