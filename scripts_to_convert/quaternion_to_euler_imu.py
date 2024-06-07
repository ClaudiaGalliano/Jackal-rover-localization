import pandas as pd 
import numpy as np


imu = pd.read_csv("csv/imu_data.csv")

q_x = imu.orientation_x
q_y = imu.orientation_y
q_z = imu.orientation_z
q_w = imu.orientation_w

time_sec = imu.time_sec
time_nanosec = imu.time_nanosec

roll_list = []
pitch_list = []
yaw_list = []
time_list = []

for i in range(len(imu)):

    sinr_cosp = 2 * (q_w[i]*q_x[i] + q_y[i]*q_z[i])
    cosr_cosp = 1 - 2 * (q_x[i]*q_x[i] + q_y[i]*q_y[i])
    roll_angle = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q_w[i]*q_y[i] - q_z[i]*q_x[i])
    pitch_angle = np.arcsin(sinp)

    siny_cosp = 2 * (q_w[i]*q_z[i] + q_x[i]*q_y[i])
    cosy_cosp = 1 - 2 * (q_y[i]*q_y[i] + q_z[i]*q_z[i])
    yaw_angle = np.arctan2(siny_cosp, cosy_cosp)

    roll_list.append(roll_angle)
    pitch_list.append(pitch_angle)
    yaw_list.append(yaw_angle)

    operation = time_sec[i] + time_nanosec[i] * (10**(-9)) 
    time_list.append(operation)


roll = pd.Series(roll_list)
pitch = pd.Series(pitch_list)
yaw = pd.Series(yaw_list)
time = pd.Series(time_list)

data_dictionary = {
    'time': time,
    'roll': roll,
    'pitch': pitch,
    'yaw': yaw
}

data = pd.DataFrame(data_dictionary)
data.to_csv("imu_orientation.csv")

print(data.yaw)