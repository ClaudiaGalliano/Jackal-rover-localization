import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt


odom = pd.read_csv("csv/odom_data.csv")

q_x = odom.orientation_x
q_y = odom.orientation_y
q_z = odom.orientation_z
q_w = odom.orientation_w

time_sec = odom.time

roll_list = []
pitch_list = []
yaw_list = []
time_list = []

for i in range(len(odom)):

    sinr_cosp = 2 * (q_w[i]*q_x[i] + q_y[i]*q_z[i])
    cosr_cosp = 1 - 2 * (q_x[i]*q_x[i] + q_y[i]*q_y[i])
    roll_angle = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q_w[i]*q_y[i] - q_z[i]*q_x[i])
    pitch_angle = np.arcsin(sinp)

    siny_cosp = 2 * (q_w[i]*q_z[i] + q_x[i]*q_y[i])
    cosy_cosp = 1 - 2 * (q_y[i]*q_y[i] + q_z[i]*q_z[i])
    yaw_angle = np.arctan2(siny_cosp, cosy_cosp)

    while yaw_angle > math.pi:
        yaw_angle -= 2*math.pi

    while yaw_angle < -math.pi:
        yaw_angle += 2*math.pi

    roll_list.append(roll_angle)
    pitch_list.append(pitch_angle)
    yaw_list.append(yaw_angle)


roll = pd.Series(roll_list)
pitch = pd.Series(pitch_list)
yaw = pd.Series(yaw_list)

data_dictionary = {
    'time': time_sec,
    'roll': roll,
    'pitch': pitch,
    'yaw': yaw
}

data = pd.DataFrame(data_dictionary)
data.to_csv("csv/odom_orientation.csv")


plt.plot(time_sec.to_numpy(), yaw.to_numpy(), label='Theta')
plt.xlabel('Time [sec]')
plt.ylabel('Theta [rad]')
plt.grid(True)
plt.legend()
plt.show()