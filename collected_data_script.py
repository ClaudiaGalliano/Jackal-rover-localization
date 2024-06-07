import numpy as np
import pandas as pd


uwb = pd.read_csv("csv/uwb_position.csv")  # position_x, position_y
imu = pd.read_csv("csv/imu_data.csv")  # linear_acceleration_x, linear_acceleration_y, angular_acceleration_z
imu_orientation = pd.read_csv("csv/imu_orientation.csv") # yaw
odom = pd.read_csv("csv/odom_data.csv")  # linear_velocity_x, linear_velocity_y, angular_velocity_z

imu_time = imu_orientation.time
uwb_time = uwb.time
odom_time = odom.time

imu_time_list = imu_time.tolist()
uwb_time_list = uwb_time.tolist()
odom_time_list = odom_time.tolist()

time_list = imu_time_list + uwb_time_list + odom_time_list
time_list.sort()
time = pd.Series(time_list)

# UWB
x_pos_uwb = uwb.position_x
y_pos_uwb = uwb.position_y

x_list = [0] * len(time)  # creo una lista di 0 lunga quanto time
y_list = [0] * len(time)
for i in range(len(uwb)):
    for k in range(len(time)):
        if uwb_time[i] == time[k]:
            x_list[k] = x_pos_uwb[i]
            y_list[k] = y_pos_uwb[i]
x = pd.Series(x_list)
y = pd.Series(y_list)

# IMU
yaw_imu = imu_orientation.yaw
a_x_imu = imu.linear_acceleration_x
a_y_imu = imu.linear_acceleration_y

yaw_list = [0] * len(time)
a_x_list = [0] * len(time)
a_y_list = [0] * len(time)

for i in range(len(imu)):
    for k in range(len(time)):
        if imu_time[i] == time[k]:
            yaw_list[k] = yaw_imu[i]
            a_x_list[k] = a_x_imu[i]
            a_y_list[k] = a_y_imu[i]
yaw = pd.Series(yaw_list)
a_x= pd.Series(a_x_list)
a_y = pd.Series(a_y_list)

# ODOM
v_x_odom = odom.linear_velocity_x
v_y_odom = odom.linear_velocity_y
w_odom = odom.angular_velocity_z

v_x_list = [0] * len(time)
v_y_list = [0] * len(time)
w_list = [0] * len(time)

for i in range(len(odom)):
    for k in range(len(time)):
        if odom_time[i] == time[k]:
            v_x_list[k] = v_x_odom[i]
            v_y_list[k] = v_y_odom[i]
            w_list[k] = w_odom[i]
v_x = pd.Series(v_x_list)
v_y= pd.Series(v_y_list)
w = pd.Series(w_list)


data_dictionary = {
    'time': time,
    'x': x,
    'y': y,
    'yaw': yaw,
    'v_x': v_x,
    'v_y': v_y,
    'w': w,
    'a_x': a_x,
    'a_y': a_y
}

data = pd.DataFrame(data_dictionary)
data.to_csv("collected_data.csv")

