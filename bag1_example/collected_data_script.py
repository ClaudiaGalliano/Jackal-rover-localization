import numpy as np
import pandas as pd


uwb = pd.read_csv("csv/uwb_data.csv")  # position_x, position_y
imu = pd.read_csv("csv/imu_data.csv")  # linear_acceleration_x, linear_acceleration_y
imu_orientation = pd.read_csv("csv/imu_orientation.csv") # yaw
odom = pd.read_csv("csv/odom_data.csv")  # linear_velocity_x, linear_velocity_y, angular_velocity_z
odom_orientation = pd.read_csv("csv/odom_orientation.csv") # yaw

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
d1_uwb = uwb.distance_44AE
d2_uwb = uwb.distance_06A5
d3_uwb = uwb.distance_CBB0
d4_uwb = uwb.distance_4F9B

d1_list = [0] * len(time)  # creo una lista di 0 lunga quanto time
d2_list = [0] * len(time)
d3_list = [0] * len(time) 
d4_list = [0] * len(time)
for i in range(len(uwb)):
    for k in range(len(time)):
        if uwb_time[i] == time[k]:
            d1_list[k] = d1_uwb[i]
            d2_list[k] = d2_uwb[i]
            d3_list[k] = d3_uwb[i]
            d4_list[k] = d4_uwb[i]
d1 = pd.Series(d1_list)
d2 = pd.Series(d2_list)
d3 = pd.Series(d3_list)
d4 = pd.Series(d4_list)

# IMU
yaw_imu = imu_orientation.yaw
a_x_imu = imu.linear_acceleration_x
a_y_imu = imu.linear_acceleration_y
w_imu = imu.angular_velocity_z

yaw_list = [0] * len(time)
a_x_list = [0] * len(time)
a_y_list = [0] * len(time)
w_list_imu = [0] * len(time)

for i in range(len(imu)):
    for k in range(len(time)):
        if imu_time[i] == time[k]:
            yaw_list[k] = yaw_imu[i]
            a_x_list[k] = a_x_imu[i]
            a_y_list[k] = a_y_imu[i]
            w_list_imu[k] = w_imu[i]
yaw = pd.Series(yaw_list)
a_x= pd.Series(a_x_list)
a_y = pd.Series(a_y_list)
w_imu = pd.Series(w_list_imu)

# ODOM
v_x_odom = odom.linear_velocity_x
v_y_odom = odom.linear_velocity_y
w_odom = odom.angular_velocity_z
yaw_odom = odom_orientation.yaw

v_x_list = [0] * len(time)
v_y_list = [0] * len(time)
w_list = [0] * len(time)
yaw_list_odom = [0] * len(time)

for i in range(len(odom)):
    for k in range(len(time)):
        if odom_time[i] == time[k]:
            v_x_list[k] = v_x_odom[i]
            v_y_list[k] = v_y_odom[i]
            w_list[k] = w_odom[i]
            yaw_list_odom[k] = yaw_odom[i]
v_x = pd.Series(v_x_list)
v_y= pd.Series(v_y_list)
w = pd.Series(w_list)
yaw_odom = pd.Series(yaw_list_odom)


data_dictionary = {
    'time': time,
    'distance_44AE': d1,
    'distance_06A5': d2,
    'distance_CBB0': d3,
    'distance_4F9B': d4,
    'yaw_imu': yaw,
    'yaw_odom': yaw_odom,
    'v_x': v_x,
    'v_y': v_y,
    'w_odom': w,
    'w_imu': w_imu,
    'a_x': a_x,
    'a_y': a_y
}

data = pd.DataFrame(data_dictionary)
data.to_csv("collected_data.csv")

