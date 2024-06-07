import pandas as pd
import numpy as np


vicon = pd.read_csv("csv/vicon_data.csv")

time_sec = vicon.time
pos_x = vicon.position_x
pos_y = vicon.position_y
q_x = vicon.orientation_x
q_y = vicon.orientation_y
q_z = vicon.orientation_z
q_w = vicon.orientation_w

roll_list = []
pitch_list = []
yaw_list = []
time_sec_list = []
pos_x_list = []
pos_y_list = []

for i in range(len(vicon)):

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

    operation = time_sec[i] * 10**(-9)
    time_sec_list.append(operation)

    transf_x = pos_x[i] * 10**(-3)  # vicon pos data is in mm
    transf_y = pos_y[i] * 10**(-3)
    pos_x_list.append(transf_x)
    pos_y_list.append(transf_y)


roll = pd.Series(roll_list)
pitch = pd.Series(pitch_list)
yaw = pd.Series(yaw_list)
time = pd.Series(time_sec_list)
x = pd.Series(pos_x_list)
y = pd.Series(pos_y_list)


# Create a DataFrame
data = {
    'time': time,
    'x': x,
    'y': y,
    'roll': roll,
    'pitch': pitch,
    'yaw': yaw
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file = 'ground_truth_data.csv'
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")