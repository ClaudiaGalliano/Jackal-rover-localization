import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


# in m
T = np.array(
    [
        [0.81907688, 0.51108873, 0.26057892, -0.03416372315],
        [-0.49096203, 0.85945434, -0.14245887, 0.00026770578],
        [-0.29676481, -0.01124959, 0.95488434, 0.02656486081],
        [0, 0, 0, 1],
    ]
)

# rot1 = np.array(
#     [
#         [0.81907688, 0.51108873, 0.26057892],
#         [-0.49096203, 0.85945434, -0.14245887],
#         [-0.29676481, -0.01124959, 0.95488434],
#     ]
# )
# rot1 = np.linalg.inv(rot1)
# rot1 = R.from_euler("XYZ", [0, 0, 30.9], degrees=True).as_matrix()


data = pd.read_csv("csv/ground_truth_data.csv")
time = data.time
x = data.x
y = data.y

roll = data.roll
pitch = data.pitch
yaw = data.yaw

v_x_list = []
v_y_list = []
v_z_list = []
w_list = []
t_medio_list = []
v_x_rot = []
v_y_rot = []
v_z_rot = []
for i in range(len(data) - 1):
    dt = time[i + 1] - time[i]
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    dtheta = yaw[i+1] - yaw[i]

    t_medio_op = (time[i + 1] + time[i]) / 2

    velocity_x = dx / dt
    velocity_y = dy / dt
    omega = dtheta/dt

    v_x_list.append(velocity_x)
    v_y_list.append(velocity_y)
    w_list.append(omega)
    t_medio_list.append(t_medio_op)

    rot = R.from_euler("XYZ", [roll[i], pitch[i], yaw[i]], degrees=False)
    v = np.array([velocity_x, velocity_y, 0])
    v = rot.inv().as_matrix() @ v


    # v = rot1 @ v
    v_x_rot.append(v[0])
    v_y_rot.append(v[1])
    v_z_rot.append(v[2])


# Threshold
for i in range(len(v_x_rot)):
    if v_x_rot[i] > 1.5:
        v_x_rot[i] = 1.5
    elif v_x_rot[i] < -1.5:
        v_x_rot[i] = -1.5

for i in range(len(v_y_rot)):
    if v_y_rot[i] > 1:
        v_y_rot[i] = 1
    elif v_y_rot[i] < -1:
        v_y_rot[i] = -1

for i in range(len(w_list)):
    if w_list[i] > 2.3:
        w_list[i] = 2.3
    elif w_list[i] < -2.3:
        w_list[i] = -2.3

v_x_rot = pd.Series(v_x_rot)
v_y_rot = pd.Series(v_y_rot)
v_z_rot = pd.Series(v_z_rot)
w = pd.Series(w_list)
t_medio = pd.Series(t_medio_list)


# Create a DataFrame
data = {"t_medio": t_medio, "v_x": v_x_rot, "v_y": v_y_rot, "v_z": v_z_rot, "w": w}

df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file = "csv/vicon_linear_velocities.csv"
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")