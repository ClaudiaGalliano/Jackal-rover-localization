import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# in m
T = np.array([
    [ 0.81907688, 0.51108873, 0.26057892, -0.03416372315],
    [-0.49096203, 0.85945434, -0.14245887, 0.00026770578],
    [-0.29676481, -0.01124959, 0.95488434, 0.02656486081],
    [0, 0, 0, 1]
])


data = pd.read_csv("csv/ground_truth_data.csv")
time = data.time
x = data.x
y = data.y
z = data.z

v_x_list = []
v_y_list = []
v_z_list = []
t_medio_list = []
for i in range(len(data) - 1):
    dt = time[i+1] - time[i]
    dx = abs(x[i+1] - x[i])
    dy = abs(y[i+1] - y[i])
    dz = abs(z[i+1] - z[i])

    t_medio_op = (time[i+1] + time[i]) / 2

    velocity_x = dx/dt
    velocity_y = dy/dt
    velocity_z = dz/dt

    v_x_list.append(velocity_x)
    v_y_list.append(velocity_y)
    v_z_list.append(velocity_z)
    t_medio_list.append(t_medio_op)


vel_list = []
vx = []
vy = []
vz = []
for i in range(len(v_x_list)):
    vel_list.append(v_x_list[i])
    vel_list.append(v_y_list[i])
    vel_list.append(v_z_list[i])
    vel_list.append(1)

    vel = np.array(vel_list)  
    vel_list = []  

    new_vel = np.dot(T, vel)
    vx.append(new_vel[0])
    vy.append(new_vel[1])
    vz.append(new_vel[2])


# Threshold 
for i in range(len(vx)):
    if vx[i] > 1.5:
        vx[i] = 1.5
    elif vx[i] < -1.5:
        vx[i] = -1.5

for i in range(len(vy)):
    if vy[i] > 1:
        vy[i] = 1
    elif vy[i] < -1:
        vy[i] = -1

v_x = pd.Series(vx) 
v_y = pd.Series(vy)
v_z = pd.Series(vz) 
t_medio = pd.Series(t_medio_list)


# Create a DataFrame
data = {
    't_medio': t_medio,
    'v_x': v_x,
    'v_y': v_y,
    'v_z': v_z
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file = 'vicon_linear_velocities.csv'
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")






