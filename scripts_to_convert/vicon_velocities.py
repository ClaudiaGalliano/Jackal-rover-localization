import pandas as pd


vicon = pd.read_csv("csv/ground_truth_data.csv")

time = vicon.time
x = vicon.x
y = vicon.y
yaw = vicon.yaw

v_x_list = []
v_y_list = []
w_list = []
t_medio_list = []
for i in range(len(vicon) - 1):
    dt = time[i+1] - time[i]
    dx = x[i+1] - x[i]
    dy = y[i+1] - y[i]
    dtheta = yaw[i+1] - yaw[i]

    t_medio_op = (time[i+1] + time[i]) / 2

    velocity_x = dx/dt
    velocity_y = dy/dt
    omega = dtheta/dt

    v_x_list.append(velocity_x)
    v_y_list.append(velocity_y)
    w_list.append(omega)
    t_medio_list.append(t_medio_op)


# Threshold 
for i in range(len(v_x_list)):
    if v_x_list[i] > 1.5:
        v_x_list[i] = 1.5
    elif v_x_list[i] < -1.5:
        v_x_list[i] = -1.5

for i in range(len(v_y_list)):
    if v_y_list[i] > 1:
        v_y_list[i] = 1
    elif v_y_list[i] < -1:
        v_y_list[i] = -1

for i in range(len(w_list)):
    if w_list[i] > 2.3:
        w_list[i] = 2.3
    elif w_list[i] < -2.3:
        w_list[i] = -2.3


v_x = pd.Series(v_x_list)
v_y = pd.Series(v_y_list)
w = pd.Series(w_list)
t_medio = pd.Series(t_medio_list)

# Create a DataFrame
data = {
    't_medio': t_medio,
    'v_x': v_x,
    'v_y': v_y,
    'w': w
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file = 'vicon_velocities.csv'
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")