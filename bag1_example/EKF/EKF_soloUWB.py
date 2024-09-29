import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise


def hx_anchor(x, anchor_pos):
 
    dx = x[0] - anchor_pos[0]
    dy = x[2] - anchor_pos[1]
    distance = np.sqrt(dx**2 + dy**2)

    return distance


def Fjacobian(x, dt):

    F = np.array([

        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

    return F


def Hjacobian_anchor(x, anchor_pos):

    dx = x[0] - anchor_pos[0]
    dy = x[2] - anchor_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    H = np.array([[dx/distance, 0, dy/distance, 0]])

    return H


# Initialize the EKF
ekf = EKF(dim_x=4, dim_z=1)

# Initialize the state 
ekf.x = np.array([0, 0, 0, 0])

ekf.F = np.eye(4)

ekf.P = np.diag([0.01, 0.01, 0.01, 0.01])  # Initial state covariance 
Q_diag = np.diag([0.1, 10, 0.1, 10])

for i in range(1):
    for j in range(1):
        if i != j:
            Q_diag[i,j] = 0.001 


for i in range(2, 4):
    for j in range(2, 4):
        if i != j:
            Q_diag[i,j] = 0.001 

ekf.Q = Q_diag  # Process noise covariance
ekf.R = np.array([[1]])  # Measurement noise covariance


# SENSOR FUSION
sensor_data = pd.read_csv("collected_data.csv")

filter_output_list = []  # state estimates

anchor_positions = [
    [5.50, -1.25],
    [5.50, 1.75],
    [0, -1.25],
    [0, 1.75]]

n = 0
last_t = 0
flag = 0

for index, row in sensor_data.iterrows():
    z = np.array([row['distance_44AE'], row['distance_06A5'], row['distance_CBB0'], row['distance_4F9B']])
    t = row['time']
    dt = t - last_t
    last_t = t  

    if dt == 0:
        dt = 1e-5

    if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:
        flag = 1
    
    if flag==1:  # start predicting
        
        # Prediction step
        ekf.F = Fjacobian(ekf.x, dt)
        ekf.predict()

        # Update step
        if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:  # uwb
            for i in range(4):
                ekf.R = np.array([[0.01]])  # 10 cm
                ekf.update(z[i], HJacobian=Hjacobian_anchor, Hx=hx_anchor, hx_args=anchor_positions[i], args=anchor_positions[i])


        if np.isnan(ekf.x).any():
            print(f"NaN detected in state vector n ° {n} at time {t}: {ekf.x}")
            break

        n = n +1
        filter_output_list.append(ekf.x)


filter_output_array = np.array(filter_output_list)
filter_output = pd.DataFrame(filter_output_array, columns=['pos_x', 'v_x', 'pos_y', 'v_y'])
filter_output.insert(0, 'time', sensor_data.time)

filter_output.to_csv('EKF/ekf_soloUWB_output.csv')

time_filter = filter_output.time.to_numpy()
x_filter = filter_output.pos_x.to_numpy()
y_filter = filter_output.pos_y.to_numpy()
v_x_filter = filter_output.v_x.to_numpy()
v_y_filter = filter_output.v_y.to_numpy()


# Ground truth data
ground_truth = pd.read_csv("csv/ground_truth_data.csv")  
ground_truth_vel = pd.read_csv("csv/vicon_velocities.csv")

time_ref = ground_truth.time.to_numpy()
x_ref = ground_truth.x.to_numpy()
y_ref = ground_truth.y.to_numpy()

t_medio_ref = ground_truth_vel.t_medio.to_numpy()
v_x_ref = ground_truth_vel.v_x.to_numpy() 
v_y_ref = ground_truth_vel.v_y.to_numpy() 

window_size = 10
v_x_smoothed_serie = ground_truth_vel.v_x.rolling(window=window_size, min_periods=1).mean()
v_x_smoothed = v_x_smoothed_serie.to_numpy()
v_y_smoothed_serie = ground_truth_vel.v_y.rolling(window=window_size, min_periods=1).mean()
v_y_smoothed = v_y_smoothed_serie.to_numpy()


# RMSE
interp_func_ref = interp1d(time_ref, ground_truth[['x', 'y']], axis=0, fill_value="extrapolate")

ref_interp = interp_func_ref(time_filter)

data_combined = np.hstack((filter_output[['pos_x', 'pos_y']], ref_interp))  # li concatena in colonne 
columns = ['x_filter', 'y_filter', 'x_ref', 'y_ref']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_filter)

combined_df.to_csv("EKF/ekf_soloUWB_data_for_rmse.csv", index=False)
std_data = pd.read_csv("EKF/ekf_soloUWB_data_for_rmse.csv")

RMSE_x = (np.sqrt(np.mean((std_data.x_filter - std_data.x_ref) ** 2))) * 10 ** 2
print(f"The RMSE on x is: {RMSE_x} cm")

RMSE_y = (np.sqrt(np.mean((std_data.y_filter - std_data.y_ref) ** 2))) * 10 ** 2
print(f"The RMSE on y is: {RMSE_y} cm")

RMSE_pos = (np.sqrt(np.mean((std_data.x_filter - std_data.x_ref) ** 2 + (std_data.y_filter - std_data.y_ref) ** 2))) * 10 ** 2
print(f"The RMSE on the position is: {RMSE_pos} cm")


# Absolute Positioning Error
ape = np.sqrt((np.array(std_data.x_filter) - np.array(std_data.x_ref))**2 + (np.array(std_data.y_filter) - np.array(std_data.y_ref))**2)
mean_ape = np.mean(ape) * 10**2
print(f'APE mean: {mean_ape} cm')

# CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

ekf_sorted, ekf_cdf = compute_cdf(ape)


# Plot
plt.plot(x_filter, y_filter, color='royalblue', label='$EKF_{UWB}$')
plt.plot(x_ref, y_ref, color='red', label='Ground truth')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/XY_plot.png")
plt.show()

plt.plot(time_filter, ape, color='royalblue', label='$EKF_{UWB}$')
plt.xlabel('Time [s]')
plt.ylabel('Absolute Positioning Error [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/APE.png")
plt.show()

plt.plot(ekf_sorted, ekf_cdf, color='royalblue', label='$EKF_{UWB}$')
plt.xlabel('Absolute Position Error [m]')
plt.ylabel('CDF')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/CDF.png")
plt.show()

plt.plot(time_filter, x_filter, color='royalblue', label='$EKF_{UWB}$')
plt.plot(time_ref, x_ref, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/x.png")
plt.show()

plt.plot(time_filter, y_filter, color='royalblue', label='$EKF_{UWB}$')
plt.plot(time_ref, y_ref, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/y.png")
plt.show()

plt.plot(time_filter, v_x_filter, color='royalblue', label='$EKF_{UWB}$')
plt.plot(t_medio_ref, v_x_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('$v_x$ [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/v_x.png")
plt.show()

plt.plot(time_filter, v_y_filter, color='royalblue', label='$EKF_{UWB}$')
plt.plot(t_medio_ref, v_y_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('$v_y$ [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/EKF_soloUWB/v_y.png")
plt.show()
