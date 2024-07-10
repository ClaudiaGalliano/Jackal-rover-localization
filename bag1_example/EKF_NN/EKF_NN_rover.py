import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import ExtendedKalmanFilter as EKF
#from filterpy.common import Q_discrete_white_noise

import tensorflow as tf
from model_class import AE_RNG
from tensorflow.python.keras.engine import data_adapter

import joblib as jb


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


def hx_anchor(x, anchor_pos):
 
    dx = x[0] - anchor_pos[0]
    dy = x[3] - anchor_pos[1]
    distance = np.sqrt(dx**2 + dy**2)

    return distance

def hx_v(x):

    v = math.sqrt(x[1]**2 + x[4]**2)
    return v


def hx_a_x(x):
    return x[2]


def hx_a_y(x):
    return x[5]


def hx_theta(x):
    return x[6]


def hx_w(x):
    return x[7]


def Fjacobian(x, dt):

    F = np.array([

        [1, dt, 0.5*dt**2, 0, 0, 0, 0, 0],
        [0, 1, dt, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0],
        [0, 0, 0, 0, 1, dt, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])

    return F


def Hjacobian_anchor(x, anchor_pos):

    dx = x[0] - anchor_pos[0]
    dy = x[3] - anchor_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    H = np.array([[dx/distance, 0, 0, dy/distance, 0, 0, 0, 0]])

    return H


def Hjacobian_v(x):

    H = np.array([[0, x[1]/math.sqrt(x[1]**2 + x[4]**2), 0, 0, x[4]/math.sqrt(x[1]**2 + x[4]**2), 0, 0, 0]])
    return H

def Hjacobian_a_x(x):

    H = np.array([[0, 0, 1, 0, 0, 0, 0, 0]])
    return H


def Hjacobian_a_y(x):

    H = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
    return H


def Hjacobian_theta(x):

    H = np.array([[0, 0, 0, 0, 0, 0, 1, 0]])
    return H


def Hjacobian_w(x):

    H = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
    return H


def eq(x1, y1, x2, y2):

    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1
    return m, q


def preproc(data, scaler):
    if data.shape == (4,):
        scaledata = scaler.transform(data.reshape(1,-1)*100)
    else:
        scaledata = scaler.transform(data*100)
    return scaledata

scaler = jb.load('scaler.pkl')


# Initialize the EKF
ekf = EKF(dim_x=8, dim_z=1)

# Initialize the state 
ekf.x = np.array([0, 0, 0, 0, 0, 0, 0, 0])

ekf.F = np.eye(8)

ekf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Initial state covariance 
Q_diag = np.diag([0.09, 10, 0.001, 0.09, 10, 0.001, 0.001, 0.001])  # Process noise covariance

for i in range(2):
    for j in range(2):
        if i != j:
            Q_diag[i,j] = 0.001 


for i in range(3, 6):
    for j in range(3, 6):
        if i != j:
            Q_diag[i,j] = 0.001 


for i in range(6, 8):
    for j in range(6, 8):
        if i != j:
            Q_diag[i,j] = 0.0001 

ekf.Q = Q_diag  # Process noise covariance
ekf.R = np.array([[1]])  # Measurement noise covariance


# SENSOR FUSION
sensor_data = pd.read_csv("collected_data.csv")

filter_output_list = []  # state estimates

anchor_positions = [
    [0, 1.75],
    [5.50, 1.75],
    [0, -1.25],
    [5.50, -1.25]]


ae_rng = AE_RNG(model_path='AE_RNG_EKF_Model.h5', train=False)

n = 0
last_t = 0
flag = 0
error_list = []

for index, row in sensor_data.iterrows():
    z = np.array([row['distance_4F9B'], row['distance_06A5'], row['distance_CBB0'], row['distance_44AE'], row['v_x'], row['a_x'], row['v_y'], row['a_y'], row['w_odom'], row['w_imu']])
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

        while ekf.x[6] > math.pi:
            ekf.x[6] -= 2*math.pi

        while ekf.x[6] < -math.pi:
            ekf.x[6] += 2*math.pi

        # Update step
        if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:  # uwb
            NN_data = np.array([[z[0], z[1], z[2], z[3]]])
            NN_data_norm = preproc(NN_data, scaler)
            prediction = ae_rng.predict(NN_data_norm)
            error, absolute_error = ae_rng.score(NN_data_norm)  

            var_list = []
            for i in range(4):
                    if error[0, i]<=0.025:
                        var = 10**(-6)
                        ekf.R = np.array([[var]])
                    elif error[0, i]>=0.07:
                        var = 0.1
                        ekf.R = np.array([[var]])
                    else:
                        m, q = eq(0.025, 10**(-6), 0.07, 0.1) 
                        var = m*error[0, i] + q
                        ekf.R = np.array([[var]])

                    var_list.append(var)
                    error_list.append(error[0, i])
  
                    ekf.update(z[i], HJacobian=Hjacobian_anchor, Hx=hx_anchor, hx_args=anchor_positions[i], args=anchor_positions[i])
            print(var_list)
            #print(error_list)
            

        if z[5]!=0 and z[7]!=0 and z[9]!=0:  # imu
            for i, (zi, Hjacobian, hx_func, R_val) in enumerate(zip([z[5], z[7], z[9]], 
                                                          [Hjacobian_a_x, Hjacobian_a_y, Hjacobian_w],
                                                          [hx_a_x, hx_a_y, hx_w],
                                                          [0.96236, 0.96236, 0.01])):
                ekf.R = np.array([[R_val]])
                ekf.update(zi, HJacobian=Hjacobian, Hx=hx_func)

    
        if  z[4]!=0 or z[8]!=0 or (z[0]==0 and z[1]==0 and z[2]==0 and z[3]==0 and z[4]==0 and z[5]==0 and z[6]==0 and z[7]==0 and z[8]==0):  # odom
            for i, (zi, Hjacobian, hx_func, R_val) in enumerate(zip([z[4], z[8]], 
                                                          [Hjacobian_v, Hjacobian_w],
                                                          [hx_v, hx_w],
                                                          [0.1, 0.01])):
                ekf.R = np.array([[R_val]])
                ekf.update(zi, HJacobian=Hjacobian, Hx=hx_func)
  

        if np.isnan(ekf.x).any():
            print(f"NaN detected in state vector n ° {n} at time {t}: {ekf.x}")
            break

        n = n +1
        filter_output_list.append(ekf.x)


filter_output_array = np.array(filter_output_list)
filter_output = pd.DataFrame(filter_output_array, columns=['pos_x', 'v_x', 'a_x', 'pos_y', 'v_y', 'a_y', 'yaw', 'w'])
filter_output.insert(0, 'time', sensor_data.time)

filter_output.to_csv('ekf_output.csv')

time_filter = filter_output.time.to_numpy()
x_filter = filter_output.pos_x.to_numpy()
y_filter = filter_output.pos_y.to_numpy()
yaw_filter = filter_output.yaw.to_numpy()
v_x_filter = filter_output.v_x.to_numpy()
v_y_filter = filter_output.v_y.to_numpy()
w_filter = filter_output.w.to_numpy()

# Ground truth data
ground_truth = pd.read_csv("ground_truth_data.csv")  
ground_truth_vel = pd.read_csv("vicon_velocities.csv")

time_ref = ground_truth.time.to_numpy()
x_ref = ground_truth.x.to_numpy()
y_ref = ground_truth.y.to_numpy()
yaw_ref = ground_truth.yaw.to_numpy()

t_medio_ref = ground_truth_vel.t_medio.to_numpy()
v_x_ref = ground_truth_vel.v_x.to_numpy() 
v_y_ref = ground_truth_vel.v_y.to_numpy() 
w_ref = ground_truth_vel.w.to_numpy()

window_size = 10
v_x_smoothed_serie = ground_truth_vel.v_x.rolling(window=window_size, min_periods=1).mean()
v_x_smoothed = v_x_smoothed_serie.to_numpy()
v_y_smoothed_serie = ground_truth_vel.v_y.rolling(window=window_size, min_periods=1).mean()
v_y_smoothed = v_y_smoothed_serie.to_numpy()
w_smoothed_serie = ground_truth_vel.w.rolling(window=window_size, min_periods=1).mean()
w_smoothed = w_smoothed_serie.to_numpy()


# RMSE
start_time = max(time_filter[0], time_ref[0], t_medio_ref[0])
end_time = min(time_filter[-1], time_ref[-1], t_medio_ref[-1])
time_uniform = np.arange(start_time, end_time, step=0.02)

interp_func_filter = interp1d(time_filter, filter_output[['pos_x', 'v_x', 'pos_y', 'v_y', 'yaw', 'w']], axis=0, fill_value="extrapolate")
interp_func_ref = interp1d(time_ref, ground_truth[['x', 'y', 'yaw']], axis=0, fill_value="extrapolate")
interp_func_v_ref = interp1d(t_medio_ref, ground_truth_vel[['v_x', 'v_y', 'w']], axis=0, fill_value="extrapolate")

filter_interp = interp_func_filter(time_uniform)
ref_interp = interp_func_ref(time_uniform)
v_ref_interp = interp_func_v_ref(time_uniform)

data_combined = np.hstack((filter_interp, ref_interp, v_ref_interp))  # li concatena in colonne 
columns = ['x_filter', 'v_x_filter', 'y_filter', 'v_y_filter', 'yaw_filter', 'w_filter', 'x_ref', 'y_ref', 'yaw_ref', 'v_x_ref', 'v_y_ref', 'w_ref']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_uniform)

combined_df.to_csv("ekf_data_for_rmse.csv", index=False)
std_data = pd.read_csv("ekf_data_for_rmse.csv")

RMSE_x = (np.sqrt(np.mean((std_data.x_filter - std_data.x_ref) ** 2))) * 10 ** 2
print(f"The RMSE on x is: {RMSE_x} cm")

RMSE_y = (np.sqrt(np.mean((std_data.y_filter - std_data.y_ref) ** 2))) * 10 ** 2
print(f"The RMSE on y is: {RMSE_y} cm")

RMSE_theta = (np.sqrt(np.mean((std_data.yaw_filter- std_data.yaw_ref) ** 2))) 
print(f"The RMSE on theta is: {RMSE_theta} rad")

RMSE_vx = (np.sqrt(np.mean((std_data.v_x_filter - std_data.v_x_ref) ** 2))) 
print(f"The RMSE on v_x is: {RMSE_vx} m/s")

RMSE_vy = (np.sqrt(np.mean((std_data.v_y_filter - std_data.v_y_ref) ** 2))) 
print(f"The RMSE on v_y is: {RMSE_vy} m/s")

RMSE_w = (np.sqrt(np.mean((std_data.w_filter - std_data.w_ref) ** 2)))
print(f"The RMSE on w is: {RMSE_w} rad/s")


# Plot
plt.plot(time_filter, x_filter, label='Filter')
plt.plot(time_ref, x_ref, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Posiition x [m]')
plt.title('Position x - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, y_filter, label='Filter')
plt.plot(time_ref, y_ref, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Posiition y [m]')
plt.title('Position y - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, yaw_filter, label='Filter')
plt.plot(time_ref, yaw_ref, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Theta [rad]')
plt.title('Theta - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, v_x_filter, label='Filter')
plt.plot(t_medio_ref, v_x_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Linear velocity x [m/s]')
plt.title('Linear velocity x - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, v_y_filter, label='Filter')
plt.plot(t_medio_ref, v_y_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Linear velocity y [m/s]')
plt.title('Linear velocity y - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, w_filter, label='Filter')
plt.plot(t_medio_ref, w_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Angular velocity w [rad/s]')
plt.title('Angular velocity w - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()


indice = range(len(error_list))
plt.scatter(indice, error_list, color='r', label='Errori')
plt.xlabel('Indice')
plt.ylabel('Errore')
plt.title('Distribuzione degli errori')
plt.legend()
plt.grid(True)
plt.show()







