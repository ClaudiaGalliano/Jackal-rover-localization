import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

def fx(x, dt):
    """    
    x: State vector [x, v_x, a_x, y, v_y, a_y, theta, w]
    Unità di misura: (x, y m; theta rad; v_x, v_y m/s; w rad/s; a_x, a_y m/s**2)
    dt: Time step
    
    Returns the predicted state.
    """
    x_pos, v_x, a_x, y_pos, v_y, a_y, theta, omega = x
    
    x_pos_new = x_pos + v_x * dt + 0.5 * a_x * dt**2
    v_x_new = v_x + a_x * dt
    y_pos_new = y_pos + v_y * dt + 0.5 * a_y * dt**2
    v_y_new = v_y + a_y * dt
    theta_new = theta + omega * dt
    omega_new = omega

    while theta_new > math.pi:
        theta_new -= 2*math.pi

    while theta_new < -math.pi:
        theta_new += 2*math.pi

    return np.array([x_pos_new, v_x_new, a_x, y_pos_new, v_y_new, a_y, theta_new, omega_new])


def hx_anchor(x, anchor_pos):
 
    dx = x[0] - anchor_pos[0]
    dy = x[3] - anchor_pos[1]
    distance = np.sqrt(dx**2 + dy**2)

    return np.array([distance])

def hx_v(x):

    v = math.sqrt(x[1]**2 + x[4]**2)

    return np.array([v])


def hx_a_x(x):
    return np.array([x[2]])


def hx_a_y(x):
    return np.array([x[5]])


def hx_theta(x):
    return np.array([x[6]])


def hx_w(x):
    return np.array([x[7]])

sigma_points = MerweScaledSigmaPoints(n=8, alpha=.1, beta=2, kappa=-5)  # kappa = 3-n (n = dim_x)

ukf = UKF(dim_x=8, dim_z=1, hx=hx_v, fx=fx, dt=0.02, points=sigma_points)
ukf.x = np.array([0, 0, 0, 0, 0, 0, 0, 0])

ukf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) * 10**(5)  # Initial state covariance 
Q_diag = np.diag([0.1, 100, 0.001, 0.1, 100, 0.001, 0.001, 0.001])  # Process noise covariance

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


ukf.Q = Q_diag  # Process noise covariance
ukf.R = np.array([[1]])  # Measurement noise covariance

sensor_data = pd.read_csv("collected_data.csv")

filter_output_list = []

anchor_positions = [
    [5.50, -1.25],
    [5.50, 1.75],
    [0, -1.25],
    [0, 1.75]]

n = 0
last_t = 0
flag = 0

cycle = 0
for index, row in sensor_data.iterrows():
    z = np.array([row['distance_44AE'], row['distance_06A5'], row['distance_CBB0'], row['distance_4F9B'], row['v_x'], row['a_x'], row['v_y'], row['a_y'], row['w_odom'], row['w_imu']])
    t = row['time']
    dt = t - last_t
    last_t = t

    if dt == 0:
        dt = 1e-5

    if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:
        flag = 1
    
    if flag==1:  # start predicting
        
        # Prediction step
        ukf.predict(dt=dt, fx=fx)
 
        # Update step
        if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:  # se ho uwb data -> update
            for i in range(4):
                ukf.R = np.array([[0.0225]])  # 15 cm
                ukf.update(z[i], hx=hx_anchor, anchor_pos=anchor_positions[i])
                print(f"Update n° {cycle}")
                cycle += 1


        if z[5]!=0 and z[7]!=0 and z[9]!=0:  # imu
            for i, (zi, hx_func, R_val) in enumerate(zip([z[5], z[7], z[9]], 
                                                          [hx_a_x, hx_a_y, hx_w],
                                                          [0.96236, 0.96236, 0.01])):
                ukf.R = np.array([[R_val]])
                ukf.update(zi, hx=hx_func)
                print(f"Update n° {cycle}")
                cycle += 1

    
        if  z[4]!=0 or z[8]!=0 or (z[0]==0 and z[1]==0 and z[2]==0 and z[3]==0 and z[4]==0 and z[5]==0 and z[6]==0 and z[7]==0 and z[8]==0):  # odom
            for i, (zi, hx_func, R_val) in enumerate(zip([z[4], z[8]], 
                                                          [hx_v, hx_w],
                                                          [0.1, 0.01])):
                ukf.R = np.array([[R_val]])
                ukf.update(zi, hx=hx_func)
                print(f"Update n° {cycle}")
                cycle += 1


    # Check for NaNs in the state vector
    if np.isnan(ukf.x).any():
        print(f"NaN detected in state vector n ° {n} at time {t}: {ukf.x}")
        break

    n = n +1
    filter_output_list.append(ukf.x)


filter_output_array = np.array(filter_output_list)
filter_output = pd.DataFrame(filter_output_array, columns=['pos_x', 'v_x', 'a_x', 'pos_y', 'v_y', 'a_y', 'yaw', 'w'])
filter_output.insert(0, 'time', sensor_data.time)

filter_output.to_csv('EKF/ekf_output.csv')

time_filter = filter_output.time.to_numpy()
x_filter = filter_output.pos_x.to_numpy()
y_filter = filter_output.pos_y.to_numpy()
yaw_filter = filter_output.yaw.to_numpy()
v_x_filter = filter_output.v_x.to_numpy()
v_y_filter = filter_output.v_y.to_numpy()
w_filter = filter_output.w.to_numpy()

# Ground truth data
ground_truth = pd.read_csv("csv/ground_truth_data.csv")  
ground_truth_vel = pd.read_csv("csv/vicon_velocities.csv")

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

combined_df.to_csv("EKF/ekf_data_for_rmse.csv", index=False)
std_data = pd.read_csv("EKF/ekf_data_for_rmse.csv")

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