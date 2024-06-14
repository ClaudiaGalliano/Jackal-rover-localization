import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


def fx(x, dt):
    """    
    x: State vector [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
    Unità di misura: (x, y m; theta rad; v_x, v_y m/s; w rad/s; a_x, a_y m/s**2)
    dt: Time step
    
    Returns the predicted state.
    """
    x_pos, y_pos, theta, v_x, v_y, omega, a_x, a_y, alpha = x
    
    x_pos_new = x_pos + v_x * dt
    y_pos_new = y_pos + v_y * dt 
    theta_new = theta + omega * dt
    v_x_new = v_x + a_x * dt
    v_y_new = v_y + a_y * dt
    omega_new = omega + alpha * dt
   
    
    return np.array([x_pos_new, y_pos_new, theta_new, v_x_new, v_y_new, omega_new, a_x, a_y, alpha])


# Measurement function
def hx(x):
    """
    Returns the measurement vector.
    """
    return x[:8]  # Measurements are for x, y, theta, v_x, v_y, omega, a_x, a_y


sigma_points = MerweScaledSigmaPoints(n=9, alpha=0.1, beta=2, kappa=-6)  # kappa = 3-n (n = dim_x)

# Initialize the UKF
ukf = UKF(dim_x=9, dim_z=8, fx=fx, hx=hx, dt=0.025, points=sigma_points)

# Initialize the state 
ukf.x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

ukf.P = np.diag([0.01, 0.01, 0.001, 0.01, 0.01, 0.001, 0.01, 0.01, 0.001])  # Initial state covariance 
ukf.Q = np.diag([1, 1, 0.001, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001])  # Process noise covariance
ukf.R = np.diag([0.04, 0.04, 0.00289, 0.01, 0.01, 0.01, 0.96236, 0.9623])  # Measurement noise covariance


# SENSOR FUSION
sensor_data = pd.read_csv("combined_sensor_data.csv")

filter_output_list = []  # state estimates

n = 0
last_t = 0
# Iterate over the sensor data
for index, row in sensor_data.iterrows():
    # Extract measurements 
    z = np.array([row['x'], row['y'], row['yaw'], row['v_x'], row['v_y'], row['w'], row['a_x'], row['a_y']])
    t = row['time']
    dt = t - last_t
    last_t = t  

    # Handle division by zero in dt
    if dt == 0:
        dt = 1e-5  

    ukf.predict()
    ukf.update(z)

    # Check for NaNs in the state vector
    if np.isnan(ukf.x).any():
        print(f"NaN detected in state vector n ° {n} at time {t}: {ukf.x}")
        break

    n = n +1
    filter_output_list.append(ukf.x)


filter_output_array = np.array(filter_output_list)
filter_output = pd.DataFrame(filter_output_array, columns=['pos_x', 'pos_y', 'yaw', 'v_x', 'v_y', 'w', 'a_x', 'a_y', 'alpha'])
filter_output.insert(0, 'time', sensor_data.time)

filter_output.to_csv('UKF/ukf_output.csv')

time_filter = filter_output.time.to_numpy()
x_filter = filter_output.pos_x.to_numpy()
y_filter = filter_output.pos_y.to_numpy()
yaw_filter = filter_output.yaw.to_numpy()
v_x_filter = filter_output.v_x.to_numpy()
v_y_filter = filter_output.v_y.to_numpy()
w_filter = filter_output.w.to_numpy()

# Ground truth data
ground_truth = pd.read_csv("csv/ground_truth_data.csv")  # from the vicon: time, x, y, yaw
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


# Standard deviation (only for x, y, yaw)
start_time = max(time_filter[0], time_ref[0])
end_time = min(time_filter[-1], time_ref[-1])
time_uniform = np.arange(start_time, end_time, step=0.025)

interp_func_filter = interp1d(time_filter, filter_output[['pos_x', 'pos_y', 'yaw']], axis=0, fill_value="extrapolate")
interp_func_ref = interp1d(time_ref, ground_truth[['x', 'y', 'yaw']], axis=0, fill_value="extrapolate")

filter_interp = interp_func_filter(time_uniform)
ref_interp = interp_func_ref(time_uniform)

data_combined = np.hstack((filter_interp, ref_interp))  # li concatena in colonne 
columns = ['x_filter', 'y_filter', 'yaw_filter', 'x_ref', 'y_ref', 'yaw_ref']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_uniform)

combined_df.to_csv("UKF/ukf_data_for_rmse.csv", index=False)
std_data = pd.read_csv("UKF/ukf_data_for_rmse.csv")

RMSE_x = (np.sqrt(np.mean((std_data.x_filter - std_data.x_ref) ** 2))) * 10 ** 2
print(f"The RMSE on x is: {RMSE_x} cm")

RMSE_y = (np.sqrt(np.mean((std_data.y_filter - std_data.y_ref) ** 2))) * 10 ** 2
print(f"The RMSE on y is: {RMSE_y} cm")


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
#plt.plot(t_medio_ref, v_x_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Linear velocity x [m/s]')
plt.title('Linear velocity x - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, v_y_filter, label='Filter')
#plt.plot(t_medio_ref, v_y_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Linear velocity y [m/s]')
plt.title('Linear velocity y - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_filter, w_filter, label='Filter')
#plt.plot(t_medio_ref, w_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [sec]')
plt.ylabel('Angular velocity w [rad/s]')
plt.title('Angular velocity w - Filter vs Ground truth')
plt.grid(True)
plt.legend()
plt.show()