import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from numpy import linalg as la


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


def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


sigma_points = MerweScaledSigmaPoints(n=8, alpha=.1, beta=2, kappa=-5)  # kappa = 3-n (n = dim_x)

ukf = UKF(dim_x=8, dim_z=1, hx=hx_v, fx=fx, dt=0.02, points=sigma_points)
ukf.x = np.array([0, 0, 0, 0, 0, 0, 0, 0])

ukf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Initial state covariance 

try:
    _ = np.linalg.cholesky(ukf.P)
except np.linalg.LinAlgError:
    ukf.P = nearestPD(ukf.P)

ukf.R = np.array([[0.1]])  # Measurement noise covariance

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

for index, row in sensor_data.iterrows():
    z = np.array([row['distance_44AE'], row['distance_06A5'], row['distance_CBB0'], row['distance_4F9B'], row['v_x'], row['a_x'], row['v_y'], row['a_y'], row['w_odom'], row['w_imu']])
    t = row['time']
    dt = t - last_t
    last_t = t

    if dt == 0:
        dt = 1e-5

    Q_diag = np.diag([0.1, 10, 0.001, 0.1, 5, 0.001, 0.001, 0.001]) 

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

    if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:
        flag = 1
    
    if flag==1:  # start predicting
        
        # Prediction step
        ukf.predict(dt=dt)
        
        try:
            _ = np.linalg.cholesky(ukf.P)
        except np.linalg.LinAlgError:
            ukf.P = nearestPD(ukf.P)
 
        # Update step
        if z[0]!=0 and z[1]!=0 and z[2]!=0 and z[3]!=0:  # se ho uwb data -> update
            for i in range(4):
                ukf.R = np.array([[0.01]])  
                ukf.update(z[i], hx=hx_anchor, anchor_pos=anchor_positions[i])
                ukf.predict(dt=dt)

                try:
                    _ = np.linalg.cholesky(ukf.P)
                except np.linalg.LinAlgError:
                    ukf.P = nearestPD(ukf.P)


        if z[5]!=0 and z[7]!=0 and z[9]!=0:  # imu
            for i, (zi, hx_func, R_val) in enumerate(zip([z[5], z[7], z[9]], 
                                                          [hx_a_x, hx_a_y, hx_w],
                                                          [1, 1, 0.00001])):
                ukf.R = np.array([[R_val]])
                ukf.update(zi, hx=hx_func)

                try:
                    _ = np.linalg.cholesky(ukf.P)
                except np.linalg.LinAlgError:
                    ukf.P = nearestPD(ukf.P)

    
        if  z[4]!=0 or z[8]!=0 or (z[0]==0 and z[1]==0 and z[2]==0 and z[3]==0 and z[4]==0 and z[5]==0 and z[6]==0 and z[7]==0 and z[8]==0):  # odom
            for i, (zi, hx_func, R_val) in enumerate(zip([z[4], z[8]], 
                                                          [hx_v, hx_w],
                                                          [0.1, 0.01])):
                ukf.R = np.array([[R_val]])
                ukf.update(zi, hx=hx_func)

                try:
                    _ = np.linalg.cholesky(ukf.P)
                except np.linalg.LinAlgError:
                    ukf.P = nearestPD(ukf.P)


    # Check for NaNs in the state vector
    if np.isnan(ukf.x).any():
        print(f"NaN detected in state vector n ° {n} at time {t}: {ukf.x}")
        break

    n = n +1
    filter_output_list.append(ukf.x)


filter_output_array = np.array(filter_output_list)
filter_output = pd.DataFrame(filter_output_array, columns=['pos_x', 'v_x', 'a_x', 'pos_y', 'v_y', 'a_y', 'yaw', 'w'])
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
interp_func_ref = interp1d(time_ref, ground_truth[['x', 'y']], axis=0, fill_value="extrapolate")

ref_interp = interp_func_ref(time_filter)

data_combined = np.hstack((filter_output[['pos_x', 'pos_y']], ref_interp))  # li concatena in colonne 
columns = ['x_filter', 'y_filter', 'x_ref', 'y_ref']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_filter)

combined_df.to_csv("EKF/ekf_data_for_rmse.csv", index=False)
std_data = pd.read_csv("EKF/ekf_data_for_rmse.csv")

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
plt.plot(x_filter, y_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(x_ref, y_ref, color='red', label='Ground truth')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/XY_plot.png")
plt.show()

plt.plot(time_filter, ape, color='royalblue', label='$UKF_{SF}$')
plt.xlabel('Time [s]')
plt.ylabel('Absolute Positioning Error [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/APE.png")
plt.show()

plt.plot(ekf_sorted, ekf_cdf, color='royalblue', label='$UKF_{SF}$')
plt.xlabel('Absolute Position Error [m]')
plt.ylabel('CDF')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/CDF.png")
plt.show()

plt.plot(time_filter, x_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(time_ref, x_ref, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/x.png")
plt.show()

plt.plot(time_filter, y_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(time_ref, y_ref, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('y [m]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/y.png")
plt.show()

plt.plot(time_filter, yaw_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(time_ref, yaw_ref, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('Theta [rad]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/yaw.png")
plt.show()

plt.plot(time_filter, v_x_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(t_medio_ref, v_x_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('$v_x$ [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/v_x.png")
plt.show()

plt.plot(time_filter, v_y_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(t_medio_ref, v_y_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('$v_y$ [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/v_y.png")
plt.show()

plt.plot(time_filter, w_filter, color='royalblue', label='$UKF_{SF}$')
plt.plot(t_medio_ref, w_smoothed, color='red', label='Ground truth')
plt.xlabel('Time [s]')
plt.ylabel('w [rad/s]')
plt.grid(True)
plt.legend()
plt.savefig(f"images/UKF/w.png")
plt.show()