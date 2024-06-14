import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


uwb = pd.read_csv("csv/uwb_position.csv")
imu_orientation = pd.read_csv("csv/imu_orientation.csv")
imu = pd.read_csv("csv/imu_data.csv")
odom = pd.read_csv("csv/odom_data.csv")

time_uwb = uwb['time'].to_numpy()
time_imu_orientation = imu_orientation['time'].to_numpy()
time_imu = imu['time_sec'].to_numpy()
time_odom = odom['time'].to_numpy()

# Crea un intervallo di tempo uniforme basato sui timestamp di entrambi i sensori
start_time = max(time_imu_orientation[0], time_imu[0], time_odom[0], time_uwb[0])
end_time = min(time_imu_orientation[-1], time_imu[-1], time_odom[-1], time_uwb[-1])
time_uniform = np.arange(start_time, end_time, step=0.025)  # Puoi scegliere il passo che preferisci

# Funzioni di interpolazione per ciascun sensore
interp_func_uwb = interp1d(time_uwb, uwb[['position_x', 'position_y']], axis=0, fill_value="extrapolate")
interp_func_imu_or = interp1d(time_imu_orientation, imu_orientation[['yaw']], axis=0, fill_value="extrapolate")
interp_func_imu = interp1d(time_imu, imu[['linear_acceleration_x', 'linear_acceleration_y']], axis=0, fill_value="extrapolate")
interp_func_odom = interp1d(time_odom, odom[['linear_velocity_x', 'linear_velocity_y', 'angular_velocity_z']], axis=0, fill_value="extrapolate")

# Interpola i dati sui timestamp uniformi
uwb_interp = interp_func_uwb(time_uniform)
imu_or_interp = interp_func_imu_or(time_uniform)
imu_interp = interp_func_imu(time_uniform)
odom_interp = interp_func_odom(time_uniform)

data_combined = np.hstack((uwb_interp, imu_or_interp, odom_interp, imu_interp))  # li concatena in colonne 
columns = ['x', 'y', 'yaw', 'v_x', 'v_y', 'w', 'a_x', 'a_y']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_uniform)

combined_df.to_csv('combined_sensor_data.csv', index=False)