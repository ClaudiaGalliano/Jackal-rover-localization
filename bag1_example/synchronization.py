import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


uwb = pd.read_csv("csv/uwb_position.csv")
uwb_range = pd.read_csv("csv/uwb_ranging_data.csv")
imu_orientation = pd.read_csv("csv/imu_orientation.csv")
imu = pd.read_csv("csv/imu_data.csv")
odom = pd.read_csv("csv/odom_data.csv")

# Convert the uwb ranges cm -> m
d1_list = []
d2_list = []
d3_list = []
d4_list = []
for i in range(len(uwb_range)):
    d1 = uwb_range.distance_44AE[i] * 10**(-2)
    d2 = uwb_range.distance_06A5[i] * 10**(-2)
    d3 = uwb_range.distance_CBB0[i] * 10**(-2)
    d4 = uwb_range.distance_4F9B[i] * 10**(-2)

    d1_list.append(d1)
    d2_list.append(d2)
    d3_list.append(d3)
    d4_list.append(d4)

distance1 = pd.Series(d1_list)
distance2 = pd.Series(d2_list)
distance3 = pd.Series(d3_list)
distance4 = pd.Series(d4_list)

data_dictionary = {
    'distance_44AE': distance1,
    'distance_06A5': distance2,
    'distance_CBB0': distance3,
    'distance_4F9B': distance4
}

data = pd.DataFrame(data_dictionary)
data.to_csv("csv/uwb_range_in_m.csv")

uwb_range_m = pd.read_csv("csv/uwb_range_in_m.csv")


time_uwb = uwb['time'].to_numpy()
time_imu_orientation = imu_orientation['time'].to_numpy()
time_imu = imu['time_sec'].to_numpy()
time_odom = odom['time'].to_numpy()

# Crea un intervallo di tempo uniforme basato sui timestamp di entrambi i sensori
start_time = max(time_imu_orientation[0], time_imu[0], time_odom[0], time_uwb[0])
end_time = min(time_imu_orientation[-1], time_imu[-1], time_odom[-1], time_uwb[-1])
time_uniform = np.arange(start_time, end_time, step=0.02)  # Puoi scegliere il passo che preferisci

# Funzioni di interpolazione per ciascun sensore
interp_func_uwb = interp1d(time_uwb, uwb[['position_x', 'position_y']], axis=0, fill_value="extrapolate")
interp_func_uwb_range = interp1d(time_uwb, uwb_range_m[['distance_44AE', 'distance_06A5', 'distance_CBB0', 'distance_4F9B']], axis=0, fill_value="extrapolate")
interp_func_imu_or = interp1d(time_imu_orientation, imu_orientation[['yaw']], axis=0, fill_value="extrapolate")
interp_func_imu = interp1d(time_imu, imu[['linear_acceleration_x', 'linear_acceleration_y']], axis=0, fill_value="extrapolate")
interp_func_odom = interp1d(time_odom, odom[['linear_velocity_x', 'linear_velocity_y', 'angular_velocity_z']], axis=0, fill_value="extrapolate")

# Interpola i dati sui timestamp uniformi
uwb_interp = interp_func_uwb(time_uniform)
uwb_range_interp = interp_func_uwb_range(time_uniform)
imu_or_interp = interp_func_imu_or(time_uniform)
imu_interp = interp_func_imu(time_uniform)
odom_interp = interp_func_odom(time_uniform)

data_combined = np.hstack((uwb_interp, uwb_range_interp, imu_or_interp, odom_interp, imu_interp))  # li concatena in colonne 
columns = ['x', 'y', 'distance_44AE', 'distance_06A5', 'distance_CBB0', 'distance_4F9B', 'yaw', 'v_x', 'v_y', 'w', 'a_x', 'a_y']
combined_df = pd.DataFrame(data_combined, columns=columns)
combined_df.insert(0, 'time', time_uniform)

combined_df.to_csv('combined_sensor_data.csv', index=False)