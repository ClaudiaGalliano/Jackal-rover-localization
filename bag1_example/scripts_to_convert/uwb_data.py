import numpy as np
import pandas as pd


uwb = pd.read_csv('csv/uwb_ranging_data.csv')

time_list = []
for i in range(len(uwb)):
    operation = uwb.time_sec[i] + uwb.time_nanosec[i] * (10**(-9))
    time_list.append(operation)


# Convert the uwb ranges cm -> m
d1_list = []
d2_list = []
d3_list = []
d4_list = []
for i in range(len(uwb)):
    d1 = uwb.distance_44AE[i] * 10**(-2)
    d2 = uwb.distance_06A5[i] * 10**(-2)
    d3 = uwb.distance_CBB0[i] * 10**(-2)
    d4 = uwb.distance_4F9B[i] * 10**(-2)

    d1_list.append(d1)
    d2_list.append(d2)
    d3_list.append(d3)
    d4_list.append(d4)


time = pd.Series(time_list)
distance1 = pd.Series(d1_list)
distance2 = pd.Series(d2_list)
distance3 = pd.Series(d3_list)
distance4 = pd.Series(d4_list)

data_dictionary = {
    'time': time,
    'distance_44AE': distance1,
    'distance_06A5': distance2,
    'distance_CBB0': distance3,
    'distance_4F9B': distance4
}

data = pd.DataFrame(data_dictionary)
data.to_csv("csv/uwb_data.csv")