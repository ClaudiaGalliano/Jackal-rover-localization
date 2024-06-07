import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


df = pd.read_csv('csv/uwb_ranging_data.csv')

estimated_position = []
position_x_list = []
position_y_list = []

time_list = []

for i in range(len(df)):

    distances_to_anchors = np.array([df.distance_44AE[i], df.distance_06A5[i], df.distance_CBB0[i], df.distance_4F9B[i]])
    anchors = np.array([[550, -125], [550, 175], [0, -125], [0, 175]])

    # Objective function to minimize
    def objective_function(position, anchors, distances_to_anchors):
        x, y = position
        error = 0
        for (x_i, y_i), d_i in zip(anchors, distances_to_anchors):  # zip efficiently combines and iterates over multiple lists
            calculated_distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
            error += (calculated_distance - d_i)**2
        return error

    # Initial guess for the position (x, y)
    initial_guess = np.array([0.5, 0.5])

    # Perform the optimization
    result = minimize(objective_function, initial_guess, args=(anchors, distances_to_anchors))

    # Extract the estimated position
    estimated_position = result.x  # the minimize function returns an object of type 'OptimizeResult' - to access the solution of the optimization use .x
    position_x_list.append((estimated_position[0] + 22)/100)
    position_y_list.append(estimated_position[1]/100)

    operation = df.time_sec[i] + df.time_nanosec[i] * (10**(-9))
    time_list.append(operation)


#print("Estimated Position_x :", position_x)
#print("Estimated Position_y :", position_y)
plt.plot(position_x_list, position_y_list)
plt.show()

time = pd.Series(time_list)
position_x = pd.Series(position_x_list)
position_y = pd.Series(position_y_list)

data_dictionary = {
    'time': time,
    'position_x': position_x,
    'position_y': position_y
}

data = pd.DataFrame(data_dictionary)
data.to_csv("uwb_position.csv")

