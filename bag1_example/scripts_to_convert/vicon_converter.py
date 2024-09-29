import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Define the bag file path and the output CSV file path
bag_file = 'bag1_0.db3'
csv_file = 'csv/vicon_data.csv'

# Initialize storage and converter options
storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

# Create a reader
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

# Get topic information
topic_types = reader.get_all_topics_and_types()
type_map = {topic.name: topic.type for topic in topic_types}

# Create lists to store the data
times = []
positions_x = []
positions_y = []
positions_z = []
orientation_x = []
orientation_y = []
orientation_z = []
orientation_w = []


# Iterate over messages in the bag
while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    if topic == '/vicon/Jackal/Jackal':  # Replace with your topic name
        # Get message type
        msg_type = get_message(type_map[topic])
        # Deserialize message
        msg = deserialize_message(data, msg_type)
        
        # Extract data
        times.append(timestamp)
        positions_x.append(msg.x_trans)
        positions_y.append(msg.y_trans)
        positions_z.append(msg.z_trans)
        orientation_x.append(msg.x_rot)
        orientation_y.append(msg.y_rot)
        orientation_z.append(msg.z_rot)
        orientation_w.append(msg.w)

# Create a DataFrame
data = {
    'time': times,
    'position_x': positions_x,
    'position_y': positions_y,
    'position_z': positions_z,
    'orientation_x': orientation_x,
    'orientation_y': orientation_y,
    'orientation_z': orientation_z,
    'orientation_w': orientation_w,
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")