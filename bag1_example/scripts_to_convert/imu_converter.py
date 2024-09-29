import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Define the bag file path and the output CSV file path
bag_file = 'bag1_0.db3'
csv_file = 'csv/imu_data.csv'

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
time_sec = []
time_nanosec = []
orientation_x = []
orientation_y = []
orientation_z = []
orientation_w = []
angular_velocity_x = []
angular_velocity_y = []
angular_velocity_z = []
linear_acceleration_x = []
linear_acceleration_y = []
linear_acceleration_z = []


# Iterate over messages in the bag
while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    if topic == '/imu/data':  # Replace with your topic name
        # Get message type
        msg_type = get_message(type_map[topic])
        # Deserialize message
        msg = deserialize_message(data, msg_type)
        
        # Extract data
        time_sec.append(msg.header.stamp.sec)
        time_nanosec.append(msg.header.stamp.nanosec)
        orientation_x.append(msg.orientation.x)
        orientation_y.append(msg.orientation.y)
        orientation_z.append(msg.orientation.z)
        orientation_w.append(msg.orientation.w)
        angular_velocity_x.append(msg.angular_velocity.x)
        angular_velocity_y.append(msg.angular_velocity.y)
        angular_velocity_z.append(msg.angular_velocity.z)
        linear_acceleration_x.append(msg.linear_acceleration.x)
        linear_acceleration_y.append(msg.linear_acceleration.y)
        linear_acceleration_z.append(msg.linear_acceleration.z)


# Create a DataFrame
data = {
    'time_sec': time_sec,
    'time_nanosec': time_nanosec,
    'orientation_x': orientation_x,
    'orientation_y': orientation_y,
    'orientation_z': orientation_z,
    'orientation_w': orientation_w,
    'angular_velocity_x': angular_velocity_x,
    'angular_velocity_y': angular_velocity_y,
    'angular_velocity_z': angular_velocity_z,
    'linear_acceleration_x': linear_acceleration_x,
    'linear_acceleration_y': linear_acceleration_y,
    'linear_acceleration_z': linear_acceleration_z
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")
