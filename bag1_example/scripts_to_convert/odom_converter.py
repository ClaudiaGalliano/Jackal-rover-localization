import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Define the bag file path and the output CSV file path
bag_file = 'bag2_0.db3'
csv_file = 'odom_data.csv'

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
position_x = []
position_y = []
position_z = []
orientation_x = []
orientation_y = []
orientation_z = []
orientation_w = []
linear_velocity_x = []
linear_velocity_y = []
linear_velocity_z = []
angular_velocity_x = []
angular_velocity_y = []
angular_velocity_z = []

time = []

# Iterate over messages in the bag
while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    if topic == '/odom':  # Replace with your topic name
        # Get message type
        msg_type = get_message(type_map[topic])
        # Deserialize message
        msg = deserialize_message(data, msg_type)
        
        # Extract data
        time_sec.append(msg.header.stamp.sec)
        time_nanosec.append(msg.header.stamp.nanosec)
        position_x.append(msg.pose.pose.position.x)
        position_y.append(msg.pose.pose.position.y)
        position_z.append(msg.pose.pose.position.z)
        orientation_x.append(msg.pose.pose.orientation.x)
        orientation_y.append(msg.pose.pose.orientation.y)
        orientation_z.append(msg.pose.pose.orientation.z)
        orientation_w.append(msg.pose.pose.orientation.w)
        linear_velocity_x.append(msg.twist.twist.linear.x)
        linear_velocity_y.append(msg.twist.twist.linear.y)
        linear_velocity_z.append(msg.twist.twist.linear.z)
        angular_velocity_x.append(msg.twist.twist.angular.x)
        angular_velocity_y.append(msg.twist.twist.angular.y)
        angular_velocity_z.append(msg.twist.twist.angular.z)


for i in range(len(time_sec)):
    operation = time_sec[i] + time_nanosec[i] * (10**(-9))
    time.append(operation)


# Create a DataFrame
data = {
    'time': time,
    'position_x': position_x,
    'position_y': position_y,
    'position_z': position_z,
    'orientation_x': orientation_x,
    'orientation_y': orientation_y,
    'orientation_z': orientation_z,
    'orientation_w': orientation_w,
    'linear_velocity_x': linear_velocity_x,
    'linear_velocity_y': linear_velocity_y,
    'linear_velocity_z': linear_velocity_z,
    'angular_velocity_x': angular_velocity_x,
    'angular_velocity_y': angular_velocity_y,
    'angular_velocity_z': angular_velocity_z
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")