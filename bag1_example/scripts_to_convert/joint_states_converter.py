import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Define the bag file path and the output CSV file path
bag_file = 'bag1_0.db3'
csv_file = 'csv/joint_states_data.csv'

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
position_front_left_wheel_joint = []
position_front_right_wheel_joint = []
position_rear_left_wheel_joint = []
position_rear_right_wheel_joint = []
velocity_front_left_wheel_joint = []
velocity_front_right_wheel_joint = []
velocity_rear_left_wheel_joint = []
velocity_rear_right_wheel_joint = []

# Iterate over messages in the bag
while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    if topic == '/joint_states':  # Replace with your topic name
        # Get message type
        msg_type = get_message(type_map[topic])
        # Deserialize message
        msg = deserialize_message(data, msg_type)
        
        # Extract data
        time_sec.append(msg.header.stamp.sec)
        time_nanosec.append(msg.header.stamp.nanosec)
        position_front_left_wheel_joint.append(msg.position[0])
        position_front_right_wheel_joint.append(msg.position[1])
        position_rear_left_wheel_joint.append(msg.position[2])
        position_rear_right_wheel_joint.append(msg.position[3])
        velocity_front_left_wheel_joint.append(msg.velocity[0])
        velocity_front_right_wheel_joint.append(msg.velocity[1])
        velocity_rear_left_wheel_joint.append(msg.velocity[2])
        velocity_rear_right_wheel_joint.append(msg.velocity[3])

# Create a DataFrame
data = {
    'time_sec': time_sec,
    'time_nanosec': time_nanosec,
    'position_front_left_wheel_joint': position_front_left_wheel_joint,
    'position_front_right_wheel_joint': position_front_right_wheel_joint,
    'position_rear_left_wheel_joint': position_rear_left_wheel_joint,
    'position_rear_right_wheel_joint': position_rear_right_wheel_joint,
    'velocity_front_left_wheel_joint': velocity_front_left_wheel_joint,
    'velocity_front_right_wheel_joint': velocity_front_right_wheel_joint,
    'velocity_rear_left_wheel_joint': velocity_rear_left_wheel_joint,
    'velocity_rear_right_wheel_joint': velocity_rear_right_wheel_joint
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")