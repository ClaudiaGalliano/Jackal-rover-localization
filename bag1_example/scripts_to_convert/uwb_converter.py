import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Define the bag file path and the output CSV file path
bag_file = 'bag2_0.db3'
csv_file = 'uwb_ranging_data.csv'

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
distance_44AE = []
distance_06A5 = []
distance_CBB0 = []
distance_4F9B = []

# Iterate over messages in the bag
while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    if topic == '/uwb_ranging':  # Replace with your topic name
        # Get message type
        msg_type = get_message(type_map[topic])
        # Deserialize message
        msg = deserialize_message(data, msg_type)
        
        # Extract data
        time_sec.append(msg.timestamp[0])
        time_nanosec.append(msg.timestamp[1])
        distance_44AE.append(msg.range_mes[0])
        distance_06A5.append(msg.range_mes[1])
        distance_CBB0.append(msg.range_mes[2])
        distance_4F9B.append(msg.range_mes[3])

# Create a DataFrame
data = {
    'time_sec': time_sec,
    'time_nanosec': time_nanosec,
    'distance_44AE': distance_44AE,
    'distance_06A5': distance_06A5,
    'distance_CBB0': distance_CBB0,
    'distance_4F9B': distance_4F9B
}

df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")