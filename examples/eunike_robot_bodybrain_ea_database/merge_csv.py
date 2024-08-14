import pandas as pd
import os

# Directory containing the CSV files
csv_directory = '/revolve2_thesis2/CSV_Files/CPPN_Tilted/Experiment4'

# List to hold DataFrames
data_frames = []

# Iterate over all CSV files in the directory
for file_name in os.listdir(csv_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_directory, file_name)
        df = pd.read_csv(file_path)
        data_frames.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(data_frames, ignore_index=True)
#merged_df = merged_df.sort_values(by='experiment_id')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('CPPN_t_experiment4.csv', index=False)

print("CSV files merged successfully into 'merged_output.csv'")