import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Define the path of the files
# The files contain raw data from MetaMotion sensors
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")

# --------------------------------------------------------------
# read_data_files Function
# Description: Reads and processes accelerometer and gyroscope data files.
# Parameters: 
#   - files: List of file paths to the data files.
# Returns:
#   - accelerometer_df: DataFrame containing processed accelerometer data.
#   - gyroscopes_df: DataFrame containing processed gyroscope data.
# --------------------------------------------------------------
def read_data_files(files):
    # Initialize DataFrames
    accelerometer_df = pd.DataFrame()
    gyroscopes_df = pd.DataFrame()

    # Counters for the set columns
    accelerometer_set = 1
    gyroscopes_set = 1

    # Path prefix for parsing
    data_path = "../../data/raw/MetaMotion/"

    # Loop through each file and process
    for f in files:
        # Extract metadata from file name
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        # Read the CSV file
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Separate into accelerometer and gyroscope datasets
        if "Accelerometer" in f:
            df["set"] = accelerometer_set
            accelerometer_set += 1
            accelerometer_df = pd.concat([accelerometer_df, df], ignore_index=True)
        elif "Gyroscope" in f:
            df["set"] = gyroscopes_set
            gyroscopes_set += 1
            gyroscopes_df = pd.concat([gyroscopes_df, df], ignore_index=True)

    # Convert epoch (ms) to datetime and set as index
    accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit="ms")
    gyroscopes_df.index = pd.to_datetime(gyroscopes_df["epoch (ms)"], unit="ms")

    # Drop unnecessary columns
    accelerometer_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
    gyroscopes_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

    return accelerometer_df, gyroscopes_df

# --------------------------------------------------------------
# Process and Merge Datasets
# --------------------------------------------------------------
# Read accelerometer and gyroscope data
acc_df, gyr_df = read_data_files(files)

# Merge data: First 3 columns (metadata) with the rest of the accelerometer data
data_merged = pd.concat([acc_df.iloc[:, :3], acc_df], axis=1)

# Remove rows with missing values
data_merged_cleared = data_merged.dropna()

# Rename the columns for better clarity
data_merged_cleared.columns = [
    "acc-x", "acc-y", "acc-z", "gyr-x", "gyr-y", "gyr-z", 
    "participant", "label", "category", "set"
]

# --------------------------------------------------------------
# Resample Data for Frequency Conversion
# Accelerometer: 12.500 Hz
# Gyroscope: 25.000 Hz
# --------------------------------------------------------------
# Placeholder: Implement resampling logic here as needed
# Example: Resample accelerometer and gyroscope data
# accelerometer_resampled = acc_df.resample('80ms').mean()
# gyroscope_resampled = gyr_df.resample('40ms').mean()

# --------------------------------------------------------------
# Export the Processed Dataset
# --------------------------------------------------------------
# Save the merged and cleaned dataset to a CSV file

