"""
************************************************************************
 *
 * make_dataset.py
 *
 * Initial Creation:
 *    Author      Nhi Nguyen
 *    Created on  2025-20-01
 *
 ************************************************************************/
"""
import pandas as pd
from glob import glob
#from common import defines



def read_data_files(path_files):
    """
    read_data_files Function
    -------------------------
    Description:
        Reads and processes accelerometer and gyroscope data files.

    Parameters:
        - files_path (string): file paths to the data files.

    Returns:
        - accelerometer_df (DataFrame): Processed accelerometer data.
        - gyroscopes_df (DataFrame): Processed gyroscope data.

    Columns in the DataFrames:
        - x_acc, y_acc, z_acc: Accelerometer data columns.
        - x_gyr, y_gyr, z_gyr: Gyroscope data columns.
        - participant: The participant identifier.
        - label: The activity label: Bench, Squat..
        - category: The activity category: Medium or Heavy
        - et: Unique identifier for each data csv files.
    """

    files = glob(path_files)

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
        participant = f.split("-")[0].replace(data_path, "").replace("\\", "/")
        participant = participant.split("/")[-1]  # Keep only the last part (e.g., "A")
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


def read_and_export_to_pickel_file():
    """
    read_and_export_to_pickel_file Function
    ---------------------------------------
    Description:
        Reads raw data files, processes them, merges accelerometer and gyroscope data,
        resamples the dataset for frequency alignment, and exports the processed data to a pickle file.

    Parameters:
        - files_path (str): Directory containing the data files.

    Returns:
        - pickel_file (str): Path to the exported pickle file.
    """

    # --------------------------------------------------------------
    # Process and Merge Datasets
    # --------------------------------------------------------------

    # Read accelerometer and gyroscope data
    acc_df, gyr_df = read_data_files("../../data/raw/MetaMotion/*.csv")

    # Merge data: iloc[:,:3] because acc_df and gyr_df have the same 3 parameter: Participant, label, category -> choose the first dataframe without 3 parameters
    # axis = 1 is merged the tables following the colunms, axis= 0, following the row
    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

    # Remove rows with missing values
    #data_merged_cleared = data_merged.dropna()

    # Rename the columns for better clarity
    data_merged.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set"
    ]

    # --------------------------------------------------------------
    # Resample Data for Frequency Conversion
    # Accelerometer: 12.500 Hz, 12.500 Hz means the accelerometer records 12.5 measurements per second (or one measurement every 80 milliseconds).
    # Gyroscope: 25.000 Hz, 25.000 Hz means the gyroscope records 25 measurements per second (or one measurement every 40 milliseconds).
    # The goal is to align the accelerometer (12.5 Hz) and gyroscope (25 Hz) data into a common time interval for analysis.
    # Choose a Common Time Interval: The smallest interval that can encompass both frequencies is  milliseconds "200ms". Resample both datasets to 200ms for consistency.
    # --------------------------------------------------------------


    # data_merged.resample("200ms").mean() lose the seterify parameter: Participant, label, category
    # Solution: sammling -> use the resample.apply(sammling)

    sampling =  {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last"
    }


    # --------------------------------------------------------------
    # Filter and Resample Data for Frequency Conversion
    # Description: Resample only the active periods of data to avoid inefficiency.
    # Important Note:
    #   - data_merged.resample("200ms").apply(sampling): Resampling in Pandas will process every 200ms interval across the entire dataset.
    #   - If your data is not continuous (e.g., exercises only 2 hours/day, 3 days/week), this may create unnecessary intervals and consume significant resources.
    #   - This take 3332143 rows instate of 9127 rows , it take long time and have alot data with None 
    # Solution:
    #   - Resample only the periods Day 'D' where data exists, using dropna() to filter valid rows first.
    # --------------------------------------------------------------

    day = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
    data_resampled = pd.concat([df.resample("200ms").apply(sampling).dropna() for df in day])
    # Convert the "set" column to integers
    data_resampled["set"] = data_resampled["set"].astype(int)

    # --------------------------------------------------------------
    # Export the Processed Dataset
    # --------------------------------------------------------------
    # Save the merged and cleaned dataset to a CSV file
    pickel_file = data_resampled.to_pickle("../../data/interim/data_process.pkl")

    return(pickel_file)

pickle_file = read_and_export_to_pickel_file()