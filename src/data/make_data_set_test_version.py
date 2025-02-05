import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2_MetaWear_2019-01-14T14.27.00.784_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "..\\..\\data\\raw\\MetaMotion\\"
first_files = files[0]


participant = first_files.split("-")[0].replace(data_path, "").replace("\\", "/")  # Normalize the path
participant = participant.split("/")[-1]  # Keep only the last part (e.g., "A")
label = first_files.split("-")[1]
category = first_files.split("-")[2].rstrip("123")
if "Accelerometer" in first_files:
    sensor = "Accelerometer"
if "Gyroscope" in first_files:
    sensor = "Gyroscope"
df = pd.read_csv(first_files)
df["participant"] = participant
df["label"] = label
df["category"] = category
df["sensor"] = sensor

df