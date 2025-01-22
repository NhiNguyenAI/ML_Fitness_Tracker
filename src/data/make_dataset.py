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
data_path = "../../data/raw/MetaMotion/"
first_files = files[0]

#------------------------------------
# structure the name file of the data
# participant -> ['../../data/raw/MetaMotion\\A',
#  label ->'bench',
#  categogy -> 'heavy2' -->  'heavy'
#  'rpe8_MetaWear_2019',
#  '01',
#  '11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv']
#-------------------------------------------------------------------
participant = first_files.split("-")[0].replace(data_path,"") 
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


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
accelerometer_df = pd.DataFrame()
gyroscopes_df = pd.DataFrame()

accelerometer_set = 1
gyroscopes_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path,"") 
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    if "Accelerometer" in f:
        sensor = "Accelerometer"
    if "Gyroscope" in f:
        sensor = "Gyroscope"

    df = pd.read_csv(f)
    df["participant"]= participant
    df["label"]= label
    df["category"]= category
    df["sensor"]= sensor

    if "Accelerometer" in f:
        df["set"] = accelerometer_set
        accelerometer_set += 1
        accelerometer_df = pd.concat([accelerometer_df, df])
    if "Gyroscope" in f:
        df["set"] = gyroscopes_set
        gyroscopes_set += 1
        gyroscopes_df = pd.concat([gyroscopes_df, df])

# --------------------------------------------------------------
# Introduction for the work with datetimes
# --------------------------------------------------------------
accelerometer_df.info

pd.to_datetime(df["epoch (ms)"], unit= "ms") # Type is datatime
pd.to_datetime(df["time (01:00)"]) #dtype: datetime64[ns]

df["time (01:00)"].dt.weekday # problem, because this is datatime object not normal datatime
# solution for setting the datatime in weekday or month
pd.to_datetime(df["time (01:00)"]).dt.weekday

# --------------------------------------------------------------
# Working with datetimes for this data
# --------------------------------------------------------------
accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit= "ms")
gyroscopes_df.index = pd.to_datetime(gyroscopes_df["epoch (ms)"], unit= "ms")

# delete column epoch, time (01:00) and elapsed, because we dont use the column more
del accelerometer_df["epoch (ms)"]
del accelerometer_df["time (01:00)"]
del accelerometer_df["elapsed (s)"]

del gyroscopes_df["epoch (ms)"]
del gyroscopes_df["time (01:00)"]
del gyroscopes_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
def read_data_files (files):

    # Extract features from filename  
    data_path = "../../data/raw/MetaMotion/"

    # Read all files
    accelerometer_df = pd.DataFrame()
    gyroscopes_df = pd.DataFrame()

    accelerometer_set = 1
    gyroscopes_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path,"") 
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        if "Accelerometer" in f:
            sensor = "Accelerometer"
        if "Gyroscope" in f:
            sensor = "Gyroscope"

        df = pd.read_csv(f)
        df["participant"]= participant
        df["label"]= label
        df["category"]= category
        df["sensor"]= sensor

        if "Accelerometer" in f:
            df["set"] = accelerometer_set
            accelerometer_set += 1
            accelerometer_df = pd.concat([accelerometer_df, df])
        if "Gyroscope" in f:
            df["set"] = gyroscopes_set
            gyroscopes_set += 1
            gyroscopes_df = pd.concat([gyroscopes_df, df])

    # Working with datetimes for this data
    accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit= "ms")
    gyroscopes_df.index = pd.to_datetime(gyroscopes_df["epoch (ms)"], unit= "ms")

    # delete column epoch, time (01:00) and elapsed, because we dont use the column more
    del accelerometer_df["epoch (ms)"]
    del accelerometer_df["time (01:00)"]
    del accelerometer_df["elapsed (s)"]

    del gyroscopes_df["epoch (ms)"]
    del gyroscopes_df["time (01:00)"]
    del gyroscopes_df["elapsed (s)"]

    return(accelerometer_df, gyroscopes_df)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------



# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------


