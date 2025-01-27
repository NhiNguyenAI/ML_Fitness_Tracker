import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../../data/interim/outliers_removed_schauvenets.pkl")

predictor_columns = list(df.columns[:6])

# --------------------------------------------------------------
# 2 Adjust plot settings
# --------------------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# Info, how many data are missing
df.info()

# In the Diagramm the missing value
subset = df[df["set"]== 35 ]
df[df["set"]== 35 ]["gyr_y"].plot()

# Solution: Interpolate()
for col in predictor_columns:
    df[col] = df[col].interpolate()
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"]== 25 ]["gyr_y"].plot()
df[df["set"]== 50 ]["gyr_y"].plot()

# try to get duaration of one set : -1 is the last data of this set
duration = df[df["set"]== 1].index[-1] -df[df["set"] == 1].index[0]
duration.seconds

# try to get duaration of all set
df["set"].unique()
for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    stop = df[df["set"]== s].index[-1]

    duration = stop -start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# set the dataframe for duration following the catagory
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0]/5
duration_df.iloc[1]/10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# First muss define the sampling_frequency and cutoff_frequncy

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
