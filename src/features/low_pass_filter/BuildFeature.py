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
# Butterworth lowpass filter for each Column
# --------------------------------------------------------------

# First muss define the sampling_frequency and cutoff_frequncy

df_lowpass = df.copy()
LowPass = LowPassFilter()

# At the beginn of the step 1: make_dataset , setting each step 200ms -> fs = 1s /200ms

fs = 1000/200
cutoff = 1.2 # try difference the cutoff to see the diffence cutoff= 1.2, custoff = 2

df_lowpass = LowPassFilter().low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)

# Example take look subset 45
subset = df_lowpass[df_lowpass["set"] == 45]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize = (20,10))
ax[0].plot(subset["acc_y"].reset_index(drop = True), label = "raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop = True), label = "butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor =(0.5, 1.15), fancybox= True, shadow =True)
ax[1].legend(loc="upper center", bbox_to_anchor =(0.5, 1.15), fancybox= True, shadow =True)

# --------------------------------------------------------------
# Butterworth lowpass filter for all Columns
# --------------------------------------------------------------
for col in predictor_columns:
    df_lowpass = LowPassFilter().low_pass_filter(df_lowpass, col, fs, cutoff, order = 5)
    # overwrite the new date acc and gyr with lowpass on the old data
    df_lowpass[col] = df_lowpass[ col + "_lowpass"]
    del df_lowpass[ col + "_lowpass"]
  
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize = (10,10))
plt.plot(range(1, len(predictor_columns)+1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Example take look subset 45
subset = df_pca[df_pca["set"] == 45]

subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 18]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

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
