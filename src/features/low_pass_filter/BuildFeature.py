import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from AbstractionFrequency import FourierTransformation
from sklearn.cluster import KMeans

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

subset = df_squared[df_squared["set"] == 14]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
window_size = int(1000/200)  # 1000ms/200ms

for cols in predictor_columns:
    #[cols] list
    df_temporal = NumAbs.abstract_numerical(df_temporal, [cols], window_size= window_size, aggregation_function="mean")

df_temporal

# The Problem hier: The Window Size is 5, it may be 3 Values of the bench and 2 value of the squat
# Solution: Frist take a look of set

df_temporal_list =[]

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for cols in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [cols], window_size= window_size, aggregation_function="mean")
        subset = NumAbs.abstract_numerical(subset, [cols], window_size= window_size, aggregation_function="std")
       
    df_temporal_list.append(subset)

# Take careful hier, df_temporal muss copy with reset_index . If dont have reset index -> Alot value have Nan
df_temporal = pd.concat(df_temporal_list)
df_temporal.info()


subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset.info()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
 
df_frequency = df_temporal.copy().reset_index()

fs = int(1000/200)
window_size = df[df["set"] == 14].index[-1] - df[df["set"] == 14].index[0]
ws = int(window_size.seconds) # 14 min for 1 set

AbsFs = FourierTransformation()

df_frequency = AbsFs.abstract_frequency(df_frequency, ["acc_y"], ws, fs)
df_frequency.info()

# Visualize results
subset = df_frequency[df_frequency["set"] == 15]
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_max_freq", "acc_y_temp_std_ws_5","acc_y_freq_weighted","acc_y_pse"]].plot()
subset[[
    "acc_y_freq_weighted",
    "acc_y_max_freq",
    "acc_y_pse",
    "acc_y_freq_0.0_Hz_ws_14",
    "acc_y_freq_0.357_Hz_ws_14",
    "acc_y_freq_0.714_Hz_ws_14",
    "acc_y_freq_1.071_Hz_ws_14",
    "acc_y_freq_2.5_Hz_ws_14", 
    "acc_y_freq_1.429_Hz_ws_14"]].plot()

df_frequency_list =[]
AbsFs = FourierTransformation()
for s in df_frequency["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_frequency[df_frequency["set"] == s].reset_index(drop=True).copy()
    subset = AbsFs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_frequency_list.append(subset)
    
df_frequency = pd.concat(df_frequency_list).set_index("epoch (ms)", drop=True)
df_frequency.info()
      

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# drop all missing values
df_frequency= df_frequency.dropna()

# drop 50 prcent data of the dataset

df_frequency = df_frequency[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
# Initialize lists to store distortion and inertia values

df_cluster = df_frequency.dropna()

cluster_column = ["acc_x", "acc_y", "acc_z"]
k_vaules = range(2,10)
inertias = []

# Fit K-means for different values of k
for k in k_vaules:
    subset = df_cluster[cluster_column]
    kmeanModel = KMeans(n_clusters=k, n_init = 20, random_state=0)
    cluster_label = kmeanModel.fit_predict(subset)
    inertias.append(kmeanModel.inertia_)

plt.figure(figsize = (10,10))
plt.plot(k_vaules, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distance")
save_path = ("../../../reports/figures/features_engineering/K_Means_Inertia_Cluster_png")
plt.savefig(save_path)
plt.show()


subset = df_cluster[cluster_column]
kmeanModel = KMeans(n_clusters=5, n_init = 20, random_state=0)
df_cluster["cluster"] = kmeanModel.fit_predict(subset)

# Plot Cluster
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

# Colors based on cluster labels
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], 
               label=f"Cluster {c}", s=40, alpha=0.7)

# Set axis labels
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("3D K-Means Clustering")
ax.legend()
save_path = ("../../../reports/figures/features_engineering/3D_K_Means_Inertia_Cluster_png")
plt.savefig(save_path)
plt.show()


# Compare label diagram and cluster diagram -> find the connection of each other
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], 
               label=f"Label {l}", s=40, alpha=0.7)

# Set axis labels
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("3D Label")
ax.legend()
save_path = ("../../../reports/figures/features_engineering/3D_Label.png")
plt.savefig(save_path)
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../../data/interim/da_features.pkl")