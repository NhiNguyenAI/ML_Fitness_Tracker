import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
import MarkOutlierIqr as iqr
import MarkOutlierSchavenet as schauvenet
import PlotBinaryOutliers as pbo
import LocalOutlierFactor as lof



# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../../data/interim/data_process.pkl")


# --------------------------------------------------------------
# 2 Adjust plot settings
# --------------------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# 3 definite the name of columns
# --------------------------------------------------------------

# columns_outliers is now included with index -> list()
columns_outliers = list(df.columns[:6])

# --------------------------------------------------------------
# 4 Boxplot for acc and gyr
# --------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# for acc
df[columns_outliers[:3] + ["label"]].boxplot(by= "label", ax=axes, layout=(1, 3))
save_path = "../../../reports/figures/interquartile_range/acc.png"
plt.savefig(save_path)

# for gyr
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
df[columns_outliers[3:6] + ["label"]].boxplot(by= "label", ax=axes, layout=(1, 3))
save_path = "../../../reports/figures/interquartile_range/gyr.png"
plt.savefig(save_path)

# --------------------------------------------------------------
# 5 loop over all colums in time
# --------------------------------------------------------------

for col in columns_outliers:
    mark_outliers_iqr_df = iqr.mark_outliers_iqr(df,col)
    save_path = f"../../../reports/figures/IQR_binary_outliers/{col}.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_iqr_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# --------------------------------------------------------------
# 6 Outlier of sensor with Schauvenet
# --------------------------------------------------------------
# Check for normal distribution
columns_outliers = list(df.columns[:6])
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
df[columns_outliers[:3] + ["label"]].plot.hist( by = "label",figsize = (20,20), layout=(3, 3))
df[columns_outliers[3:6] + ["label"]].plot.hist(by = "label", figsize = (20,20), layout=(3, 3))

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset
for col in columns_outliers:
    mark_outliers_schauvenet_df = mark_outliers_chauvenet(df,col)
    save_path = f"../../../reports/figures/chauvenets_criteron/{col}.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_schauvenet_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# --------------------------------------------------------------
# 7 Outlier of sensor with local outlier factor
# -------------------------------------------------------------
dataset_lof_01, outliers01, X_scores01 = lof.mark_outliers_lof(df,columns_outliers)
for col in columns_outliers:
    save_path = f"../../../reports/figures/local_outlier_factor/{col}.png"
    pbo.plot_binary_outliers(dataset=dataset_lof_01, col=col, outlier_col= "outlier_lof", reset_index =True, save_path=save_path)

# --------------------------------------------------------------
# 8 Check outliers grouped by label
# --------------------------------------------------------------

label = "bench"
df[df["label"]==label]
# IQR Method
for col in columns_outliers:
    mark_outliers_iqr_df = iqr.mark_outliers_iqr(df[df["label"]==label],col)
    save_path = f"../../../reports/figures/compare_outlier_method/bench_{col}(iqr).png"
    pbo.plot_binary_outliers(dataset=mark_outliers_iqr_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# chauvenets criteron method
for col in columns_outliers:
    mark_outliers_schauvenet_df = mark_outliers_chauvenet(df[df["label"]==label],col)
    save_path = f"../../../reports/figures/compare_outlier_method/bench_{col}(chauvenet).png"
    pbo.plot_binary_outliers(dataset=mark_outliers_schauvenet_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# LOF Method
columns_outliers = list(df.columns[:6])
dataset_lof_02, outliers01, X_scores01 = lof.mark_outliers_lof(df[df["label"]==label],columns_outliers)
for col in columns_outliers:
    save_path = f"../../../reports/figures/compare_outlier_method/bench_{col}(lof).png"
    pbo.plot_binary_outliers(dataset=dataset_lof_02, col=col, outlier_col= "outlier_lof", reset_index =True, save_path=save_path)

# --------------------------------------------------------------
# 9 Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset_schauvenet_02 = mark_outliers_chauvenet(df, col = col)
dataset_schauvenet_02[dataset_schauvenet_02["gyr_z_outlier"]] # retur all outlier value(true) of gyr_z

dataset_schauvenet_02.loc[dataset_schauvenet_02["gyr_z_outlier"], "gyr_z"] = np.nan # set outliers value in gyr_z to NAN
dataset_schauvenet_02.info()
len(dataset_schauvenet_02[dataset_schauvenet_02["gyr_z_outlier"]])

# Create a loop
outliers_removed_df = df.copy()
for col in columns_outliers:
    for label in df["label"].unique():
        mark_outliers_chavenet_dataset = mark_outliers_chauvenet(df[df["label"]==label],col)

        # Replace values marked as outliers with NaN
        mark_outliers_chavenet_dataset.loc[mark_outliers_chavenet_dataset[col + "_outlier"], col] = np.nan

        # update the column in the original dataframe
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = mark_outliers_chavenet_dataset[col]

        n_outliers = len(mark_outliers_chavenet_dataset) - len(mark_outliers_chavenet_dataset[col].dropna())

        print(f"Removed {n_outliers} from {col} for {label}")

outliers_removed_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../../data/interim/outliers_removed_schauvenets.pkl")