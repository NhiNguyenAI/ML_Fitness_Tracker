import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor


# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../../data/interim/data_process.pkl")

# --------------------------------------------------------------
# 2. Plotting outliers
# --------------------------------------------------------------

# --------------------------------------------------------------
# 2.1 Adjust plot settings
# --------------------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100

# take careful, working now on dataframe -> plot with boxplot
type(df[["acc_x"]]) # dataframe
type(df["acc_x"])   # Series

df[["acc_x"]].boxplot()

# boxplot of acc_x following the label

df[["acc_x", "label"]].boxplot(by = "label", figsize = (20,10))

# looking for multi axis at the same time, and plot boxplot
columns_outliers = df.columns[:6] 
# columns_outliers is now included with index -> list()
columns_outliers = list(df.columns[:6])

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

# above this diagramm have alot of outliers data can't viasualze#
# Solusion: Use customer function to visualize the outliers in time
def plot_binary_outliers(dataset, col, outlier_col, reset_index, save_path=None):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column
col = "acc_x"
mark_outliers_iqr_df = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset=mark_outliers_iqr_df, col=col, outlier_col= col + "_outlier", reset_index =True)

# Loop over all columns
columns_outliers = list(df.columns[:6])
for col in columns_outliers:
    mark_outliers_iqr_df = mark_outliers_iqr(df,col)
    save_path = f"../../../reports/figures/IQR_binary_outliers/{col}.png"
    plot_binary_outliers(dataset=mark_outliers_iqr_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# Problem: Looking the data on basically own one big pile. Some data in diagram is extrem higher oder lower. For Example in this case is ar the break time,  people can do everthing, that result the data of the watch sometime not like in the exercise like squat

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------
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

# Check for normal distribution
columns_outliers = list(df.columns[:6])
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
df[columns_outliers[:3] + ["label"]].plot.hist( by = "label",figsize = (20,20), layout=(3, 3))
df[columns_outliers[3:6] + ["label"]].plot.hist(by = "label", figsize = (20,20), layout=(3, 3))

# Insert Chauvenet's function
columns_outliers = list(df.columns[:6])
for col in columns_outliers:
    mark_outliers_iqr_df = mark_outliers_chauvenet(df,col)
    save_path = f"../../../reports/figures/interquartile_range/{col}.png"
    plot_binary_outliers(dataset=mark_outliers_iqr_df, col=col, outlier_col= col + "_outlier", reset_index =True, save_path=save_path)

# Loop over all columns


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


# Loop over all columns


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

