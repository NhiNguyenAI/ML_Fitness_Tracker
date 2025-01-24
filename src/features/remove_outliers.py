import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_process.pkl")

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

# for gyr
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
df[columns_outliers[3:6] + ["label"]].boxplot(by= "label", ax=axes, layout=(1, 3))


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


# Plot a single column


# Loop over all columns


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Insert Chauvenet's function


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