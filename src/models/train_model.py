import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis = 1) # axis = 1 , remove the column

X = df_train.drop(["label"], axis = 1)
y = df_train["label"]

# Take 25 precent of dataset for the test 0.25
# Set the stratify = y , y is label, because i want to ensure equal distrubution of all labels
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state=42, stratify= y)

fig, ax = plt.subplots(figsize = (10,5))
df_train["label"].value_counts().plot(kind = "bar",color= 'green', ax = ax, label = "Label")
y_train.value_counts().plot(kind = "bar", ax = ax, color = 'black', label = "y_train")
y_test.value_counts().plot(kind = "bar", ax = ax, color= 'blue', label = "y_test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
