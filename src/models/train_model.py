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
df.info()

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.copy()
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
basis_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

square_features = ["acc_r","gyr_r"]

pca_features = ["pca_1", "pca_2", "pca_3"]

time_features = [f for f in df_train.columns if "_temp_" in f]

frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]

cluster_features = ["cluster"]

print("basis_feature", len(basis_features))
print("square_features", len(square_features))
print("pca_features", len(pca_features))
print("time_features", len(time_features))
print("frequency_features", len(frequency_features))
print("cluster_features", len(cluster_features))

# Make sure that set feature muss be list
set_festure_1 = list(set(basis_features))
set_festure_2 = list(set(basis_features + square_features + pca_features))
set_festure_3 = list(set(set_festure_2 + time_features))
set_festure_4 = list(set(set_festure_3 + frequency_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
'''
Feature selection: Decision Tree

Forward selection for classification which selects a pre-defined number of features (max_features) that show the best accuracy. 
We assume a decision tree learning for this purpose, but this can easily be changed.
It return the best features.

'''
max_features = 10
learner = ClassificationAlgorithms()

# this step is not using the train test slit, this is using the tranning data
selected_features, ordered_features, ordered_scores = learner.forward_selection( max_features, x_train, y_train)

# plot the best  festures of the model
fig, ax = plt.subplots(figsize = (10,5))
plt.plot(np.arange(1, max_features+1), ordered_scores, marker = "o")
plt.xticks(np.arange(1, max_features+1))
plt.x_label = "Number of features"
plt.y_label = "Accuracy"
plt.title("Forward feature selection")
plt.show()


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
