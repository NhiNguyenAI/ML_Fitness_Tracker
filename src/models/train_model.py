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
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state=42, stratify= y)

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
selected_features, ordered_features, ordered_scores = learner.forward_selection( max_features, X_train, y_train)

# After the Selection of the best features, we can plot the best features of the model
# Selected _features = ordered_scores
selected_features = [
                        'acc_z_freq_0.0_Hz_ws_14',
                        'acc_x_freq_0.0_Hz_ws_14',
                        'duration',
                        'acc_y_freq_0.0_Hz_ws_14',
                        'gyr_r_freq_0.0_Hz_ws_14',
                        'acc_x_freq_2.143_Hz_ws_14',
                        'acc_z_temp_mean_ws_5',
                        'acc_x_temp_std_ws_5',
                        'gyr_r_freq_1.429_Hz_ws_14',
                        'gyr_z'
                    ]
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
possible_feature_sets = [
                            set_festure_1,
                            set_festure_2,
                            set_festure_3,
                            set_festure_4,
                            selected_features
                        ]

feature_names = [
                  "set_festure_1",
                  "set_festure_2",
                  "set_festure_3",
                  "set_festure_4",
                  "selected_features"
                ]

# Number of iterations for training
iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by = "accuracy", ascending = False, inplace = True)

# plot the best  festures of the model
fig, ax = plt.subplots(figsize = (10,5))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.x_label = "model"
plt.y_label = "Accuracy"
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# choose the best model after plot the best features of the model
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
            X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy= accuracy_score(y_test, class_test_y)

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

# Create confusion matrix
classes  = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

# create a list of participants
participants_df = df.drop(["set", "category"], axis = 1)
participants_df.info()

# Choose a participant for training

X_train = participants_df[participants_df["participant"] != "A"].drop("label", axis = 1)
y_train = participants_df[participants_df["participant"] != "A"]["label"]

# Choose a participant for testing

X_test = participants_df[participants_df["participant"] == "A"].drop("label", axis = 1)
y_test = participants_df[participants_df["participant"] == "A"]["label"]

# Clearn the participant data for the traning
X_train = X_train.drop(["participant"], axis = 1)
X_test = X_test.drop(["participant"], axis = 1)


# plot bar plot for the y_train and y_test
fig, ax = plt.subplots(figsize = (10,5))
df_train["label"].value_counts().plot(kind = "bar",color= 'green', ax = ax, label = "Total")
y_train.value_counts().plot(kind = "bar", ax = ax, color = 'black', label = "y_train")
y_test.value_counts().plot(kind = "bar", ax = ax, color= 'blue', label = "y_test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
            X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy= accuracy_score(y_test, class_test_y)

# Create confusion matrix
classes  = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Try a complex model with the selected features
# Use the neural network model and evaluate results without grid search
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
            X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
)

accuracy= accuracy_score(y_test, class_test_y)

# Create confusion matrix
classes  = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
