"""
This method trains and evaluates the performance of classification models on the data, including models such as Decision Tree, Neural Network, Random Forest, KNN, and Naive Bayes. These models are trained with different feature sets and their accuracy is compared after each training iteration.

Parameters:

    - **possible_feature_sets**: Sets of features that can be used for training the model.
    - **feature_names**: Names of the features in the data.
    - **X_train**: Training data for the features.
    - **y_train**: Training labels (output data).
    - **X_test**: Feature data for testing the model.
    - **y_test**: Labels data for testing the model.
    - **iterations**: The number of training iterations to compute average performance.

Returns:
    - score_df: A DataFrame containing the accuracy of each model for each feature set.

The function will train the classification models for each feature set and store the accuracy results in a DataFrame. The classification models trained include:
- Neural Network (NN)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)
- Naive Bayes (NB)

This method, based on the principles from the book "Machine Learning for the Quantified Self".

"""


from sklearn.metrics import accuracy_score
import pandas as pd
from LearningAlgorithms import ClassificationAlgorithms as learner

def classification_training_models(possible_feature_sets,feature_names,X_train, y_train, X_test, y_test, iterations):
    
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
    return score_df