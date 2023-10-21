import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd 

# Function to calculate information gain
def information_gain(y, splits):
    # Calculate entropy before splitting
    entropy_before = -np.sum([(len(split) / len(y)) * np.log2(len(split) / len(y)) for split in splits])

    # Calculate entropy after splitting
    entropy_after = np.sum([(len(split) / len(y)) * (-np.sum((np.bincount(split) / len(split)) * np.log2(np.bincount(split) / len(split))) if len(split) > 0 else 0) for split in splits])

    # Calculate information gain
    return entropy_before - entropy_after

# Function to build a decision tree with feature subset
def build_decision_tree(X, y, feature_subset):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(feature_subset) == 0:
        return np.bincount(y).argmax()
    feature_idx, threshold, splits = find_best_split(X, y, feature_subset)
    if len(splits[0]) == 0 or len(splits[1]) == 0:
        return np.bincount(y).argmax()
    subtree = (feature_idx, threshold, [build_decision_tree(X[splits[0]], y[splits[0]], feature_subset), build_decision_tree(X[splits[1]], y[splits[1]], feature_subset)])
    return subtree

# Function to find the best split based on information gain
def find_best_split(X, y, feature_subset):
    best_info_gain = -1
    best_feature_idx = None
    best_threshold = None
    best_splits = None

    for feature_idx in feature_subset:
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        for threshold in unique_values:
            left_split = y[feature_values <= threshold]
            right_split = y[feature_values > threshold]
            splits = [left_split, right_split]
            info_gain = information_gain(y, splits)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = feature_idx
                best_threshold = threshold
                best_splits = splits

    return best_feature_idx, best_threshold, best_splits

# Function to make predictions using a single decision tree
def predict_tree(tree, X):
    if isinstance(tree, np.int64):
        return tree
    feature_idx, threshold, subtrees = tree
    if X[feature_idx] <= threshold:
        return predict_tree(subtrees[0], X)
    else:
        return predict_tree(subtrees[1], X)

# Function to train a random forest
def train_random_forest(X, y, n_trees, feature_subset_size):
    forest = []
    for _ in range(n_trees):
        # Randomly select a feature subset
        feature_subset = np.random.choice(X.shape[1], feature_subset_size, replace=False)
        # Create a bootstrap sample
        sample_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_sampled = X[sample_indices]
        y_sampled = y[sample_indices]
        # Build a decision tree with the feature subset
        tree = build_decision_tree(X_sampled, y_sampled, feature_subset)
        forest.append(tree)
    return forest

# Function to make predictions using the random forest
def predict_random_forest(forest, X):
    predictions = [predict_tree(tree, X) for tree in forest]
    return np.bincount(predictions).argmax()

# Function to calculate training and test errors
def calculate_errors(X_train, y_train, X_test, y_test, forest):
    train_errors = []
    test_errors = []
    for n_trees in range(1, len(forest) + 1):
        train_predictions = [predict_random_forest(forest[:n], X_train[i]) for i in range(len(X_train))]
        test_predictions = [predict_random_forest(forest[:n], X_test[i]) for i in range(len(X_test))]
        train_error = 1 - accuracy_score(y_train, train_predictions)
        test_error = 1 - accuracy_score(y_test, test_predictions)
        train_errors.append(train_error)
        test_errors.append(test_error)
    return train_errors, test_errors

# Load your dataset, split it into training and test sets, and define X_train, y_train, X_test, y_test
column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
dtype_dict = {
    'age': float,
    'job': str,
    'marital': str,
    'education': str,
    'default': str,
    'balance': float,
    'housing': str,
    'loan': str,
    'contact': str,
    'day': float,
    'month': str,
    'duration': float,
    'campaign': float,
    'pdays': float,
    'previous': float,
    'poutcome': str,
    'y': str 
}
train_file = "datasets/bank/train.csv"
test_file = "datasets/bank/test.csv"
train_df = pd.read_csv(train_file, names=column_headers, dtype=dtype_dict)
X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

test_df = pd.read_csv(test_file, names=column_headers, dtype=dtype_dict)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)


# Define the range of the number of trees and feature subset sizes
n_trees_range = range(1, 501)
feature_subset_sizes = [2, 4, 6]

# Initialize lists to store errors for each feature subset size
train_errors_per_feature_size = []
test_errors_per_feature_size = []

for feature_subset_size in feature_subset_sizes:
    forest = train_random_forest(X_train, y_train, max(n_trees_range), feature_subset_size)
    train_errors, test_errors = calculate_errors(X_train, y_train, X_test, y_test, forest)
    train_errors_per_feature_size.append(train_errors)
    test_errors_per_feature_size.append(test_errors)

# Create a figure to visualize the results
plt.figure(figsize=(12, 6))
for i, feature_subset_size in enumerate(feature_subset_sizes):
    plt.plot(n_trees_range, train_errors_per_feature_size[i], label=f'Train (k={feature_subset_size})')
    plt.plot(n_trees_range, test_errors_per_feature_size[i], label=f'Test (k={feature_subset_size})')

plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Random Forest Performance')
plt.legend()
plt.grid(True)
plt.show()
