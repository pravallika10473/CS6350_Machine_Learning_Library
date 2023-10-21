from baggedtree1 import bagged_trees_classifier, DecisionTreeClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

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

# Initialize variables to store results
num_experiments = 1
num_samples = 1
num_trees_bagged = 1
num_trees_single = 1

bias_single_tree = 0
variance_single_tree = 0
squared_error_single_tree = 0

bias_bagged_trees = 0
variance_bagged_trees = 0
squared_error_bagged_trees = 0

for _ in range(num_experiments):
    # Step 1: Sample 1,000 examples uniformly without replacement
    sample_indices = np.random.choice(len(X_train), num_samples, replace=False)
    X_sample = X_train[sample_indices]
    y_sample = y_train[sample_indices]

    # Step 2: Train a Bagged Trees ensemble with 100 trees
    bagged_trees = BaggedTrees(n_estimators=num_trees_bagged)
    bagged_trees.fit(X_sample, y_sample)

    # Train a single decision tree for comparison
    single_tree = DecisionTree(max_depth=None)
    single_tree.fit(X_sample, y_sample)

    # Predictions for single tree
    y_single_tree_predictions = []
    for x in X_test:
        prediction = single_tree.predict([x])
        y_single_tree_predictions.append(prediction)

    # Predictions for Bagged Trees
    y_bagged_trees_predictions = bagged_trees.predict(X_test)

    # Calculate bias for single tree
    bias_single_tree += np.mean(np.square(y_single_tree_predictions - y_test))

    # Calculate variance for single tree
    variance_single_tree += np.var(y_single_tree_predictions)

    # Calculate bias for Bagged Trees
    bias_bagged_trees += np.mean(np.square(y_bagged_trees_predictions - y_test))

    # Calculate variance for Bagged Trees
    variance_bagged_trees += np.var(y_bagged_trees_predictions)

# Calculate squared error for both single tree and Bagged Trees
squared_error_single_tree = bias_single_tree + variance_single_tree
squared_error_bagged_trees = bias_bagged_trees + variance_bagged_trees

# Calculate the average bias, variance, and squared error
bias_single_tree /= num_experiments
variance_single_tree /= num_experiments
bias_bagged_trees /= num_experiments
variance_bagged_trees /= num_experiments

# Print the results
print("Single Tree - Bias:", bias_single_tree)
print("Single Tree - Variance:", variance_single_tree)
print("Single Tree - Squared Error:", squared_error_single_tree)
print("Bagged Trees - Bias:", bias_bagged_trees)
print("Bagged Trees - Variance:", variance_bagged_trees)
print("Bagged Trees - Squared Error:", squared_error_bagged_trees)
