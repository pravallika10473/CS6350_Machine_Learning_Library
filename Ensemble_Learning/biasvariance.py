from baggedtree import bagged_trees_classifier, DecisionTreeClassifier
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


# Function to compute bias and variance
def compute_bias_variance(predictions, ground_truth):
    # Compute bias (average prediction - ground-truth label)
    bias = np.mean(predictions) - ground_truth

    # Compute variance
    variance = np.var(predictions)

    return bias, variance

# Function to perform the experiment
# Function to perform the experiment
# Function to perform the experiment
def run_experiment(X_train, y_train, X_test, y_test, num_iterations=100, num_bagged_trees=500):
    single_tree_biases = []
    single_tree_variances = []
    bagged_tree_biases = []
    bagged_tree_variances = []

    for _ in range(num_iterations):
        # Step 1: Sample 1,000 examples uniformly without replacement from the training dataset
        n_samples = X_train.shape[0]
        sample_indices = np.random.choice(n_samples, size=1000, replace=False)
        sampled_X_train, sampled_y_train = X_train[sample_indices], y_train[sample_indices]


        # Step 2: Run bagged trees learning algorithm based on the 1,000 training examples and learn 500 trees
        bagged_trees_classifier = BaggedTreesClassifier(num_bagged_trees)
        bagged_trees_classifier.fit(sampled_X_train, sampled_y_train)

        # Step 3: Compute bias and variance for single trees and bagged trees
        # Single trees
        single_tree_predictions = np.array([tree.predict(X_test) for tree in bagged_trees_classifier.trees])
        avg_single_tree_predictions = np.mean(single_tree_predictions, axis=0)
        single_tree_bias, single_tree_variance = compute_bias_variance(avg_single_tree_predictions, y_test)
        single_tree_biases.append(single_tree_bias)
        single_tree_variances.append(single_tree_variance)

        # Bagged trees
        bagged_tree_predictions = bagged_trees_classifier.predict(X_test)
        bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_tree_predictions, y_test)
        bagged_tree_biases.append(bagged_tree_bias)
        bagged_tree_variances.append(bagged_tree_variance)

    # Calculate average bias, variance, and general squared error
    avg_single_tree_bias = np.mean(single_tree_biases)
    avg_single_tree_variance = np.mean(single_tree_variances)
    avg_bagged_tree_bias = np.mean(bagged_tree_biases)
    avg_bagged_tree_variance = np.mean(bagged_tree_variances)

    return avg_single_tree_bias, avg_single_tree_variance, avg_bagged_tree_bias, avg_bagged_tree_variance

# Run the experiment
avg_single_tree_bias, avg_single_tree_variance, avg_bagged_tree_bias, avg_bagged_tree_variance = run_experiment(
    X_train, y_train, X_test, y_test)

# Print the results
print("Average bias for single decision tree:", avg_single_tree_bias)
print("Average variance for single decision tree:", avg_single_tree_variance)
print("Average bias for bagged trees:", avg_bagged_tree_bias)
print("Average variance for bagged trees:", avg_bagged_tree_variance)