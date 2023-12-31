import pandas as pd
import numpy as np

# Load the training and test datasets using Pandas
train_data = pd.read_csv('dataset/bank-note/train.csv', header=None)
test_data = pd.read_csv('dataset/bank-note/test.csv', header=None)

# Extract features and labels from the datasets
X_train = train_data.iloc[:, :-1].values  # Features in training data
y_train = train_data.iloc[:, -1].values    # Labels in training data
X_test = test_data.iloc[:, :-1].values     # Features in test data
y_test = test_data.iloc[:, -1].values       # Labels in test data

# Initialize the weight vector to zeroes
n = X_train.shape[1]
w = np.zeros(n)

# Set the learning rate and the maximum number of epochs
r = 0.1
T = 10

# Training the Perceptron
for epoch in range(1, T+1):
    # Shuffle the data
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]

        if yi * np.dot(w, xi) <= 0:
            w = w + r * yi * xi

# Calculate the average prediction error on the test dataset
test_errors = 0
for i in range(len(X_test)):
    if y_test[i] * np.dot(w, X_test[i]) <= 0:
        test_errors += 1

average_error = test_errors / len(X_test)

# Report the learned weight vector and average prediction error
print("Learned Weight Vector:", w)
print("Average Prediction Error on Test Data:", average_error)


