import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Load the data
train_data = pd.read_csv("dataset/bank-note/train.csv", header=None)
test_data = pd.read_csv("dataset/bank-note/test.csv", header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)

# Set the maximum epochs T to 100
max_epochs = 100

# Hyperparameter C values
C_values = [100/873, 500/873, 700/873]

# Learning rate schedule parameters
gamma0 = 0.1
a = 0.01

# SVM with stochastic sub-gradient descent
def svm_sgd(X, y, C, gamma0, a, max_epochs, learning_rate_schedule):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            
            if learning_rate_schedule == 1:
                # Schedule: gamma_t = gamma0 / (1 + t)
                eta = gamma0 / (1 + updates)
            elif learning_rate_schedule == 2:
                # Schedule: gamma_t = gamma0 / (1 + (gamma0 */a )* t)
                eta = gamma0 / (1 + (gamma0 / a) * updates)
            else:
                raise ValueError("Invalid learning rate schedule")
                
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w_prev = w.copy()  # Store previous weights
                b_prev = b  # Store previous bias
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
                
    return w, b, w_prev, b_prev

# Training and testing for different hyperparameter settings and learning rate schedules
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    # Initialize variables for differences
    w_diff = 0
    b_diff = 0
    train_error_diff = 0
    test_error_diff = 0

    for schedule in [1, 2]:
        w, b, w_prev, b_prev = svm_sgd(X_train, y_train, C, gamma0, a, max_epochs, schedule)

        # Calculate training error
        train_predictions = np.sign(np.dot(X_train, w) + b)
        train_error = np.mean(train_predictions != y_train)

        # Calculate test error
        test_predictions = np.sign(np.dot(X_test, w) + b)
        test_error = np.mean(test_predictions != y_test)

        # Update differences
        if schedule == 2:
            w_diff = np.linalg.norm(w - w_prev)
            b_diff = np.abs(b - b_prev)
            train_error_diff = np.abs(train_error - train_error_prev)
            test_error_diff = np.abs(test_error - test_error_prev)

        train_error_prev = train_error
        test_error_prev = test_error

    # Print differences for each C
    print(f"C: {C}")
    print(f"Weights Difference between Schedules: {w_diff}")
    print(f"Bias Difference between Schedules: {b_diff}")
    print(f"Training Error Difference between Schedules: {train_error_diff}")
    print(f"Test Error Difference between Schedules: {test_error_diff}")
    print("=" * 30)


