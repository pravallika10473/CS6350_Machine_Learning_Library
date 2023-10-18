import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('dataset/concrete/train.csv')
X_train = train_data.iloc[:, :-1].values  # Features
y_train = train_data.iloc[:, -1].values  # Target (SLUMP)

# Load the test data
test_data = pd.read_csv('dataset/concrete/test.csv')
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Add a column of ones to the features for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

def gradient_descent(X, y, w, learning_rate, tolerance, max_iterations=100000):
    cost_history = []
    iteration = 0
    while iteration < max_iterations:
        y_pred = np.dot(X, w)
        error = y_pred - y
        gradient = np.dot(X.T, error)
        w_new = w - learning_rate * gradient
        weight_change = np.linalg.norm(w_new - w)
        w = w_new
        cost = (1 / (2 * len(y))) * np.sum(error ** 2)
        cost_history.append(cost)
        if weight_change < tolerance:
            break
        iteration += 1
    return w, cost_history

# Initialize parameters
w = np.zeros(X_train.shape[1])  # Initialize weight vector
learning_rate = 0.01  # Initial learning rate (reduced)
tolerance = 1e-6

# Perform gradient descent with learning rate tuning
final_w, cost_history = gradient_descent(X_train, y_train, w, learning_rate, tolerance)

# Plot cost function change with steps
plt.plot(range(1, len(cost_history) + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iterations')
plt.show()

# Evaluate on test data
y_pred_test = np.dot(X_test, final_w)
test_cost = (1 / (2 * len(y_test))) * np.sum((y_pred_test - y_test) ** 2)
print(f"Final weight vector: {final_w}")
print(f"Learning rate (r): {learning_rate}")
print(f"Test data cost function: {test_cost}")
