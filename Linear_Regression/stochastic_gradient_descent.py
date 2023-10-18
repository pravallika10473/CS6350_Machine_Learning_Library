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

# Initialize parameters
w = np.zeros(X_train.shape[1])  # Initialize weight vector
learning_rate = 0.001  # Initial learning rate
max_iterations = 10000
cost_history = []

def stochastic_gradient_descent(X, y, w, learning_rate, max_iterations):
    for _ in range(max_iterations):
        # Randomly sample an example
        random_index = np.random.randint(0, len(y))
        xi = X[random_index]
        yi = y[random_index]

        # Calculate the prediction
        y_pred = np.dot(xi, w)

        # Update the weight vector
        error = y_pred - yi
        gradient = xi * error
        w -= learning_rate * gradient

        # Calculate the cost function for the entire dataset
        y_pred_all = np.dot(X, w)
        cost = (1 / (2 * len(y))) * np.sum((y_pred_all - y) ** 2)
        cost_history.append(cost)

    return w, cost_history

# Perform SGD
final_w, cost_history = stochastic_gradient_descent(X_train, y_train, w, learning_rate, max_iterations)

# Plot cost function change with updates
plt.plot(range(1, len(cost_history) + 1), cost_history)
plt.xlabel('Updates')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Updates')
plt.show()

# Evaluate on test data
y_pred_test = np.dot(X_test, final_w)
test_cost = (1 / (2 * len(y_test))) * np.sum((y_pred_test - y_test) ** 2)
print(f"Final weight vector: {final_w}")
print(f"Learning rate (r): {learning_rate}")
print(f"Test data cost function: {test_cost}")