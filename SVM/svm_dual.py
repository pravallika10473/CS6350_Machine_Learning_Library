import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

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
def svm_sgd(X, y, C, gamma0, a, max_epochs):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            eta = gamma0 / (1 + (gamma0 / a) * updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
                
    return w, b

# Dual SVM with scipy.optimize.minimize
def dual_svm(X, y, C):
    n, d = X.shape
    alpha_0 = np.zeros(n)
    
    # Define the objective function for minimization
    def objective(alpha):
        return 0.5 * np.dot(alpha, alpha) - np.sum(alpha)

    # Define the equality constraint
    def constraint(alpha):
        return np.dot(alpha, y)

    # Set bounds for alpha (0 <= alpha <= C)
    bounds = [(0, C) for _ in range(n)]

    # Define the optimization problem
    cons = [{'type': 'eq', 'fun': constraint}]
    result = minimize(objective, alpha_0, method='SLSQP', bounds=bounds, constraints=cons)

    # Extract alpha values from the result
    alpha = result.x

    # Calculate weights and bias
    w = np.dot(alpha * y, X)
    b = np.mean(y - np.dot(X, w))

    return w, b

# Training and testing for different hyperparameter settings
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    # Primal SVM (stochastic sub-gradient descent)
    w_primal, b_primal = svm_sgd(X_train, y_train, C, gamma0, a, max_epochs)
    
    # Dual SVM
    w_dual, b_dual = dual_svm(X_train, y_train, C)
    
    # Calculate training and test errors for primal SVM
    train_predictions_primal = np.sign(np.dot(X_train, w_primal) + b_primal)
    test_predictions_primal = np.sign(np.dot(X_test, w_primal) + b_primal)
    
    train_error_primal = np.mean(train_predictions_primal != y_train)
    test_error_primal = np.mean(test_predictions_primal != y_test)
    
    # Calculate training and test errors for dual SVM
    train_predictions_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    test_predictions_dual = np.sign(np.dot(X_test, w_dual) + b_dual)
    
    train_error_dual = np.mean(train_predictions_dual != y_train)
    test_error_dual = np.mean(test_predictions_dual != y_test)
    
    # Print results
    print(f"C: {C}")
    print("Primal SVM:")
    print(f"Training Error: {train_error_primal}, Test Error: {test_error_primal}")
    
    print("Dual SVM:")
    print(f"Training Error: {train_error_dual}, Test Error: {test_error_dual}")
    
    # Print differences between primal and dual parameters
    w_diff = np.linalg.norm(w_primal - w_dual)
    b_diff = np.abs(b_primal - b_dual)
    
    print(f"Weights Difference between Primal and Dual: {w_diff}")
    print(f"Bias Difference between Primal and Dual: {b_diff}")
    
    print("="*30)
