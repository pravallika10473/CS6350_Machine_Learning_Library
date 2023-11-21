import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Load the data
train_data = pd.read_csv("dataset/bank-note/train.csv", header=None)
test_data = pd.read_csv("dataset/bank-note/test.csv", header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# Set the maximum epochs T to 100
max_epochs = 100

# Hyperparameter C values
C_values = [100/873, 500/873, 700/873]

# Learning rate schedule parameters
gamma0 = 0.1

# SVM with stochastic sub-gradient descent
def svm_sgd(X, y, C, gamma0, max_epochs):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            eta = gamma0 / (1 + updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
                
    return w, b

# Training and testing for different hyperparameter settings
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    w, b = svm_sgd(X_train, y_train, C, gamma0, max_epochs)
    
    # Calculate training error
    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)
    
    # Calculate test error
    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)
    
    # Print results
    print(f"C: {C}")
    print(f"Training Error: {train_error}, Test Error: {test_error}")
    print("="*30)