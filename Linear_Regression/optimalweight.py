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

# Calculate the optimal weight vector analytically
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Make sure bias term is included
optimal_w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print("Optimal Weight Vector:", optimal_w)
