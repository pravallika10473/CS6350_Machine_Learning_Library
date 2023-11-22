import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

# Load the data
train_data = pd.read_csv("dataset/bank-note/train.csv", header=None)
test_data = pd.read_csv("dataset/bank-note/test.csv", header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

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

# SVM in the dual domain with Gaussian kernel
def dual_objective_gaussian(alpha, X, y, C, gamma):
    n, d = X.shape
    w = np.dot((alpha * y), X)
    
    pairwise_distances = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
    kernel_matrix = np.exp(-pairwise_distances / gamma)
    
    hinge_loss = np.maximum(1 - np.dot(kernel_matrix, w), 0)
    regularization_term = 0.5 * np.dot(alpha * y, np.dot(kernel_matrix, alpha * y))

    dual_objective_value = C * np.sum(hinge_loss) + regularization_term

    
    return dual_objective_value


# Initialize variables for dual SVM with Gaussian kernel
alpha_init = np.zeros(len(train_data))
# Hyperparameter C values for SVM
C_values = [100/873, 500/873, 700/873]
# Hyperparameter gamma values for Gaussian kernel
gamma_values = [0.1, 0.5, 1, 5, 100]

results_gaussian = []

for C in C_values:
    for gamma in gamma_values:
        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        
        # Optimize the dual objective function with Gaussian kernel using scipy.optimize.minimize
        result = minimize(dual_objective_gaussian, alpha_init, args=(X_train, y_train, C, gamma),
                          bounds=[(0, C) for _ in range(len(X_train))])
        alpha_optimal = result.x
        
        # Recover feature weights and bias for dual SVM with Gaussian kernel
        w_dual_gaussian = np.dot((alpha_optimal * y_train), X_train)
        b_dual_gaussian = y_train - np.dot(X_train, w_dual_gaussian)
        b_dual_gaussian = np.mean(b_dual_gaussian)
        
        # Calculate training error for dual SVM with Gaussian kernel
        pairwise_distances_train = np.sum((X_train[:, np.newaxis] - X_train[np.newaxis, :]) ** 2, axis=-1)
        kernel_matrix_train = np.exp(-pairwise_distances_train / gamma)
        train_predictions_dual_gaussian = np.sign(np.dot(kernel_matrix_train, w_dual_gaussian) + b_dual_gaussian)
        train_error_dual_gaussian = np.mean(train_predictions_dual_gaussian != y_train)
        
        # Calculate test error for dual SVM with Gaussian kernel
        pairwise_distances_test = np.sum((test_data.iloc[:, :-1].values[:, np.newaxis] - X_train[np.newaxis, :]) ** 2, axis=-1)
        kernel_matrix_test = np.exp(-pairwise_distances_test / gamma)
        test_predictions_dual_gaussian = np.sign(np.dot(kernel_matrix_test, w_dual_gaussian) + b_dual_gaussian)
        test_error_dual_gaussian = np.mean(test_predictions_dual_gaussian != test_data.iloc[:, -1].values)
        
        results_gaussian.append({
            'C': C,
            'gamma': gamma,
            'w': w_dual_gaussian,
            'b': b_dual_gaussian,
            'train_error': train_error_dual_gaussian,
            'test_error': test_error_dual_gaussian
        })

# Find the best combination
best_result_gaussian = min(results_gaussian, key=lambda x: x['test_error'])

# Print results for different combinations of C and gamma
for res_gaussian in results_gaussian:
    C, gamma = res_gaussian['C'], res_gaussian['gamma']
    w_dual_gaussian, b_dual_gaussian = res_gaussian['w'], res_gaussian['b']

    print(f"C: {C}, Gamma: {gamma}")
    print("Dual SVM with Gaussian Kernel:")
    print("  w_dual_gaussian:", w_dual_gaussian)
    print("  b_dual_gaussian:", b_dual_gaussian)
    print("  Train Error: ", res_gaussian['train_error'])
    print("  Test Error: ", res_gaussian['test_error'])
    print("="*30)

# Print the best combination
print("Best Combination:")
print(f"C: {best_result_gaussian['C']}, Gamma: {best_result_gaussian['gamma']}")
print("  Train Error: ", best_result_gaussian['train_error'])
print("  Test Error: ", best_result_gaussian['test_error'])
