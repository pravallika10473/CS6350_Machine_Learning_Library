import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data and perform the necessary preprocessing here
train_file = "datasets/bank/train.csv"
test_file = "datasets/bank/test.csv"

# Assuming a binary classification problem with labels -1 and 1
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
train_df = pd.read_csv(train_file, names=column_headers, dtype=dtype_dict)

X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values

test_df = pd.read_csv(test_file, names=column_headers, dtype=dtype_dict)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values


class DecisionTreeStump:
    def __init__(self, attribute, threshold, label_if_less, label_if_greater):
        self.attribute = attribute
        self.threshold = threshold
        self.label_if_less = label_if_less
        self.label_if_greater = label_if_greater

    def predict(self, x):
        if x[self.attribute] <= self.threshold:
            return self.label_if_less
        else:
            return self.label_if_greater

    @staticmethod
    def calculate_information_gain(X, y, attribute, threshold, weights):
        n = len(y)
        left_indices = X[:, attribute] <= threshold
        right_indices = X[:, attribute] > threshold

        left_weight = np.sum(weights[left_indices])
        right_weight = np.sum(weights[right_indices])

        if left_weight == 0 or right_weight == 0:
            return 0

        left_entropy = -np.sum(weights[left_indices] * np.log2(weights[left_indices] / left_weight))
        right_entropy = -np.sum(weights[right_indices] * np.log2(weights[right_indices] / right_weight))

        total_entropy = (left_weight / n) * left_entropy + (right_weight / n) * right_entropy
        return total_entropy

    @staticmethod
    def find_best_split(X, y, weights):
        num_features = X.shape[1]
        best_threshold = None
        best_attribute = None
        min_entropy = float('inf')

        for attribute in range(num_features):
            unique_values = np.unique(X[:, attribute])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                entropy = DecisionTreeStump.calculate_information_gain(X, y, attribute, threshold, weights)

                if entropy < min_entropy:
                    min_entropy = entropy
                    best_threshold = threshold
                    best_attribute = attribute

        return best_attribute, best_threshold


class AdaBoost:
    def __init__(self, num_iterations, learning_rate=0.5):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.alphas = []
        self.stumps = []
        self.training_errors = []  # Store training errors
        self.test_errors = []  # Store test errors
        self.stump_errors = []  # Store errors of each decision stump

    def fit(self, X, y):
        X = self.convert_to_numeric(X)
        y = y.astype(float)
        n = len(y)
        weights = np.ones(n) / n

        for t in range(self.num_iterations):
            stump = self.train_stump(X, y, weights)
            predictions = np.array([stump.predict(x) for x in X])
            predictions = predictions.astype(float)

            error = np.sum(weights * (predictions != y))

            if error == 0:
                alpha = 1.0
            else:
                alpha = self.learning_rate * np.log((1 - error) / max(error, 1e-10))
            self.alphas.append(alpha)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)

            # Calculate and store training and test errors for this iteration
            train_predictions = self.predict(X_train)
            test_predictions = self.predict(X_test)
            train_error = 1 - accuracy_score(y_train, train_predictions)
            test_error = 1 - accuracy_score(y_test, test_predictions)
            self.training_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(error / n)

    def convert_to_numeric(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]

            if not np.issubdtype(col.dtype, np.number):
                col = pd.to_numeric(col, errors='coerce')
                col[np.isnan(col)] = 0

            X[:, i] = col

        return X

    def train_stump(self, X, y, weights):
        best_error = float('inf')
        best_stump = None

        for _ in range(2):
            attribute, threshold = DecisionTreeStump.find_best_split(X, y, weights)
            for label_if_less in [-1, 1]:
                predictions = np.where(X[:, attribute] <= threshold, label_if_less, -label_if_less)
                error = np.sum(weights * (predictions != y))
                if error < best_error:
                    best_error = error
                    best_stump = DecisionTreeStump(attribute=attribute, threshold=threshold,
                                                   label_if_less=label_if_less, label_if_greater=-label_if_less)

        return best_stump

    def predict(self, X):
        n = X.shape[0]
        predictions = np.zeros(n)

        for alpha, stump in zip(self.alphas, self.stumps):
            predictions += alpha * np.array([stump.predict(x) for x in X])

        return np.sign(predictions)


# Example usage
num_iterations = 500  # Set the number of iterations to 500 as mentioned in the question
learning_rate = 0.50  # Adjust the learning rate as needed
adaboost = AdaBoost(num_iterations, learning_rate)

# Fit the AdaBoost classifier
adaboost.fit(X_train, y_train)

# Calculate and print the training and test errors
train_predictions = adaboost.predict(X_train)
test_predictions = adaboost.predict(X_test)
train_error = 1 - accuracy_score(y_train, train_predictions)
test_error = 1 - accuracy_score(y_test, test_predictions)

print("Training Error:", train_error)
print("Test Error:", test_error)

# Plot training and test errors along with the errors of decision stumps
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_iterations + 1), adaboost.training_errors, label='Training Error', marker='o')
plt.plot(range(1, num_iterations + 1), adaboost.test_errors, label='Test Error', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Iteration')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_iterations + 1), adaboost.stump_errors, label='Decision Stump Error', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Decision Stump Errors vs. Iteration')
plt.legend()

plt.tight_layout()
plt.show()


