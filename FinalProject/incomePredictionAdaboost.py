import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the cleaned and encoded training data
train_data = pd.read_csv("dataset/train_final_encoded.csv")

# Split the training data into features and labels
X_train = train_data.drop('income>50K', axis=1)
y_train = train_data['income>50K']

# Train an AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(random_state=42)
adaboost_classifier.fit(X_train, y_train)

# List of numerical and categorical columns
numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']


# Identify missing categorical features in the test data
missing_categorical_columns = set(categorical_columns) - set(X_train.columns)

# Load the cleaned and encoded test data
test_data = pd.read_csv("dataset/test_final_encoded.csv")

# Create empty columns for missing categorical features in the test data
for col in missing_categorical_columns:
    test_data[col] = 0

# Ensure that the test data includes all the columns that the model was trained on
expected_columns = list(X_train.columns)
X_test = test_data[expected_columns]

# Generate predictions on the test data
predictions = adaboost_classifier.predict_proba(X_test)  # Probability predictions

# Assuming that predictions[:, 1] represents the probability of "income > 50K"
# You can adjust this based on the output of your AdaBoostClassifier

# Generate an array of IDs for the test data starting from 1
test_ids = range(1, len(test_data) + 1)

# Create a DataFrame for the predictions with IDs starting from 1
predictions_df = pd.DataFrame({'ID': test_ids, 'Prediction': predictions[:, 1]})

# Save the predictions in the desired format
predictions_df.to_csv("output/adaboostPredictions.csv", index=False)
