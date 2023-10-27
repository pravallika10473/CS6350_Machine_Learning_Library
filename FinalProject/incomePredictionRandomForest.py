import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the training and test datasets
train_file = "datasets/train_final.csv"
test_file = "datasets/test_final.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# List of numerical and categorical columns
numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Replace missing values ('?') with median for numerical columns
for col in numerical_columns:
    train_data[col] = train_data[col].replace('?', np.nan)
    test_data[col] = test_data[col].replace('?', np.nan)

    train_data[col] = train_data[col].astype(float)
    test_data[col] = test_data[col].astype(float)

    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(test_data[col].median())

# Replace missing values ('?') with mode for categorical columns
for col in categorical_columns:
    train_data[col] = train_data[col].replace('?', np.nan)
    test_data[col] = test_data[col].replace('?', np.nan)

    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# Save the cleaned data
train_data.to_csv("datasets/train_final_cleaned.csv", index=False)
test_data.to_csv("datasets/test_final_cleaned.csv", index=False)

# Apply one-hot encoding to categorical columns
train_data = pd.get_dummies(train_data, columns=categorical_columns)
test_data = pd.get_dummies(test_data, columns=categorical_columns)

# Save the cleaned and one-hot encoded data
train_data.to_csv("datasets/train_final_encoded.csv", index=False)
test_data.to_csv("datasets/test_final_encoded.csv", index=False)

# Load the cleaned and encoded training data
train_data = pd.read_csv("datasets/train_final_encoded.csv")

# Split the training data into features and labels
X_train = train_data.drop('income>50K', axis=1)
y_train = train_data['income>50K']

# Train a Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Identify missing categorical features in the test data
missing_categorical_columns = set(categorical_columns) - set(X_train.columns)

# Load the cleaned and encoded test data
test_data = pd.read_csv("datasets/test_final_encoded.csv")

# Create empty columns for missing categorical features in the test data
for col in missing_categorical_columns:
    test_data[col] = 0

# Ensure that the test data includes all the columns that the model was trained on
expected_columns = list(X_train.columns)
X_test = test_data[expected_columns]

# Generate predictions on the test data
predictions = random_forest.predict_proba(X_test)  # Probability predictions

# Assuming that predictions[:, 1] represents the probability of "income > 50K"
# You can adjust this based on the output of your RandomForestClassifier

# Generate an array of IDs for the test data starting from 1
test_ids = range(1, len(test_data) + 1)

# Create a DataFrame for the predictions with IDs starting from 1
predictions_df = pd.DataFrame({'ID': test_ids, 'Prediction': predictions[:, 1]})

# Save the predictions in the desired format
predictions_df.to_csv("randomForestPredictions.csv", index=False)
