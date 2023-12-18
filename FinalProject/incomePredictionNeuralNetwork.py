import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the cleaned and encoded training data
train_data = pd.read_csv("dataset/train_final_encoded.csv")

# Split the training data into features and labels
X = train_data.drop('income>50K', axis=1)
y = train_data['income>50K']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Identify missing categorical features in the training data
missing_categorical_columns = set(categorical_columns) - set(X_train.columns)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(lr=0.001)  # You can adjust the learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# Load the cleaned and encoded test data
test_data = pd.read_csv("dataset/test_final_encoded.csv")

# Create empty columns for missing categorical features in the test data
for col in missing_categorical_columns:
    test_data[col] = 0

# Ensure that the test data includes all the columns that the model was trained on
X_test = test_data[X_train.columns]
X_test_scaled = scaler.transform(X_test)

# Generate predictions on the test data
predictions = model.predict(X_test_scaled)

# Assuming that predictions represents the probability of "income > 50K"
# You can adjust this based on the output of your neural network

# Generate an array of IDs for the test data starting from 1
test_ids = range(1, len(test_data) + 1)

# Create a DataFrame for the predictions with IDs starting from 1
predictions_df = pd.DataFrame({'ID': test_ids, 'Prediction': predictions.flatten()})

# Save the predictions in the desired format
predictions_df.to_csv("output/neuralNetworkPredictions.csv", index=False)

