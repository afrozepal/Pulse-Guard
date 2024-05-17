import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('Heart_Disease_Prediction.csv')

# Preprocessing steps
df['Heart Disease'] = df['Heart Disease'].replace(['Presence', 'Absence'], [1, 0])
X = df.drop(columns=['Heart Disease'])
Y = df['Heart Disease']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model and scaler from disk
with open('heart_disease_model.pkl', 'rb') as file:
    model_lr = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Provided row (Verify this data is correct)
input_data = [[59, 1, 1, 178, 270, 0, 2, 145, 0, 4.2, 3, 0, 7]]
feature_names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
input_data_df = pd.DataFrame(input_data, columns=feature_names)

# Scale the input data
input_data_scaled = scaler.transform(input_data_df)

# Make predictions
prediction = model_lr.predict(input_data_scaled)
prediction_proba = model_lr.predict_proba(input_data_scaled)

print("Input Data:", input_data)
print("Scaled Input Data:", input_data_scaled)
print("Prediction:", prediction)
print("Prediction Probability:", prediction_proba)

if prediction[0] == 1:
    print("Prediction: Heart Disease present")
else:
    print("Prediction: Heart Disease absent")

# Additional details for debugging
print(f"Prediction Probability: {prediction_proba[0]}")
