# Predict_model.py

import sqlite3
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.impute import SimpleImputer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = joblib.load('insurance_model.pkl')
logging.info("Model loaded successfully")
print()

# Connecting to the database
conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')
data = pd.read_sql_query('SELECT * FROM Insurance_Prediction', conn)

# Close the database connection
conn.close()

# Data Preprocessing
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data['children'] = pd.to_numeric(data['children'], errors='coerce')
data['charges'] = pd.to_numeric(data['charges'], errors='coerce')

# Convert categorical columns to 'category' dtype
categorical_columns = [
    'gender', 'smoker', 'region', 'medical_history', 'family_medical_history',
    'exercise_frequency', 'occupation', 'coverage_level'
]
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Feature Engineering (Binning BMI)
bins = [0, 18.5, 24.9, 29.9, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
data['bmi_category'] = pd.cut(data['bmi'], bins=bins, labels=labels)

# One-hot encode bmi_category and other categorical variables
data_encoded = pd.get_dummies(data, columns=[
    'gender', 'smoker', 'region', 'medical_history', 'family_medical_history',
    'exercise_frequency', 'occupation', 'coverage_level', 'bmi_category'], drop_first=True)

# Splitting the data into evaluation and live data
eval_data = data_encoded[700000:900000]
live_data = data_encoded[900000:]

# Features and target
X_eval = eval_data.drop(columns=['charges'])
y_eval = eval_data['charges']
X_live = live_data.drop(columns=['charges'])
y_live = live_data['charges']

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')
X_eval_imputed = imputer.fit_transform(X_eval)
X_live_imputed = imputer.transform(X_live)

# Convert the NumPy arrays back to pandas DataFrame with the original column names
X_eval_imputed = pd.DataFrame(X_eval_imputed, columns=X_eval.columns)
X_live_imputed = pd.DataFrame(X_live_imputed, columns=X_live.columns)

# Predictions on evaluation data
y_pred_eval = model.predict(X_eval_imputed)
mse_eval = mean_squared_error(y_eval, y_pred_eval)
r2_eval = r2_score(y_eval, y_pred_eval)
mae_eval = mean_absolute_error(y_eval, y_pred_eval)
rmse_eval = np.sqrt(mse_eval)

logging.info(f"Evaluation Data - Mean Squared Error: {mse_eval}")
print()
logging.info(f"Evaluation Data - R^2: {r2_eval}")
print()
logging.info(f"Evaluation Data - Mean Absolute Error: {mae_eval}")
print()
logging.info(f"Evaluation Data - Root Mean Squared Error: {rmse_eval}")
print()

# Predictions on live data
y_pred_live = model.predict(X_live_imputed)
mse_live = mean_squared_error(y_live, y_pred_live)
r2_live = r2_score(y_live, y_pred_live)
mae_live = mean_absolute_error(y_live, y_pred_live)
rmse_live = np.sqrt(mse_live)

logging.info(f"Live Data - Mean Squared Error: {mse_live}")
print()
logging.info(f"Live Data - R^2: {r2_live}")
print()
logging.info(f"Live Data - Mean Absolute Error: {mae_live}")
print()
logging.info(f"Live Data - Root Mean Squared Error: {rmse_live}")
