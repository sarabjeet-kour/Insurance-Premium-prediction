# Train_model.py

import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Connecting to the database
conn = sqlite3.connect('C:\\Users\\Sarabjeet Kour\\Database (1).db')
data = pd.read_sql_query('SELECT * FROM Insurance_Prediction', conn)
logging.info("Dataset successfully loaded")
print()

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

# Handling missing values: Impute with median
imputer = SimpleImputer(strategy='median')
data_encoded_imputed = pd.DataFrame(imputer.fit_transform(data_encoded), columns=data_encoded.columns)

# Splitting the data into training, evaluation, and live data
train_data = data_encoded_imputed[:700000]
eval_data = data_encoded_imputed[700000:900000]
live_data = data_encoded_imputed[900000:]

# Features and target
X_train = train_data.drop(columns=['charges'])
y_train = train_data['charges']

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(model, 'insurance_model.pkl')

print()
logging.info("Model training complete and saved as 'insurance_model.pkl'")
