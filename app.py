import joblib
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = joblib.load('insurance_model.pkl')
logging.info("Model loaded successfully")

# List of expected features (from your model training output)
expected_features = [
    'age', 'bmi', 'children', 'gender_female', 'gender_male', 'smoker_yes',
    'region_northwest', 'region_southeast', 'region_southwest',
    'medical_history_Diabetes', 'medical_history_Heart disease', 'medical_history_High blood pressure',
    'family_medical_history_Diabetes', 'family_medical_history_Heart disease',
    'family_medical_history_High blood pressure', 'exercise_frequency_Never',
    'exercise_frequency_Occasionally', 'exercise_frequency_Rarely',
    'occupation_Blue collar', 'occupation_Student', 'occupation_Unemployed',
    'occupation_White collar', 'coverage_level_Premium', 'coverage_level_Standard',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        input_data = request.get_json()

        # Ensure all expected features are present in the input data
        for feature in expected_features:
            if feature not in input_data:
                input_data[feature] = 0  # Add missing feature with default value 0

        # Reorder the input data to match the feature order of the model
        input_data = {feature: input_data.get(feature, 0) for feature in expected_features}

        # Convert the input data to a DataFrame (needed for prediction)
        input_df = pd.DataFrame([input_data])

        # Make the prediction
        prediction = model.predict(input_df)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Handle any errors and return a JSON error response
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
