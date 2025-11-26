import pandas as pd
import joblib
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Define the filename for the exported scaler
scaler_filename = 'scaler.joblib'

# Save the fitted scaler using joblib.dump()
joblib.dump(scaler, scaler_filename)

print(f"Fitted StandardScaler exported to '{scaler_filename}'")

# 1. Initialize the Flask application


# 2. Define the mean values for columns where '0's were replaced during training.
# These values are derived from the 'Prepare Data' subtask.
IMPUTATION_MEANS = {
    'Glucose': 121.68676239102434,
    'BloodPressure': 72.40518463990689,
    'SkinThickness': 29.153419047619042,
    'Insulin': 155.5482233502538,
    'BMI': 32.457463672391015
}

# 3. Load the trained logistic regression model and the fitted StandardScaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the incoming request
    data = request.get_json(force=True)

    # Convert the received JSON data into a pandas DataFrame
    # Ensure column order matches the training data by creating a DataFrame from a list of dicts
    # and then reindexing to the original feature column order
    input_df = pd.DataFrame(data, index=[0])

    # Define the original feature columns in the order they were trained
    # This assumes X_train.columns gives the correct order
    original_feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = input_df[original_feature_columns]

    # Replicate the '0' imputation logic for the specified columns
    for column in IMPUTATION_MEANS.keys():
        if column in input_df.columns:
            input_df[column] = input_df[column].replace(0, IMPUTATION_MEANS[column])

    # Apply the loaded StandardScaler to transform the preprocessed DataFrame
    scaled_data = scaler.transform(input_df)

    # Use the loaded logistic regression model to make predictions
    prediction = model.predict(scaled_data)

    # Convert the prediction result into a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # To run the Flask app, save this code as 'app.py' and execute `python app.py` in your terminal.
    # For deployment, ensure debug=False and use a production-ready WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5000)

print("Flask API script 'app.py' generated. You can save this code to a file and run it to start the API server.")
