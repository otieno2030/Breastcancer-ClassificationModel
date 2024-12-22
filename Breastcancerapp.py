from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
# Load the new model in your Flask app
model = joblib.load('breast_cancer_model_v2.joblib')

# Initialize the required preprocessing steps
scaler = joblib.load('scaler.joblib')  # Load the pre-fitted scaler
imputer = joblib.load('imputer.joblib')  # Load the pre-fitted imputer



@app.route('/', methods=['GET'])
def index():
    return render_template('Breastcancer.Html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.headers)
    print(request.form)
    # Get form data from request
    meanRadius = request.form.get('meanRadius')
    meanTexture = request.form.get('meanTexture')
    meanPerimeter = request.form.get('meanPerimeter')
    meanArea = request.form.get('meanArea')
    meanSmoothness = request.form.get('meanSmoothness')

    # Convert input data to array
    input_data = np.array([meanRadius, meanTexture, meanPerimeter, meanArea, meanSmoothness]).reshape(1, -1)

    # Preprocess the input (impute, scale)
    input_data_imputed = imputer.fit_transform(input_data)
    input_data_scaled = scaler.transform(input_data_imputed)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return prediction as JSON
    return jsonify({'prediction': 'Malignant' if prediction[0] == 1 else 'Benign'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404
#the JavaScript code, creates a new FormData object and passes it to the fetch() function. This sends the form data in the correct format to the Flask app.
if __name__ == '__main__':
    app.run(debug=True)