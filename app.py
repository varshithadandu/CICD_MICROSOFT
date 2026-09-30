# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Extract and validate the features (expecting a list of 4 values)
        feature_values = data["features"]
        if len(feature_values) != 4:
            return jsonify({"error": "Expected exactly 4 features"}), 400
        features = np.array(feature_values).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features)[0]
        
        # Map the numerical prediction to the class name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        # Return the prediction as JSON
        return jsonify(result)
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
