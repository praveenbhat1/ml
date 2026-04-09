import os
import joblib
import warnings
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load models and preprocessing tools
try:
    model = joblib.load("random_forest_crop_model.pkl")
    encoder = joblib.load("encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    print("Error loading models:", e)

def preprocess_input(input_data):
    features = pd.DataFrame([input_data])
    
    categorical_cols = list(encoder.feature_names_in_)
    numerical_cols = [c for c in features.columns if c not in categorical_cols]
    
    categorical_data = features[categorical_cols]
    numerical_data = features[numerical_cols]

    encoded_data = encoder.transform(categorical_data)
    all_feature_names = list(numerical_data.columns) + list(encoder.get_feature_names_out(categorical_cols))
    
    processed_df = pd.DataFrame(np.hstack([numerical_data.values, encoded_data]), columns=all_feature_names)
    
    # Reorder columns to match training
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
    return processed_df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Convert necessary types
        numeric_fields = ['Crop_Year', 'Annual_Rainfall', 'Pesticide', 'Fertilizer', 'Production', 'Area']
        for field in numeric_fields:
            data[field] = float(data[field])
            
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        
        # 1 usually means High, 0 means Low (based on Yield_Class logic in mltest.py)
        result = "High Yield" if prediction[0] == 1 else "Low Yield"
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
