import pickle
import os
import numpy as np
import pandas as pd
from django.conf import settings

# Update paths to be more specific
MODEL_PATH = os.path.join(settings.BASE_DIR, 'apps', 'home', 'best_financial_exclusion_model.pkl')
ENCODER_PATH = os.path.join(settings.BASE_DIR, 'apps', 'home', 'onehot_encoder.pkl')
FEATURES_PATH = os.path.join(settings.BASE_DIR, 'apps', 'home', 'columns_full_features.pkl')
# Initialize variables
model = None
encoder = None
feature_names = None

try:
    # Load model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Load OneHot Encoder
    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    # Load feature names
    with open(FEATURES_PATH, "rb") as file:
        feature_names = pickle.load(file)

except FileNotFoundError as e:
    print(f"Model loading error: {str(e)}")
    # Handle missing files appropriately


def predict_model(inputs):
    """Function to preprocess inputs and return model predictions"""
    if not all([model, encoder, feature_names]):
        return {"error": "Model not properly initialized", "status": "error"}

    try:
        # Convert numeric inputs from string to appropriate types
        processed_inputs = {}
        for key, value in inputs.items():
            if key in ['age', 'income']:  # Numeric fields
                processed_inputs[key] = float(value) if value else 0.0
            else:  # Categorical fields
                processed_inputs[key] = value

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([processed_inputs], columns=feature_names)

        # Debug print to verify data
        print("Input DataFrame before encoding:")
        print(input_df)

        # OneHot Encoding
        if encoder:
            encoded_data = encoder.transform(input_df)
            print("Data after encoding:")
            print(encoded_data)
        else:
            encoded_data = input_df

        # Make prediction
        prediction = model.predict(encoded_data)
        print(f"Raw prediction: {prediction}")

        return {
            "prediction": prediction.tolist(),
            "status": "success",
            "input_data": processed_inputs  # For debugging
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback to console
        return {
            "error": f"Prediction error: {str(e)}",
            "status": "error",
            "traceback": traceback.format_exc()
        }
