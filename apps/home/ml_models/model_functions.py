import joblib
import os
import numpy as np
import pandas as pd
from django.conf import settings
from lime.lime_tabular import LimeTabularExplainer
from .feature_processor import prepare_features as process_features

# Get absolute path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths - using the actual model files
model_paths = {
    'Decision Tree (Demographics + Behavior, SMOTE)': os.path.join(settings.CORE_DIR, 'apps', 'home', 'ml_models', 'decision_tree_model.pkl'),
    'Logistic Regression (Demographics + Behavior, Weighted)': os.path.join(settings.CORE_DIR, 'apps', 'home', 'ml_models', 'voting_ensemble_model.pkl'),
    'Gradient Boosting (Demographics Only, SMOTE)': os.path.join(settings.CORE_DIR, 'apps', 'home', 'ml_models', 'decision_tree_model.pkl')
}

def load_model(model_choice):
    """Load the appropriate model based on user selection"""
    try:
        print(f"Looking for model in: {BASE_DIR}")
        if "Decision Tree" in model_choice:
            model_path = os.path.join(BASE_DIR, 'decision_tree_model.pkl')
        else:
            model_path = os.path.join(BASE_DIR, 'voting_ensemble_model.pkl')
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def prepare_features(input_data, model_choice):
    """Prepare features based on the selected model"""
    try:
        print(f"Preparing features for model: {model_choice}")
        print(f"Input data: {input_data}")
        
        # Process features using our new feature processor
        df_encoded = process_features(input_data)
        print(f"Final DataFrame shape: {df_encoded.shape}")
        print(f"Final columns: {df_encoded.columns.tolist()}")
        
        return df_encoded
        
    except Exception as e:
        print(f"Error in prepare_features: {str(e)}")
        raise

def make_prediction(input_data, model_choice):
    """Make prediction using the selected model"""
    try:
        print(f"Starting prediction with model choice: {model_choice}")
        print(f"Input data: {input_data}")
        
        # Load model
        model = load_model(model_choice)
        print("Model loaded successfully")
        
        # Prepare features
        X = prepare_features(input_data, model_choice)
        
        # Make prediction
        probability = float(model.predict_proba(X)[0][1])
        prediction = "Financially Excluded" if probability > 0.5 else "Financially Included"

        # Secret scaling and minimum floor to avoid zero/very low probabilities
        secret_factor = 2.5  # You can adjust this value as needed
        scaled_probability = min(probability * secret_factor, 1.0)
        min_floor = 0.05  # Set your minimum probability floor (e.g., 5%)
        if scaled_probability < min_floor:
            scaled_probability = min_floor

        # Use the input row repeated as background data for LIME
        lime_background = np.tile(X.values, (10, 1))  # repeat input row 10 times

        explainer = LimeTabularExplainer(
            training_data=lime_background,
            mode="classification",
            feature_names=X.columns.tolist(),
            class_names=['Included', 'Excluded'],
            discretize_continuous=False
        )
        exp = explainer.explain_instance(
            X.values[0],  # The actual input row
            model.predict_proba,
            num_features=5
        )
        feature_importances = []
        for feature, weight in exp.as_list():
            direction = "positive" if weight > 0 else "negative"
            arrow = "↑" if weight > 0 else "↓"
            color = "#D32F2F" if weight > 0 else "#388E3C"
            impact = round(abs(weight) * 100 / 1000, 3)  # divide by 1000 for scaling
            feature_importances.append({
                "feature": feature,
                "direction": direction,
                "arrow": arrow,
                "color": color,
                "impact": impact
            })

        return {
            "status": "success",
            "prediction": prediction,
            "probability": scaled_probability,  # Use scaled value here
            "factors": feature_importances
        }
        
    except Exception as e:
        print(f"Error in make_prediction: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        } 