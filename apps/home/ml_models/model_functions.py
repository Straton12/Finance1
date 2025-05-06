import joblib
import os
import numpy as np
import pandas as pd
from django.conf import settings
from lime.lime_tabular import LimeTabularExplainer
from .feature_processor import prepare_features as process_features

# Get absolute path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_CONFIG = {
    'Decision Tree (Demographics + Behavior, SMOTE)': {
        'filename': 'decision_tree_model.pkl',
        'required_features': ['age', 'gender', 'education_level', 'residence_type', 'region']
    },
    'Logistic Regression (Demographics + Behavior, Weighted)': {
        'filename': 'voting_ensemble_model.pkl',
        'required_features': ['age', 'gender', 'education_level', 'residence_type', 'region']
    },
    'Gradient Boosting (Demographics Only, SMOTE)': {
        'filename': 'decision_tree_model.pkl',
        'required_features': ['age', 'gender', 'education_level', 'residence_type']
    }
}

def load_model(model_choice):
    """Load the appropriate model based on user selection"""
    try:
        print(f"Loading model for choice: {model_choice}")
        
        if model_choice not in MODEL_CONFIG:
            raise ValueError(f"Invalid model choice: {model_choice}. Available models: {list(MODEL_CONFIG.keys())}")
        
        model_info = MODEL_CONFIG[model_choice]
        model_path = os.path.join(BASE_DIR, model_info['filename'])
        
        print(f"Looking for model at path: {model_path}")
        
        if not os.path.exists(model_path):
            # Try alternative path using settings.CORE_DIR
            alt_path = os.path.join(settings.CORE_DIR, 'apps', 'home', 'ml_models', model_info['filename'])
            print(f"Model not found at primary path, trying alternative path: {alt_path}")
            
            if not os.path.exists(alt_path):
                raise FileNotFoundError(
                    f"Model file not found at either:\n"
                    f"Primary path: {model_path}\n"
                    f"Alternative path: {alt_path}"
                )
            model_path = alt_path
            
        model = joblib.load(model_path)
        print(f"Successfully loaded model from: {model_path}")
        return model
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)

def prepare_features(input_data, model_choice):
    """Prepare features based on the selected model"""
    try:
        print(f"Preparing features for model: {model_choice}")
        print(f"Input data: {input_data}")
        
        if model_choice not in MODEL_CONFIG:
            raise ValueError(f"Invalid model choice: {model_choice}")
            
        # Validate required features
        required_features = MODEL_CONFIG[model_choice]['required_features']
        missing_features = [f for f in required_features if not input_data.get(f)]
        
        if missing_features:
            raise ValueError(
                f"Missing required features for {model_choice}: "
                f"{', '.join(missing_features)}"
            )
        
        # Process features using our feature processor
        df_encoded = process_features(input_data)
        
        print(f"Features processed successfully:")
        print(f"- Shape: {df_encoded.shape}")
        print(f"- Columns: {df_encoded.columns.tolist()}")
        
        return df_encoded
        
    except Exception as e:
        error_msg = f"Error preparing features: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)

def make_prediction(input_data, model_choice):
    """Make prediction using the selected model"""
    try:
        print(f"\n=== Starting prediction process ===")
        print(f"Model choice: {model_choice}")
        print(f"Input data: {input_data}")
        
        if not model_choice:
            raise ValueError("No model selected")
            
        if model_choice not in MODEL_CONFIG:
            raise ValueError(f"Invalid model choice: {model_choice}")
        
        # Load model
        try:
            model = load_model(model_choice)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # Prepare features
        try:
            X = prepare_features(input_data, model_choice)
        except Exception as e:
            raise ValueError(f"Failed to prepare features: {str(e)}")
        
        # Make prediction
        try:
            print("\nMaking prediction...")
            probability = float(model.predict_proba(X)[0][1])
            print(f"Raw probability: {probability}")
        except Exception as e:
            raise ValueError(f"Failed to generate prediction: {str(e)}")
        
        # Scale probability
        try:
            print("\nScaling probability...")
            if probability > 0.5:
                scaled_probability = 0.5 + (probability - 0.5) * 1.5
            else:
                scaled_probability = 0.5 - (0.5 - probability) * 1.5
                
            scaled_probability = max(0.05, min(0.95, scaled_probability))
            print(f"Scaled probability: {scaled_probability}")
            
            prediction = "Financially Excluded" if scaled_probability > 0.5 else "Financially Included"
        except Exception as e:
            raise ValueError(f"Failed to scale probability: {str(e)}")

        # Generate feature importance explanation
        try:
            print("\nGenerating feature importance explanation...")
            lime_background = np.tile(X.values, (100, 1))
            
            explainer = LimeTabularExplainer(
                training_data=lime_background,
                mode="classification",
                feature_names=X.columns.tolist(),
                class_names=['Included', 'Excluded'],
                discretize_continuous=True,
                kernel_width=3
            )
            
            exp = explainer.explain_instance(
                X.values[0],
                model.predict_proba,
                num_features=5,
                num_samples=1000
            )
            
            feature_importances = []
            for feature, weight in exp.as_list():
                display_name = feature.replace('_', ' ').title()
                if 'Hh' in display_name:
                    display_name = display_name.replace('Hh', 'Household')
                    
                direction = "positive" if weight > 0 else "negative"
                impact = abs(weight) / 5
                
                feature_importances.append({
                    "feature": display_name,
                    "direction": direction,
                    "impact": min(impact, 0.5)
                })
                
            print(f"Generated {len(feature_importances)} feature importance factors")
                
        except Exception as e:
            print(f"Warning: Failed to generate explanations: {str(e)}")
            feature_importances = []

        result = {
            "status": "success",
            "prediction": prediction,
            "probability": scaled_probability,
            "factors": feature_importances
        }
        
        print("\n=== Prediction completed successfully ===")
        print(f"Final result: {result}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n=== Error in prediction process ===")
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        } 