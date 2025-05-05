import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Get the base directory for models
BASE_DIR = Path(__file__).resolve().parent

def load_model(model_choice):
    """Load the selected model from disk"""
    # Map the exact dropdown value to the model file
    model_map = {
        "Decision Tree (Demographics + Behavior, SMOTE)": BASE_DIR / "decision_tree_model.pkl",
        "Logistic Regression (Demographics + Behavior, Weighted)": BASE_DIR / "logistic_regression_model.pkl",
        "Gradient Boosting (Demographics Only, SMOTE)": BASE_DIR / "gradient_boosting_model.pkl"
    }
    model_path = model_map.get(model_choice)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model file not found for: {model_choice}")
    return joblib.load(model_path)

def prepare_input_data(input_data, model_choice):
    """Prepare input data for prediction"""
    # Convert boolean values to Yes/No strings
    bool_columns = ['mobile_money', 'bank_account', 'savings_account', 'loan', 
                   'insurance', 'pension', 'has_debit_card', 'has_credit_card',
                   'savings_microfinance', 'savings_sacco', 'savings_group']
    
    for col in bool_columns:
        if col in input_data:
            input_data[col] = 'Yes' if input_data[col] else 'No'
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding for categorical variables
    categorical_columns = ['gender', 'education_level', 'residence_type', 
                         'marital_status', 'relationship_to_hh', 'region']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Get dummy variables
    df = pd.get_dummies(df)
    
    # Align columns to the exact order used in training
    selected_features = joblib.load(BASE_DIR / 'selected_features.pkl')
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[selected_features]
    
    return df

def calculate_feature_importance(model, input_data, prediction):
    """Calculate feature importance for the prediction"""
    feature_importance = []
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        return []
    
    # Get feature names and their importance scores
    for feature, importance in zip(input_data.columns, importances):
        feature_importance.append({
            'feature': feature.replace('_', ' ').title(),
            'direction': 'positive' if importance > 0 else 'negative',
            'impact': abs(importance)
        })
    
    # Sort by absolute importance and get top factors
    feature_importance.sort(key=lambda x: abs(x['impact']), reverse=True)
    return feature_importance[:5]  # Return top 5 factors

def make_prediction(input_data, model_choice):
    """Make prediction using the selected model"""
    try:
        # Load the model
        model = load_model(model_choice)
        
        # Prepare input data
        X = prepare_input_data(input_data, model_choice)
        
        # Make prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        # Calculate feature importance
        factors = calculate_feature_importance(model, X, prediction)
        
        return {
            'status': 'success',
            'prediction': "Financially Excluded" if prediction == 1 else "Financially Included",
            'probability': prediction_proba[1],  # Probability of being excluded
            'factors': factors
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        } 