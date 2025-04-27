import joblib
import os
import numpy as np
import pandas as pd

# Get absolute path to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model paths - using the actual model files
model_paths = {
    'Decision Tree (Demographics + Behavior, SMOTE)': 'decision_tree_model.pkl',
    'Logistic Regression (Demographics + Behavior, Weighted)': 'voting_ensemble_model.pkl',  # Using ensemble model
    'Gradient Boosting (Demographics Only, SMOTE)': 'decision_tree_model.pkl'  # Using decision tree as fallback
}

def load_model(model_choice):
    """Load the selected model"""
    try:
        model_file = model_paths.get(model_choice)
        if not model_file:
            raise ValueError(f"Invalid model choice: {model_choice}")
        
        model_path = os.path.join(current_dir, model_file)
        print(f"Looking for model in: {model_path}")  # Debug print
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")  # Debug print
        raise

def prepare_features(input_data, model_choice):
    """Prepare features based on the selected model"""
    # Create DataFrame with input data
    df = pd.DataFrame([input_data])
    
    # Define categorical columns and their expected values
    categorical_mapping = {
        'gender': ['female', 'male'],
        'education_level': ['no_formal_education', 'primary', 'secondary', 'university', 'vocational'],
        'residence_type': ['rural', 'urban', 'peri_urban'],
        'marital_status': ['divorced', 'married', 'separated', 'single', 'widowed'],
        'relationship_to_hh': ['head', 'spouse', 'son_daughter', 'parent', 'other_relative'],
        'region': ['central', 'coast', 'eastern', 'nairobi', 'north_eastern', 'nyanza', 'rift_valley', 'western']
    }
    
    # Standardize categorical values
    for col, values in categorical_mapping.items():
        if col in df.columns:
            if df[col].iloc[0] not in values:
                df[col] = values[0]  # Use first value as default
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=list(categorical_mapping.keys()))
    
    # Ensure all expected dummy columns exist
    for col, values in categorical_mapping.items():
        for value in values:
            dummy_col = f'{col}_{value}'
            if dummy_col not in df_encoded.columns:
                df_encoded[dummy_col] = 0
    
    # Add age column if not present
    if 'age' not in df_encoded.columns:
        df_encoded['age'] = input_data.get('age', 35)
    
    # Define behavioral columns
    behavioral_columns = [
        'mobile_money', 'bank_account', 'savings_account',
        'loan', 'insurance', 'pension', 'has_debit_card',
        'has_credit_card', 'savings_microfinance', 'savings_sacco',
        'savings_group'
    ]
    
    # Handle behavioral features based on model choice
    if "Demographics Only" not in model_choice:
        for col in behavioral_columns:
            df_encoded[col] = 1 if input_data.get(col, False) else 0
    else:
        for col in behavioral_columns:
            df_encoded[col] = 0
    
    # Add employment status columns
    employment_columns = [
        'employment_status_employed',
        'employment_status_self_employed',
        'employment_status_unemployed',
        'employment_status_student',
        'employment_status_retired',
        'employment_status_homemaker',
        'employment_status_other',
        'employment_status_not_specified'
    ]
    
    for col in employment_columns:
        df_encoded[col] = 0
    
    # Create list of expected columns in correct order
    expected_columns = ['age']
    
    # Add categorical columns in specific order
    for col, values in categorical_mapping.items():
        for value in values:
            expected_columns.append(f'{col}_{value}')
    
    # Add behavioral columns
    expected_columns.extend(behavioral_columns)
    
    # Add employment status columns
    expected_columns.extend(employment_columns)
    
    # Reindex to ensure exact column order
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)
    
    return df_encoded

def make_prediction(input_data, model_choice):
    """Make prediction using the selected model"""
    try:
        # Load model
        model = load_model(model_choice)
        
        # Prepare features
        X = prepare_features(input_data, model_choice)
        
        # Make prediction
        probability = float(model.predict_proba(X)[0][1])  # Convert to float for stability
        prediction = "Financially Excluded" if probability > 0.5 else "Financially Included"
        
        # Get feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = model.coef_[0]
        
        # Prepare factors
        factors = []
        if feature_importance is not None and len(feature_importance) == X.shape[1]:
            feature_impacts = list(zip(X.columns, feature_importance))
            sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
            
            for feature, impact in sorted_impacts[:5]:
                # Format the feature name
                formatted_feature = feature.replace('_', ' ').title()
                
                # Calculate impact percentage (multiply by 100 for percentage)
                impact_percentage = abs(float(impact)) * 100
                
                # Determine direction and color
                increases_exclusion = float(impact) > 0
                
                factors.append({
                    "feature": formatted_feature,
                    "direction": "increase" if increases_exclusion else "decrease",
                    "weight": impact_percentage,  # Already a percentage
                    "arrow": "↑" if increases_exclusion else "↓",
                    "color": "#D32F2F" if increases_exclusion else "#388E3C"
                })
        
        return {
            "prediction": prediction,
            "probability": probability * 100,  # Convert to percentage
            "factors": factors
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise 