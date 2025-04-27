import numpy as np
import pandas as pd
from .model_loader import ml_components


def prepare_input_data(form_data):
    """Convert form data to model input format."""
    try:
        # Ensure the model feature list exists
        if 'features' not in ml_components:
            raise ValueError("Feature list is missing from ml_components.")

        # Create a DataFrame with all features initialized to 0
        input_data = {feature: 0 for feature in ml_components['features']}

        # Field mapping from form input to model feature names
        field_mapping = {
            'age': 'age',
            'gender': 'gender',
            'education_level': 'education_level',
            'employment_status': 'employment_status',
            'mobile_money_registered': 'mobile_money_registered',
            'bank_account_current': 'bank_account_current',
            'bank_account_savings': 'bank_account_savings',
            'mobile_banking_registered': 'mobile_banking_registered',
            'insurance_nhif': 'insurance_nhif',
            'pension_nssf': 'pension_nssf',
            'loan_bank': 'loan_bank',
            'loan_family_friend': 'loan_family_friend',
            'loan_mobile_banking': 'loan_mobile_banking',
            'loan_sacco': 'loan_sacco'
        }

        # Update input_data with form values
        for form_field, model_feature in field_mapping.items():
            if form_field in form_data:
                try:
                    input_data[model_feature] = int(form_data[form_field])
                except ValueError:
                    raise ValueError(f"Invalid input for {
                                     form_field}: expected an integer.")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure feature order matches the trained model
        df = df[ml_components['features']]

        print("Input Data Prepared Successfully", df.shape)
        return df

    except Exception as e:
        print(f"Error in prepare_input_data: {str(e)}")
        raise


def predict_exclusion(input_data, model_type='voting'):
    """Make prediction using the selected model."""
    try:
        # Prepare input data
        df = prepare_input_data(input_data)
        print("Input DataFrame Shape:", df.shape)

        # Ensure feature selector exists
        if 'selector' not in ml_components:
            raise ValueError("Feature selector is missing from ml_components.")

        # Perform feature selection
        selected_indices = ml_components['selector'].get_support()
        print("Selected Features Count:", sum(selected_indices))

        X = ml_components['selector'].transform(df)
        print("Transformed Data Shape:", X.shape)

        # Check for feature selection mismatch
        if X.shape[1] != sum(selected_indices):
            raise ValueError(f"Feature selection mismatch: Expected {
                             sum(selected_indices)} features, got {X.shape[1]}")

        # Get the appropriate model
        model = ml_components.get(
            'voting' if model_type == 'voting' else 'decision_tree')
        if model is None:
            raise ValueError(
                f"Model '{model_type}' not found in ml_components.")

        # Make prediction
        probability = model.predict_proba(X)[0][1]
        prediction = probability >= 0.5  # Adjust threshold if needed

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            importances = model.estimators_[0].feature_importances_
        else:
            # Default to zeros if unavailable
            importances = np.zeros_like(selected_indices, dtype=float)

        # Extract feature importance details
        features = ml_components['features']
        selected_features = [f for f, selected in zip(
            features, selected_indices) if selected]
        feature_importances = sorted(
            zip(selected_features, importances * 100), key=lambda x: x[1], reverse=True)[:5]

        return {
            'prediction': bool(prediction),
            'probability': round(float(probability), 4),
            'model_used': 'Voting Ensemble' if model_type == 'voting' else 'Decision Tree',
            'top_factors': [(f, round(imp, 2)) for f, imp in feature_importances]
        }

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise
