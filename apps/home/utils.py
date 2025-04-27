'''from django.conf import settings
import os
import joblib
import numpy as np
import pandas as pd

# Load models once when the module is imported
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'ml_models',
                          'voting_ensemble_model.pkl')
selector_path = os.path.join(current_dir, 'ml_models', 'feature_selector.pkl')
selected_features_path = os.path.join(
    current_dir, 'ml_models', 'selected_features.pkl')

model = joblib.load(model_path)
selector = joblib.load(selector_path)
selected_features = joblib.load(selected_features_path)

# Feature groups
feature_groups = {
    'formal_savings': ['bank_account_everyday'],
    'informal_savings': ['savings_secret_place'],
    'digital_financial': ['mobile_money_registered', 'savings_mobile_banking', 'loan_mobile_banking'],
    'formal_credit': ['loan_sacco'],
    'informal_credit': ['loan_group_chama', 'loan_family_friend', 'loan_goods_credit', 'loan_hire_purchase'],
    'insurance': ['insurance_nhif'],
    'pension': ['pension_nssf']
}


def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Binary encode yes/no
    for col in df.columns:
        if df[col].iloc[0] in ['Yes', 'No']:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Compute scores
    for group, features in feature_groups.items():
        df[f'{group}_score'] = df[features].sum(axis=1)

    df['formal_financial_score'] = df[[f for f in (
        feature_groups['formal_savings'] + feature_groups['formal_credit'] +
        feature_groups['insurance'] + feature_groups['pension']
    ) if f in df.columns]].sum(axis=1)

    df['informal_financial_score'] = df[[f for f in feature_groups['informal_credit'] +
                                         feature_groups['informal_savings'] if f in df.columns]].sum(axis=1)
    df['digital_financial_score'] = df[[
        f for f in feature_groups['digital_financial'] if f in df.columns]].sum(axis=1)

    df['financial_engagement_score'] = (
        1.5 * df['formal_financial_score'] +
        1.0 * df['informal_financial_score'] +
        2.0 * df['digital_financial_score']
    )

    df['product_category_diversity'] = df[[
        f"{g}_score" for g in feature_groups]].gt(0).sum(axis=1)
    df['risk_management_score'] = df[[f for f in feature_groups['insurance'] +
                                      feature_groups['pension'] if f in df.columns]].sum(axis=1)

    # Formal vs informal
    df['formal_informal_ratio'] = np.where(
        df['informal_financial_score'] == 0,
        np.where(df['formal_financial_score'] > 0, 'Formal_Only', 'None'),
        np.where(df['formal_financial_score'] == 0, 'Informal_Only', 'Mixed')
    )

    df['credit_to_savings_ratio'] = np.where(
        (df['formal_savings_score'] + df['informal_savings_score']) == 0,
        np.where((df['formal_credit_score'] +
                 df['informal_credit_score']) > 0, 'Credit_Only', 'None'),
        np.where((df['formal_credit_score'] +
                 df['informal_credit_score']) == 0, 'Savings_Only', 'Mixed')
    )

    # Encode categoricals
    categorical = ['gender', 'education_level', 'residence_type', 'marital_status',
                   'relationship_to_hh', 'region', 'formal_informal_ratio', 'credit_to_savings_ratio']
    df_encoded = pd.get_dummies(df, columns=categorical, drop_first=False)

    # Fill any missing expected features with 0
    for col in selected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[selected_features]
    df_final = selector.transform(df_encoded)
    return df_final


def predict_financial_exclusion(form_data):
    # Convert form data to dict
    user_input = {
        'age': form_data['age'],
        'population_weight': form_data['population_weight'],
        'gender': form_data['gender'],
        'education_level': form_data['education_level'],
        'residence_type': form_data['residence_type'],
        'marital_status': form_data['marital_status'],
        'relationship_to_hh': form_data['relationship_to_hh'],
        'region': form_data['region'],
        'mobile_money_registered': form_data['mobile_money_registered'],
        'bank_account_everyday': form_data['bank_account_everyday'],
        'savings_mobile_banking': form_data['savings_mobile_banking'],
        'savings_secret_place': form_data['savings_secret_place'],
        'loan_mobile_banking': form_data['loan_mobile_banking'],
        'loan_sacco': form_data['loan_sacco'],
        'loan_group_chama': form_data['loan_group_chama'],
        'loan_family_friend': form_data['loan_family_friend'],
        'loan_goods_credit': form_data['loan_goods_credit'],
        'loan_hire_purchase': form_data['loan_hire_purchase'],
        'insurance_nhif': form_data['insurance_nhif'],
        'debit_card': form_data['debit_card'],
        'pension_nssf': form_data['pension_nssf']
    }

    # Preprocess input
    X_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    return prediction, probability
'''