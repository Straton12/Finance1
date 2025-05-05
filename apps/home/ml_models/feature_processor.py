import pandas as pd
import numpy as np

def calculate_financial_scores(data):
    """Calculate various financial scores from input data"""
    scores = {}
    
    # Basic financial scores
    scores['formal_savings_score'] = int(data.get('bank_account', False))
    scores['informal_savings_score'] = int(data.get('savings_group', False))
    scores['digital_financial_score'] = int(data.get('mobile_money', False))
    
    # Credit scores
    has_bank_loan = int(data.get('loan', False))
    has_informal_loan = int(data.get('savings_microfinance', False) or data.get('savings_sacco', False))
    scores['formal_credit_score'] = has_bank_loan
    scores['informal_credit_score'] = has_informal_loan
    
    # Composite scores
    scores['formal_financial_score'] = scores['formal_savings_score'] + scores['formal_credit_score']
    scores['informal_financial_score'] = scores['informal_savings_score'] + scores['informal_credit_score']
    scores['financial_engagement_score'] = (
        1.5 * scores['formal_financial_score'] +
        1.0 * scores['informal_financial_score'] +
        2.0 * scores['digital_financial_score']
    )
    
    # Product diversity
    financial_products = [
        scores['formal_savings_score'],
        scores['informal_savings_score'],
        scores['formal_credit_score'],
        scores['informal_credit_score'],
        scores['digital_financial_score']
    ]
    scores['financial_product_diversity'] = sum(1 for score in financial_products if score > 0)
    scores['product_category_diversity'] = len(set(financial_products))
    
    # Risk management score
    scores['risk_management_score'] = int(data.get('insurance', False)) + int(data.get('pension', False))
    
    return scores

def calculate_ratios(data):
    """Calculate formal/informal and credit/savings ratios"""
    scores = calculate_financial_scores(data)
    
    # Formal vs informal ratio
    if scores['informal_financial_score'] == 0:
        if scores['formal_financial_score'] > 0:
            formal_informal = 'Formal_Only'
        else:
            formal_informal = 'None'
    else:
        if scores['formal_financial_score'] == 0:
            formal_informal = 'Informal_Only'
        else:
            formal_informal = 'Mixed'
            
    # Credit to savings ratio
    total_savings = scores['formal_savings_score'] + scores['informal_savings_score']
    total_credit = scores['formal_credit_score'] + scores['informal_credit_score']
    
    if total_savings == 0:
        if total_credit > 0:
            credit_savings = 'Credit_Only'
        else:
            credit_savings = 'None'
    else:
        if total_credit == 0:
            credit_savings = 'Savings_Only'
        else:
            credit_savings = 'Mixed'
            
    return {
        'formal_informal_ratio': formal_informal,
        'credit_to_savings_ratio': credit_savings
    }

def prepare_features(input_data):
    """Transform input data to match model's expected features"""
    # Start with a copy of input data
    data = input_data.copy()
    
    # Add population weight (default to 1 as it's used for training but not relevant for prediction)
    data['population_weight'] = 1
    
    # Map mobile money and debit card
    data['mobile_money_registered'] = int(data.get('mobile_money', False))
    data['debit_card'] = int(data.get('has_debit_card', False))
    
    # Add financial scores
    scores = calculate_financial_scores(data)
    data.update(scores)
    
    # Add ratios
    ratios = calculate_ratios(data)
    data.update(ratios)
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Handle categorical variables with specific encodings
    categorical_mappings = {
        'marital_status': {
            'divorced': 'divorced/separated',
            'separated': 'divorced/separated',
            'married': 'married/living with partner',
            'single': 'single',
            'widowed': 'widowed'
        },
        'relationship_to_hh': {
            'other_relative': 'other relative',
            'son_daughter': 'son/daughter'
        },
        'region': {
            'eastern': 'mid eastern',  # Simplified mapping
            'central': 'central',
            'coast': 'coast',
            'north_eastern': 'north eastern',
            'rift_valley': 'central rift'  # Simplified mapping
        }
    }
    
    # Apply mappings
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    
    # One-hot encode categorical variables
    categorical_cols = ['gender', 'education_level', 'residence_type', 'marital_status', 
                       'relationship_to_hh', 'region', 'formal_informal_ratio', 'credit_to_savings_ratio']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
    
    # Ensure all expected columns exist
    expected_columns = [
        'age', 'population_weight', 'mobile_money_registered', 'savings_mobile_banking',
        'debit_card', 'savings_secret_place', 'loan_goods_credit', 'formal_savings_score',
        'informal_savings_score', 'digital_financial_score', 'formal_credit_score',
        'informal_credit_score', 'financial_product_diversity', 'formal_financial_score',
        'informal_financial_score', 'financial_engagement_score', 'product_category_diversity',
        'risk_management_score', 'gender_female', 'gender_male', 'education_level_primary',
        'education_level_secondary', 'education_level_unknown', 'residence_type_rural',
        'residence_type_urban', 'marital_status_divorced/separated',
        'marital_status_married/living with partner', 'marital_status_single',
        'marital_status_widowed', 'relationship_to_hh_head', 'relationship_to_hh_other relative',
        'relationship_to_hh_parent', 'relationship_to_hh_son/daughter', 'relationship_to_hh_spouse',
        'region_central', 'region_central rift', 'region_coast', 'region_lower eastern',
        'region_mid eastern', 'region_north eastern', 'region_upper eastern',
        'formal_informal_ratio_Formal_Only', 'formal_informal_ratio_Informal_Only',
        'formal_informal_ratio_Mixed', 'formal_informal_ratio_None',
        'credit_to_savings_ratio_Mixed', 'credit_to_savings_ratio_None',
        'credit_to_savings_ratio_Savings_Only'
    ]
    
    # Add missing columns with 0s
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Select only expected columns in correct order
    return df_encoded[expected_columns] 