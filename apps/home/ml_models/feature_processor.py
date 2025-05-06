import pandas as pd
import numpy as np

def calculate_financial_scores(data):
    """Calculate various financial scores from input data"""
    scores = {}
    
    # Basic financial scores with weighted components
    scores['formal_savings_score'] = (
        2 * int(data.get('bank_account', False)) +  # Bank account has higher weight
        int(data.get('savings_account', False)) +
        int(data.get('has_debit_card', False)) +
        2 * int(data.get('has_credit_card', False))  # Credit card indicates higher financial inclusion
    )
    
    scores['informal_savings_score'] = (
        2 * int(data.get('savings_group', False)) +  # Group savings are significant
        int(data.get('savings_microfinance', False)) +
        int(data.get('savings_sacco', False))
    )
    
    scores['digital_financial_score'] = (
        3 * int(data.get('mobile_money', False)) +  # Mobile money is very important in Kenya
        int(data.get('has_debit_card', False)) +
        int(data.get('has_credit_card', False))
    )
    
    # Credit scores with risk weighting
    has_bank_loan = int(data.get('loan', False))
    has_informal_loan = (
        int(data.get('savings_microfinance', False)) or 
        int(data.get('savings_sacco', False))
    )
    scores['formal_credit_score'] = 2 * has_bank_loan  # Formal loans weighted higher
    scores['informal_credit_score'] = has_informal_loan
    
    # Composite scores with adjusted weights
    scores['formal_financial_score'] = (
        2 * scores['formal_savings_score'] +  # Savings weighted higher than credit
        scores['formal_credit_score']
    )
    
    scores['informal_financial_score'] = (
        scores['informal_savings_score'] +
        scores['informal_credit_score']
    )
    
    scores['financial_engagement_score'] = (
        2.0 * scores['formal_financial_score'] +    # Formal services weighted highest
        1.0 * scores['informal_financial_score'] +  # Informal services still count
        1.5 * scores['digital_financial_score']     # Digital services are important
    )
    
    # Product diversity with more granular scoring
    financial_products = [
        scores['formal_savings_score'] > 0,
        scores['informal_savings_score'] > 0,
        scores['formal_credit_score'] > 0,
        scores['informal_credit_score'] > 0,
        scores['digital_financial_score'] > 0,
        int(data.get('insurance', False)),
        int(data.get('pension', False))
    ]
    scores['financial_product_diversity'] = sum(financial_products)
    scores['product_category_diversity'] = len(set(financial_products))
    
    # Risk management score with insurance types
    scores['risk_management_score'] = (
        2 * int(data.get('insurance', False)) +  # Insurance weighted higher
        int(data.get('pension', False))
    )
    
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
    
    # Map boolean fields
    boolean_fields = [
        'mobile_money', 'bank_account', 'savings_account', 'loan',
        'insurance', 'pension', 'has_debit_card', 'has_credit_card',
        'savings_microfinance', 'savings_sacco', 'savings_group'
    ]
    
    for field in boolean_fields:
        data[field] = int(data.get(field, False))
    
    # Map mobile money and debit card to specific fields
    data['mobile_money_registered'] = data['mobile_money']
    data['debit_card'] = data['has_debit_card']
    
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
            'head': 'head',
            'spouse': 'spouse',
            'other_relative': 'other relative',
            'son_daughter': 'son/daughter',
            'parent': 'parent',
            'not_related': 'not related'
        },
        'region': {
            'nairobi': 'nairobi',
            'central': 'central',
            'coast': 'coast',
            'eastern': 'mid eastern',
            'north_eastern': 'north eastern',
            'nyanza': 'nyanza',
            'rift_valley': 'central rift',
            'western': 'western'
        },
        'education_level': {
            'no_formal_education': 'no formal education',
            'primary': 'primary',
            'secondary': 'secondary',
            'university': 'university'
        },
        'gender': {
            'male': 'male',
            'female': 'female'
        },
        'residence_type': {
            'urban': 'urban',
            'rural': 'rural'
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
    
    # Define all expected columns
    expected_columns = [
        'age', 'mobile_money_registered', 'debit_card',
        'formal_savings_score', 'informal_savings_score', 'digital_financial_score',
        'formal_credit_score', 'informal_credit_score', 'financial_product_diversity',
        'formal_financial_score', 'informal_financial_score', 'financial_engagement_score',
        'product_category_diversity', 'risk_management_score',
        
        # Gender
        'gender_male', 'gender_female',
        
        # Education
        'education_level_no_formal_education', 'education_level_primary',
        'education_level_secondary', 'education_level_university',
        
        # Residence
        'residence_type_urban', 'residence_type_rural',
        
        # Marital Status
        'marital_status_divorced/separated', 'marital_status_married/living_with_partner',
        'marital_status_single', 'marital_status_widowed',
        
        # Relationship
        'relationship_to_hh_head', 'relationship_to_hh_spouse',
        'relationship_to_hh_other_relative', 'relationship_to_hh_son/daughter',
        'relationship_to_hh_parent', 'relationship_to_hh_not_related',
        
        # Region
        'region_nairobi', 'region_central', 'region_coast', 'region_mid_eastern',
        'region_north_eastern', 'region_nyanza', 'region_central_rift', 'region_western',
        
        # Ratios
        'formal_informal_ratio_Formal_Only', 'formal_informal_ratio_Informal_Only',
        'formal_informal_ratio_Mixed', 'formal_informal_ratio_None',
        'credit_to_savings_ratio_Credit_Only', 'credit_to_savings_ratio_Mixed',
        'credit_to_savings_ratio_None', 'credit_to_savings_ratio_Savings_Only'
    ]
    
    # Add missing columns with 0s
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Select only expected columns in correct order
    result = df_encoded[expected_columns]
    
    # Print shape information for debugging
    print(f"Final feature shape: {result.shape}")
    print(f"Features included: {result.columns.tolist()}")
    
    return result 