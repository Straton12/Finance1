
# home/forms.py
from django import forms
import joblib


class PredictionForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        features = joblib.load('ml_model/selected_features.pkl')
        for feature in features:
            self.fields[feature] = forms.ChoiceField(
                choices=[('1', 'Yes'), ('0', 'No')],
                widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
                label=feature.replace('_', ' ').title(),
                required=True
            )


class FinancialExclusionForm(forms.Form):
    age = forms.IntegerField(min_value=18, max_value=100)
    gender = forms.ChoiceField(
        choices=[('Male', 'Male'), ('Female', 'Female')])
    mobile_money_registered = forms.ChoiceField(
        choices=[(1, 'Yes'), (0, 'No')])
    bank_account_current = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    # Add all other fields from your form

    model_type = forms.ChoiceField(
        choices=[('voting', 'Voting Ensemble Model'),
                 ('decision_tree', 'Decision Tree Model')],
        initial='voting'
    )


# home/forms.py


class FinancialExclusionForm(forms.Form):
    AGE_CHOICES = [(i, str(i)) for i in range(18, 101)]
    GENDER_CHOICES = [('Male', 'Male'), ('Female', 'Female')]
    EDUCATION_CHOICES = [
        ('None', 'None'),
        ('Primary', 'Primary'),
        ('Secondary', 'Secondary'),
        ('Tertiary', 'Tertiary')
    ]
    RESIDENCE_CHOICES = [('Urban', 'Urban'), ('Rural', 'Rural')]
    MARITAL_CHOICES = [
        ('Single', 'Single'),
        ('Married', 'Married'),
        ('Divorced/Separated', 'Divorced/Separated'),
        ('Widowed', 'Widowed')
    ]
    RELATIONSHIP_CHOICES = [
        ('Head', 'Head'),
        ('Spouse', 'Spouse'),
        ('Son/Daughter', 'Son/Daughter'),
        ('Other relative', 'Other relative')
    ]
    REGION_CHOICES = [
        ('Nairobi', 'Nairobi'),
        ('Central', 'Central'),
        ('Coast', 'Coast'),
        ('Eastern', 'Eastern'),
        ('North Eastern', 'North Eastern'),
        ('Nyanza', 'Nyanza'),
        ('Rift Valley', 'Rift Valley'),
        ('Western', 'Western'),
        ('South Rift', 'South Rift')
    ]
    YES_NO_CHOICES = [('Yes', 'Yes'), ('No', 'No')]

    # Basic demographics
    age = forms.ChoiceField(choices=AGE_CHOICES)
    population_weight = forms.FloatField(
        min_value=0.1, max_value=3.0, initial=1.0)
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    education_level = forms.ChoiceField(choices=EDUCATION_CHOICES)
    residence_type = forms.ChoiceField(choices=RESIDENCE_CHOICES)
    marital_status = forms.ChoiceField(choices=MARITAL_CHOICES)
    relationship_to_hh = forms.ChoiceField(choices=RELATIONSHIP_CHOICES)
    region = forms.ChoiceField(choices=REGION_CHOICES)

    # Financial services
    mobile_money_registered = forms.ChoiceField(choices=YES_NO_CHOICES)
    bank_account_everyday = forms.ChoiceField(choices=YES_NO_CHOICES)
    savings_mobile_banking = forms.ChoiceField(choices=YES_NO_CHOICES)
    savings_secret_place = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_mobile_banking = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_sacco = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_group_chama = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_family_friend = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_goods_credit = forms.ChoiceField(choices=YES_NO_CHOICES)
    loan_hire_purchase = forms.ChoiceField(choices=YES_NO_CHOICES)
    insurance_nhif = forms.ChoiceField(choices=YES_NO_CHOICES)
    debit_card = forms.ChoiceField(choices=YES_NO_CHOICES)
    pension_nssf = forms.ChoiceField(choices=YES_NO_CHOICES)


#########################
################
##############


class PredictionForm(forms.Form):
    MODEL_CHOICES = [
        ('decision_tree', 'Decision Tree (Demographics + Behavior, SMOTE)'),
        ('logistic_regression', 'Logistic Regression (Demographics + Behavior, Weighted)'),
        ('gradient_boosting', 'Gradient Boosting (Demographics Only, SMOTE)'),
    ]

    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]

    EDUCATION_LEVELS = [
        ('no_formal_education', 'No Formal Education'),
        ('primary', 'Primary'),
        ('secondary', 'Secondary'),
        ('university', 'University'),
    ]

    RESIDENCE_TYPES = [
        ('urban', 'Urban'),
        ('rural', 'Rural'),
    ]

    MARITAL_STATUSES = [
        ('single', 'Single'),
        ('married', 'Married'),
        ('divorced', 'Divorced'),
        ('widowed', 'Widowed'),
        ('separated', 'Separated'),
    ]

    RELATIONSHIPS = [
        ('head', 'Head'),
        ('spouse', 'Spouse'),
        ('son_daughter', 'Son/Daughter'),
        ('parent', 'Parent'),
        ('other_relative', 'Other Relative'),
        ('not_related', 'Not Related'),
    ]

    REGIONS = [
        ('nairobi', 'Nairobi'),
        ('central', 'Central'),
        ('coast', 'Coast'),
        ('eastern', 'Eastern'),
        ('north_eastern', 'North Eastern'),
        ('nyanza', 'Nyanza'),
        ('rift_valley', 'Rift Valley'),
        ('western', 'Western'),
    ]

    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Choose prediction model:",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    age = forms.IntegerField(
        label="Age",
        min_value=18,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter age (18-100)'
        })
    )

    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    education_level = forms.ChoiceField(
        choices=EDUCATION_LEVELS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    residence_type = forms.ChoiceField(
        choices=RESIDENCE_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    marital_status = forms.ChoiceField(
        choices=MARITAL_STATUSES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    relationship_to_hh = forms.ChoiceField(
        choices=RELATIONSHIPS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    region = forms.ChoiceField(
        choices=REGIONS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    mobile_money = forms.BooleanField(
        label="Has Mobile Money Account",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    bank_account = forms.BooleanField(
        label="Has Bank Account",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    savings_account = forms.BooleanField(
        label="Has Savings Account",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    insurance = forms.BooleanField(
        label="Has Any Insurance",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    pension = forms.BooleanField(
        label="Has Pension",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
