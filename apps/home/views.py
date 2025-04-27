# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from lime.lime_tabular import LimeTabularExplainer
from .forms import PredictionForm
from .models import SurveyData2016
from django.db.models import Count
from django.views.decorators.http import require_GET
from rest_framework.decorators import api_view
import os
import joblib
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.http import JsonResponse
from .models import SurveyData2016, SurveyData2021
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from rest_framework.response import Response
from django.conf import settings
from django.db.models import Count, Q
from django.shortcuts import render
from rest_framework import viewsets
from .serializers import SurveyData2016Serializer, SurveyData2021Serializer
from django.db import models
from django.views.decorators.csrf import csrf_exempt
import logging
from .forms import FinancialExclusionForm
from .ml_models.predictor import predict_exclusion
from .ml_models.model_functions import make_prediction
#from .model_loader import predict_model
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Case, When, IntegerField, F
from django.db.models.functions import Coalesce

logger = logging.getLogger(__name__)


@login_required(login_url="/login/")
def index(request):
    return render(request, "home/index.html")


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:
        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


@csrf_exempt
@require_http_methods(["GET"])
def age_distribution(request):
    try:
        age_data = list(SurveyData2016.objects.values_list("age", flat=True))
        print(f"Raw age data count: {len(age_data)}")  # Debug log

        age_data = [age for age in age_data if age is not None]
        print(f"Filtered age data count: {len(age_data)}")  # Debug log

        age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 85, 100]
        age_labels = ["0-17", "18-24", "25-34", "35-44",
                      "45-54", "55-64", "65-74", "75-84", "85+"]

        hist, _ = np.histogram(age_data, bins=age_bins)
        total = len(age_data)

        # Convert counts to percentages
        percentages = [round((count / total) * 100, 2)
                       if total > 0 else 0 for count in hist]
        print(f"Age distribution percentages: {percentages}")  # Debug log

        response_data = {
            "labels": age_labels,
            "data": percentages
        }
        print(f"Age distribution response: {response_data}")  # Debug log
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in age_distribution: {str(e)}")
        print(f"Age distribution error: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def gender_distribution(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total records: {total}")  # Debug log

        if total == 0:
            print("No data available for gender distribution")  # Debug log
            return JsonResponse({'error': 'No data available'}, status=404)

        gender_data = SurveyData2016.objects.exclude(gender__isnull=True).values('gender').annotate(
            count=models.Count('gender')
        ).order_by('gender')

        print(f"Gender data from DB: {list(gender_data)}")  # Debug log

        labels = []
        data = []
        for entry in gender_data:
            labels.append(entry['gender'].capitalize())
            percentage = round((entry['count'] / total) * 100, 2)
            data.append(percentage)

        response_data = {
            'labels': labels,
            'data': data,
        }
        print(f"Gender distribution response: {response_data}")  # Debug log
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in gender_distribution: {str(e)}")
        print(f"Gender distribution error: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=500)



@csrf_exempt
@require_http_methods(["GET"])
def education_distribution(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total records: {total}")  # Debug log
        
        if total == 0:
            print("No data available for education distribution")  # Debug log
            return JsonResponse({'error': 'No data available'}, status=404)

        education_counts = {
            'Primary': SurveyData2016.objects.filter(education_level__icontains='primary').count(),
            'Secondary': SurveyData2016.objects.filter(education_level__icontains='secondary').count(),
            'Tertiary': SurveyData2016.objects.filter(education_level__icontains='tertiary').count(),
            'Other': SurveyData2016.objects.exclude(
                education_level__isnull=True
            ).exclude(
                education_level__icontains='primary'
            ).exclude(
                education_level__icontains='secondary'
            ).exclude(
                education_level__icontains='tertiary'
            ).count(),
            'Unknown': SurveyData2016.objects.filter(education_level__isnull=True).count()
        }
        print(f"Education counts: {education_counts}")  # Debug log

        # Convert to percentages
        education_percentages = {k: round((v / total) * 100, 2) for k, v in education_counts.items()}
        print(f"Education percentages: {education_percentages}")  # Debug log

        response_data = {
            'labels': list(education_percentages.keys()),
            'data': list(education_percentages.values())
        }
        print(f"Education distribution response: {response_data}")  # Debug log
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in education_distribution: {str(e)}")
        print(f"Education distribution error: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def residence_type_distribution(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total records: {total}")  # Debug log
        
        if total == 0:
            print("No data available for residence type distribution")  # Debug log
            return JsonResponse({'error': 'No data available'}, status=404)

        residence_data = SurveyData2016.objects.exclude(residence_type__isnull=True).values('residence_type').annotate(
            count=Count('residence_type')
        ).order_by('residence_type')
        
        print(f"Residence data from DB: {list(residence_data)}")  # Debug log

        labels = []
        data = []
        for entry in residence_data:
            labels.append(entry['residence_type'].capitalize())
            percentage = round((entry['count'] / total) * 100, 2)
            data.append(percentage)

        response_data = {
            'labels': labels,
            'data': data,
        }
        print(f"Residence type distribution response: {response_data}")  # Debug log
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in residence_type_distribution: {str(e)}")
        print(f"Residence type distribution error: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=500)


def age_distribution1(request):
    """Fetch age data and return JSON format for Chart.js."""
    try:
        # Get age data and remove null values
        age_data = list(SurveyData2021.objects.values_list("age", flat=True))
        age_data = [age for age in age_data if age is not None]
        
        if not age_data:
            return JsonResponse({'error': 'No age data available'}, status=404)

        # Define age bins and labels
        age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 85, 100]
        age_labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]

        # Calculate histogram
        hist, _ = np.histogram(age_data, bins=age_bins)
        total = len(age_data)
        
        # Convert to percentages
        percentages = [round((count / total) * 100, 2) if total > 0 else 0 for count in hist]

        # Add debug logging
        print(f"Age Distribution 2021 - Total records: {total}")
        print(f"Age Distribution 2021 - Percentages: {percentages}")

        return JsonResponse({
            "labels": age_labels,
            "data": percentages,
            "colors": [
                'rgba(99, 102, 241, 0.7)',
                'rgba(129, 140, 248, 0.7)',
                'rgba(165, 180, 252, 0.7)',
                'rgba(199, 210, 254, 0.7)',
                'rgba(224, 231, 255, 0.7)',
                'rgba(238, 242, 255, 0.7)',
                'rgba(244, 244, 255, 0.7)',
                'rgba(249, 250, 251, 0.7)',
                'rgba(243, 244, 246, 0.7)'
            ]
        })
    except Exception as e:
        logger.error(f"Error in age_distribution1: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def gender_distribution1(request):
    gender_data = SurveyData2021.objects.exclude(gender__isnull=True).values('gender').annotate(
        count=models.Count('gender')
    ).order_by('gender')

    labels = []
    data = []
    for entry in gender_data:
        labels.append(entry['gender'].capitalize())
        data.append(entry['count'])

    return JsonResponse({
        'labels': labels,
        'data': data,
    })
    

def survey_data_view(request):
    # Fetch all data from the SurveyData2016 and SurveyData2021 models
    survey_data_2016 = SurveyData2016.objects.all()
    survey_data_2021 = SurveyData2021.objects.all()
    
    # Pass the data to the template
    context = {
        'survey_data_2016': survey_data_2016,
        'survey_data_2021': survey_data_2021,
    }
    
    return render(request, 'home/ui-tables.html', context)




def education_distribution1(request):
    survey_data = SurveyData2021.objects.all()
    education_counts = {
        'primary': 0,
        'secondary': 0,
        'tertiary': 0,
        'other': 0,
        'unknown': 0
    }

    for respondent in survey_data:
        if respondent.education_level:
            level = respondent.education_level.lower()
            if 'primary' in level:
                education_counts['primary'] += 1
            elif 'secondary' in level:
                education_counts['secondary'] += 1
            elif 'tertiary' in level:
                education_counts['tertiary'] += 1
            else:
                education_counts['other'] += 1
        else:
            education_counts['unknown'] += 1

    return JsonResponse({
        'labels': ['Primary', 'Secondary', 'Tertiary', 'Other', 'Unknown'],
        'data': list(education_counts.values())
    })


logger = logging.getLogger(__name__)


# Set up logger
logger = logging.getLogger(__name__)


def education_distribution1(request):
    survey_data = SurveyData2021.objects.all()
    education_counts = {
        'primary': 0,
        'secondary': 0,
        'tertiary': 0,
        'other': 0,
        'unknown': 0
    }

    for respondent in survey_data:
        if respondent.education_level:
            level = respondent.education_level.lower()
            if 'primary' in level:
                education_counts['primary'] += 1
            elif 'secondary' in level:
                education_counts['secondary'] += 1
            elif 'tertiary' in level:
                education_counts['tertiary'] += 1
            else:
                education_counts['other'] += 1
        else:
            education_counts['unknown'] += 1

    return JsonResponse({
        'labels': ['Primary', 'Secondary', 'Tertiary', 'Other', 'Unknown'],
        'data': list(education_counts.values())
    })


logger = logging.getLogger(__name__)


# Set up logger
logger = logging.getLogger(__name__)


@csrf_exempt
@require_GET
def pension_and_exclusion_stats(request):
    """Fetch pension and financial exclusion stats and return them in JSON format for Chart.js."""
    try:
        # Get all survey data
        survey_data = SurveyData2016.objects.all()

        # Function to count unique values in a field
        def get_value_counts(field_name):
            return (survey_data
                    .values(field_name)
                    .annotate(count=Count(field_name))
                    .order_by('-count'))

        # Count for each pension type
        pension_nssf = get_value_counts('pension_nssf')
        pension_mbao = get_value_counts('pension_mbao')
        pension_other = get_value_counts('pension_other')

        # Format as {label: count}
        def format_counts(queryset, field_name):
            return {
                item[field_name] if item[field_name] else "Unknown": item['count']
                for item in queryset
            }

        # Prepare the pension statistics data
        pension_stats = {
            'NSSF': format_counts(pension_nssf, 'pension_nssf'),
            'MBAO': format_counts(pension_mbao, 'pension_mbao'),
            'Other': format_counts(pension_other, 'pension_other'),
        }

        # Financial exclusion statistics
        excluded = survey_data.filter(
            financially_excluded__iexact='Yes').count()
        total = survey_data.count()
        included = total - excluded
        exclusion_percentage = round(
            (excluded / total) * 100, 2) if total else 0

        # Return JSON response
        return JsonResponse({
            'pension': {
                pension_type: {
                    'labels': list(data.keys()),
                    'data': list(data.values())
                }
                for pension_type, data in pension_stats.items()
            },
            'exclusion': {
                'labels': ['Excluded', 'Included'],
                'data': [excluded, included],
                'percentage': exclusion_percentage
            }
        })

    except Exception as e:
        # Log error for debugging
        logger.error(f"Error fetching pension and exclusion stats: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)
    
    
@csrf_exempt
@require_GET
def pension_and_exclusion_stats1(request):
    """Fetch pension and financial exclusion stats and return raw counts."""
    try:
        # Get raw counts for each pension type
        nssf_count = SurveyData2021.objects.filter(pension_nssf__in=['yes', 'Yes']).count()
        mbao_count = SurveyData2021.objects.filter(pension_mbao__in=['yes', 'Yes']).count()
        other_count = SurveyData2021.objects.filter(pension_other__in=['yes', 'Yes']).count()
        
        # Get raw counts for financial exclusion
        excluded_count = SurveyData2021.objects.filter(financially_excluded__in=['yes', 'Yes']).count()
        included_count = SurveyData2021.objects.filter(financially_excluded__in=['no', 'No']).count()
        
        # Log the counts for debugging
        logger.info(f"NSSF: {nssf_count}, MBAO: {mbao_count}, Other: {other_count}")
        logger.info(f"Excluded: {excluded_count}, Included: {included_count}")
        
        return JsonResponse({
            'pension': {
                'NSSF': {
                    'labels': ['Have', 'Do not have'],
                    'data': [nssf_count, SurveyData2021.objects.count() - nssf_count]
                },
                'MBAO': {
                    'labels': ['Have', 'Do not have'],
                    'data': [mbao_count, SurveyData2021.objects.count() - mbao_count]
                },
                'Other': {
                    'labels': ['Have', 'Do not have'],
                    'data': [other_count, SurveyData2021.objects.count() - other_count]
                }
            },
            'exclusion': {
                'labels': ['Financially Excluded', 'Financially Included'],
                'data': [excluded_count, included_count]
            }
        })
    except Exception as e:
        logger.error(f"Error in pension_and_exclusion_stats1: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def ui_tables(request):
    survey_data_2016 = list(
        SurveyData2016.objects.all().order_by('respondent_id')[:10])
    survey_data_2021 = list(
        SurveyData2021.objects.all().order_by('respondent_id')[:10])

    # Add this debug context
    debug_context = {
        '2016_data_exists': bool(survey_data_2016),
        '2016_count': len(survey_data_2016),
        '2021_data_exists': bool(survey_data_2021),
        '2021_count': len(survey_data_2021),
        'all_context': {
            'survey_data_2016': survey_data_2016,
            'survey_data_2021': survey_data_2021
        }
    }
    print("Debug Context:", debug_context)

    return render(request, 'home/ui-tables.html', {
        'survey_data_2016': survey_data_2016,
        'survey_data_2021': survey_data_2021,
        'debug': debug_context  # Add this line
    })


# Load models once when the app starts
# Update these paths in your views.py
model_path = os.path.join(
    # Changed .pdf to .pkl
    settings.CORE_DIR, 'apps', 'home', 'ml_models', 'voting_ensemble_model.pkl')
selector_path = os.path.join(
    # Changed .pdf to .pkl
    settings.CORE_DIR, 'apps', 'home', 'ml_models', 'feature_selector.pkl')
features_path = os.path.join(
    # Changed .pdf to .pkl
    settings.CORE_DIR, 'apps', 'home', 'ml_models', 'selected_features.pkl')

model = joblib.load(model_path)
selector = joblib.load(selector_path)
selected_features = joblib.load(features_path)

# Feature groups (same as in your Streamlit app)
feature_groups = {
    'formal_savings': ['bank_account_everyday'],
    'informal_savings': ['savings_secret_place'],
    'digital_financial': ['mobile_money_registered', 'savings_mobile_banking', 'loan_mobile_banking'],
    'formal_credit': ['loan_sacco'],
    'informal_credit': ['loan_group_chama', 'loan_family_friend', 'loan_goods_credit', 'loan_hire_purchase'],
    'insurance': ['pension_nssf']
}


def home_view(request):
    return render(request, 'ui-maps.html')


@csrf_exempt
@require_http_methods(["GET", "POST"])
def predict_view(request):
    """Handle prediction requests"""
    if request.method == 'GET':
        return render(request, 'home/ui-maps.html')

    try:
        # Create a dictionary to store the form data
        data = {
            # Demographic data - keep as strings for categorical variables
            'gender': request.POST.get('gender', '').lower(),
            'education_level': request.POST.get('education_level', '').lower(),
            'residence_type': request.POST.get('residence_type', '').lower(),
            'marital_status': request.POST.get('marital_status', '').lower(),
            'relationship_to_hh': request.POST.get('relationship_to_hh', '').lower(),
            'region': request.POST.get('region', '').lower(),
        }

        # Handle age separately since it needs to be numeric
        try:
            data['age'] = int(request.POST.get('age', 35))
        except (ValueError, TypeError):
            data['age'] = 35  # Default age if conversion fails

        # Define all possible behavioral fields
        behavioral_fields = [
            'mobile_money', 'bank_account', 'savings_account',
            'loan', 'insurance', 'pension', 'has_debit_card',
            'has_credit_card', 'savings_microfinance', 'savings_sacco',
            'savings_group'
        ]

        # Set behavioral fields to boolean values
        for field in behavioral_fields:
            data[field] = request.POST.get(field) == 'on'

        # Get model choice with default
        model_choice = request.POST.get(
            'model_choice', 'Decision Tree (Demographics + Behavior, SMOTE)')
        print(f"Model choice: {model_choice}")  # Debug print
        print(f"Input data: {data}")  # Debug print

        # Make prediction
        results = make_prediction(data, model_choice)
        print(f"Prediction results: {results}")  # Debug print

        # Return JSON response
        return JsonResponse({
            'status': 'success',
            'prediction': results['prediction'],
            # Already a percentage from make_prediction
            'probability': results['probability'],
            'factors': results['factors']
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug print
        return JsonResponse({
            'status': 'error',
            'error': f'Error making prediction: {str(e)}'
        }, status=500)


def financial_stats(request):
    stats = {
        'savings_mobile_banking': SurveyData2016.objects.aggregate(
            savings_mobile_banking_count=models.Count('savings_mobile_banking')
        )['savings_mobile_banking_count'],

        'bank_current': SurveyData2016.objects.aggregate(
            bank_current_count=models.Count('bank_account_current')
        )['bank_current_count'],

        'bank_savings': SurveyData2016.objects.aggregate(
            bank_savings_count=models.Count('bank_account_savings')
        )['bank_savings_count'],

        'bank_everyday': SurveyData2016.objects.aggregate(
            bank_everyday_count=models.Count('bank_account_everyday')
        )['bank_everyday_count'],

        'postbank': SurveyData2016.objects.aggregate(
            postbank_count=models.Count('postbank_account')
        )['postbank_count']
    }
    return JsonResponse(stats)


def financial_stats1(request):
    try:
        total = SurveyData2021.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Count 'Yes' responses for each category
        stats = {
            'savings_mobile_banking': round((SurveyData2021.objects.filter(savings_mobile_banking='Yes').count() / total) * 100, 2),
            'bank_current': round((SurveyData2021.objects.filter(bank_account_current='Yes').count() / total) * 100, 2),
            'bank_savings': round((SurveyData2021.objects.filter(bank_account_savings='Yes').count() / total) * 100, 2),
            'bank_everyday': round((SurveyData2021.objects.filter(bank_account_everyday='Yes').count() / total) * 100, 2),
            'postbank': round((SurveyData2021.objects.filter(postbank_account='Yes').count() / total) * 100, 2)
        }

        # Add debug logging
        print(f"2021 Financial Stats: {stats}")
        print(f"Total records: {total}")

        return JsonResponse(stats)
    except Exception as e:
        logger.error(f"Error in financial_stats1: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)



def residence_type_distribution1(request):
    # Get the counts for each residence type
    residence_data = SurveyData2021.objects.exclude(residence_type__isnull=True).values('residence_type').annotate(
        count=Count('residence_type')
    ).order_by('residence_type')

    labels = []
    data = []
    for entry in residence_data:
        labels.append(entry['residence_type'].capitalize())
        data.append(entry['count'])

    return JsonResponse({
        'labels': labels,
        'data': data,
    })


class SurveyData2016ViewSet(viewsets.ModelViewSet):
    queryset = SurveyData2016.objects.all()
    serializer_class = SurveyData2016Serializer


class SurveyData2021ViewSet(viewsets.ModelViewSet):
    queryset = SurveyData2021.objects.all()
    serializer_class = SurveyData2021Serializer


@api_view(['GET'])
def survey_data_2016_table(request):
    data = SurveyData2016.objects.all().order_by('respondent_id')[:10]
    serializer = SurveyData2016Serializer(data, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def survey_data_2021_table(request):
    data = SurveyData2021.objects.all().order_by('respondent_id')[:10]
    serializer = SurveyData2021Serializer(data, many=True)
    return Response(serializer.data)


################
#######################


def load_models():
    """Load all ML models and components"""
    models = {
        # Update with your actual paths
        'decision_tree': joblib.load(settings.MODEL_PATH),
        # Update with your actual paths
        'logistic_regression': joblib.load(settings.MODEL_PATH),
        # Update with your actual paths
        'gradient_boosting': joblib.load(settings.MODEL_PATH)
    }
    feature_selector = joblib.load(settings.FEATURE_SELECTOR_PATH)
    selected_features = joblib.load(settings.SELECTED_FEATURES_PATH)
    return models, feature_selector, selected_features


def prepare_input_data(form_data, selected_features):
    """Convert form data to model input format"""
    input_data = {
        'age': [form_data['age']],
        'gender': [form_data['gender']],
        'education_level': [form_data['education_level']],
        'residence_type': [form_data['residence_type']],
        'marital_status': [form_data['marital_status']],
        'relationship_to_hh': [form_data['relationship_to_hh']],
        'region': [form_data['region']],
        'mobile_money': [1 if form_data.get('mobile_money', False) else 0],
        'bank_account': [1 if form_data.get('bank_account', False) else 0],
        'savings_account': [1 if form_data.get('savings_account', False) else 0],
        'insurance': [1 if form_data.get('insurance', False) else 0],
        'pension': [1 if form_data.get('pension', False) else 0]
    }

    # Create DataFrame and one-hot encode categorical variables
    df = pd.DataFrame(input_data)
    categorical_cols = ['gender', 'education_level', 'residence_type',
                        'marital_status', 'relationship_to_hh', 'region']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Ensure all expected columns are present
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    # Select only the features the model expects
    return df[selected_features]


def generate_explanation(model, input_data, feature_names):
    """Generate LIME explanation for the prediction"""
    explainer = LimeTabularExplainer(
        training_data=np.zeros((1, len(feature_names))),  # Dummy data
        mode="classification",
        feature_names=feature_names,
        class_names=['Financially Included', 'Financially Excluded'],
        discretize_continuous=False
    )

    exp = explainer.explain_instance(
        input_data[0],
        model.predict_proba,
        num_features=len(feature_names)
    )

    # Process explanation for template
    explanation = []
    for feature, weight in exp.as_list():
        explanation.append({
            'feature': feature,
            'weight': abs(weight),
            'direction': 'increase' if weight > 0 else 'decrease'
        })

    # Sort by absolute weight
    explanation.sort(key=lambda x: x['weight'], reverse=True)
    return explanation[:5]  # Return top 5 factors


@require_http_methods(["GET"])
def banking_services_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        # Calculate percentages for each banking service
        services_data = {
            'Mobile Banking': SurveyData2016.objects.filter(savings_mobile_banking='Yes').count(),
            'Current Account': SurveyData2016.objects.filter(bank_account_current='Yes').count(),
            'Savings Account': SurveyData2016.objects.filter(bank_account_savings='Yes').count(),
            'Everyday Account': SurveyData2016.objects.filter(bank_account_everyday='Yes').count(),
            'Post Bank': SurveyData2016.objects.filter(postbank_account='Yes').count()
        }
        
        # Convert to percentages
        services_percentages = {
            key: round((value / total) * 100, 2) 
            for key, value in services_data.items()
        }
        
        data = {
            'labels': list(services_percentages.keys()),
            'data': list(services_percentages.values())
        }
        
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in banking_services_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def digital_services_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        # Calculate percentages for each digital service
        services_data = {
            'Mobile Money': SurveyData2016.objects.filter(mobile_money_registered='Yes').count(),
            'Mobile Banking': SurveyData2016.objects.filter(savings_mobile_banking='Yes').count(),
            'Digital Loans': SurveyData2016.objects.filter(loan_digital_app='Yes').count()
        }
        
        # Convert to percentages
        services_percentages = {
            key: round((value / total) * 100, 2) 
            for key, value in services_data.items()
        }
        
        data = {
            'labels': list(services_percentages.keys()),
            'data': list(services_percentages.values())
        }
        
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in digital_services_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def loan_sources_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        # Calculate percentages for each loan source
        loan_data = {
            'Bank Loans': SurveyData2016.objects.filter(loan_bank='Yes').count(),
            'SACCO Loans': SurveyData2016.objects.filter(loan_sacco='Yes').count(),
            'Mobile Loans': SurveyData2016.objects.filter(loan_mobile_banking='Yes').count(),
            'MFI Loans': SurveyData2016.objects.filter(loan_microfinance='Yes').count(),
            'Informal Loans': SurveyData2016.objects.filter(loan_family_friend='Yes').count()
        }
        
        # Convert to percentages
        loan_percentages = {
            key: round((value / total) * 100, 2) 
            for key, value in loan_data.items()
        }
        
        data = {
            'labels': list(loan_percentages.keys()),
            'data': list(loan_percentages.values())
        }
        
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in loan_sources_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def insurance_types_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        # Calculate percentages for each insurance type
        insurance_counts = {
            'Health Insurance': SurveyData2016.objects.filter(insurance_health_other='Yes').count(),
            'Life Insurance': SurveyData2016.objects.filter(insurance_life='Yes').count(),
            'Property Insurance': SurveyData2016.objects.filter(insurance_home='Yes').count(),
            'Education Insurance': SurveyData2016.objects.filter(insurance_education='Yes').count(),
            'NHIF': SurveyData2016.objects.filter(insurance_nhif='Yes').count()
        }
        
        # Convert to percentages
        insurance_percentages = {
            key: round((value / total) * 100, 2) 
            for key, value in insurance_counts.items()
        }
        
        data = {
            'labels': list(insurance_percentages.keys()),
            'data': list(insurance_percentages.values())
        }
        
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in insurance_types_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def savings_methods_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        # Calculate percentages for each savings method
        savings_counts = {
            'Bank Savings': SurveyData2016.objects.filter(bank_account_savings='Yes').count(),
            'SACCO Savings': SurveyData2016.objects.filter(savings_sacco='Yes').count(),
            'Mobile Savings': SurveyData2016.objects.filter(savings_mobile_banking='Yes').count(),
            'Group Savings': SurveyData2016.objects.filter(savings_group_friends='Yes').count(),
            'Informal Savings': SurveyData2016.objects.filter(savings_secret_place='Yes').count()
        }
        
        # Convert to percentages
        savings_percentages = {
            key: round((value / total) * 100, 2) 
            for key, value in savings_counts.items()
        }
        
        data = {
            'labels': list(savings_percentages.keys()),
            'data': list(savings_percentages.values()),
            'backgroundColor': [
                'rgba(99, 102, 241, 0.7)',
                'rgba(236, 72, 153, 0.7)',
                'rgba(34, 197, 94, 0.7)',
                'rgba(249, 115, 22, 0.7)',
                'rgba(6, 182, 212, 0.7)'
            ],
            'borderColor': [
                'rgba(99, 102, 241, 1)',
                'rgba(236, 72, 153, 1)',
                'rgba(34, 197, 94, 1)',
                'rgba(249, 115, 22, 1)',
                'rgba(6, 182, 212, 1)'
            ]
        }
        
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in savings_methods_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def credit_types_api(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)
            
        data = {
            'labels': ['Bank Loans', 'Digital Loans', 'SACCO Loans', 'Chama/Group Loans', 'Government Loans', 'Employer Loans'],
            'data': [
                SurveyData2016.objects.filter(loan_bank='Yes').count() * 100 / total,
                SurveyData2016.objects.filter(loan_digital_app='Yes').count() * 100 / total,
                SurveyData2016.objects.filter(loan_sacco='Yes').count() * 100 / total,
                SurveyData2016.objects.filter(loan_group_chama='Yes').count() * 100 / total,
                SurveyData2016.objects.filter(loan_govt='Yes').count() * 100 / total,
                SurveyData2016.objects.filter(loan_employer='Yes').count() * 100 / total
            ]
        }
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def digital_vs_traditional_loans_api(request):
    try:
        # Get total count of valid responses for each loan type
        total_respondents = SurveyData2021.objects.count()
        logger.debug(f"Total respondents: {total_respondents}")

        if total_respondents == 0:
            return Response({
                "error": "No data available"
            }, status=404)

        # Count for digital app loans (case-insensitive)
        digital_loans = SurveyData2021.objects.filter(
            loan_digital_app__iexact='yes'
        ).count()
        logger.debug(f"Digital app loans count: {digital_loans}")

        # Count for mobile banking loans (case-insensitive)
        mobile_loans = SurveyData2021.objects.filter(
            loan_mobile_banking__iexact='yes'
        ).count()
        logger.debug(f"Mobile banking loans count: {mobile_loans}")

        # Count for traditional bank loans (case-insensitive)
        traditional_loans = SurveyData2021.objects.filter(
            loan_bank__iexact='yes'
        ).count()
        logger.debug(f"Traditional bank loans count: {traditional_loans}")

        # Calculate percentages based on total respondents
        digital_percent = round((digital_loans / total_respondents) * 100, 2)
        mobile_percent = round((mobile_loans / total_respondents) * 100, 2)
        traditional_percent = round((traditional_loans / total_respondents) * 100, 2)

        logger.debug(f"Percentages - Digital: {digital_percent}%, Mobile: {mobile_percent}%, Traditional: {traditional_percent}%")

        response_data = {
            "labels": ["Digital App Loans", "Mobile Banking Loans", "Traditional Bank Loans"],
            "data": [digital_percent, mobile_percent, traditional_percent],
            "colors": ["#FF6384", "#36A2EB", "#FFCE56"]
        }
        
        logger.debug(f"Returning response data: {response_data}")
        return Response(response_data)

    except Exception as e:
        logger.error(f"Error in digital_vs_traditional_loans_api: {str(e)}")
        return Response({
            "error": f"An error occurred while processing the request: {str(e)}"
        }, status=500)

@csrf_exempt
@require_GET
def informal_lending_api(request):
    try:
        # Count responses for each informal lending source
        informal_sources = {
            'Shylock': SurveyData2021.objects.filter(
                loan_shylock__in=['yes', 'Yes', 'loan_shylock']
            ).count(),
            'Family/Friends': SurveyData2021.objects.filter(
                loan_family_friend__in=['yes', 'Yes', 'loan_family_friend']
            ).count(),
            'Group/Chama': SurveyData2021.objects.filter(
                loan_group_chama__in=['yes', 'Yes', 'loan_group_chama']
            ).count(),
            'Shopkeeper': SurveyData2021.objects.filter(
                loan_shopkeeper_cash__in=['yes', 'Yes', 'loan_shopkeeper']
            ).count()
        }
        
        # Debug logging
        for source, count in informal_sources.items():
            print(f"DEBUG: {source}: {count}")
        
        data = {
            'labels': list(informal_sources.keys()),
            'data': list(informal_sources.values()),
            'colors': ['#f59e0b', '#d97706', '#b45309', '#92400e']
        }
        return JsonResponse(data)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def health_insurance_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            insurance_nhif__isnull=True,
            insurance_health_other__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        nhif = SurveyData2021.objects.filter(insurance_nhif='Yes').count()
        other_health = SurveyData2021.objects.filter(insurance_health_other='Yes').count()
        
        # Calculate percentages
        if total_respondents > 0:
            nhif_percent = round((nhif/total_respondents)*100, 2)
            other_percent = round((other_health/total_respondents)*100, 2)
        else:
            nhif_percent = other_percent = 0
        
        # Log the values for debugging
        logger.info(f"NHIF: {nhif}/{total_respondents} = {nhif_percent}%")
        logger.info(f"Other Health Insurance: {other_health}/{total_respondents} = {other_percent}%")
        
        data = {
            'labels': ['NHIF', 'Other Health Insurance'],
            'data': [nhif_percent, other_percent],
            'colors': ['#10b981', '#059669']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in health_insurance_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def asset_insurance_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            insurance_motor__isnull=True,
            insurance_home__isnull=True,
            insurance_life__isnull=True,
            insurance_education__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        assets = {
            'Motor': SurveyData2021.objects.filter(insurance_motor='Yes').count(),
            'Home': SurveyData2021.objects.filter(insurance_home='Yes').count(),
            'Life': SurveyData2021.objects.filter(insurance_life='Yes').count(),
            'Education': SurveyData2021.objects.filter(insurance_education='Yes').count()
        }
        
        # Calculate percentages
        if total_respondents > 0:
            percentages = {k: round((v/total_respondents)*100, 2) for k, v in assets.items()}
        else:
            percentages = {k: 0 for k in assets.keys()}
        
        # Log the values for debugging
        for asset, percent in percentages.items():
            logger.info(f"{asset}: {assets[asset]}/{total_respondents} = {percent}%")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in asset_insurance_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def agri_insurance_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            insurance_crop__isnull=True,
            insurance_livestock__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        crop = SurveyData2021.objects.filter(insurance_crop='Yes').count()
        livestock = SurveyData2021.objects.filter(insurance_livestock='Yes').count()
        
        # Calculate percentages
        if total_respondents > 0:
            crop_percent = round((crop/total_respondents)*100, 2)
            livestock_percent = round((livestock/total_respondents)*100, 2)
        else:
            crop_percent = livestock_percent = 0
        
        # Log the values for debugging
        logger.info(f"Crop Insurance: {crop}/{total_respondents} = {crop_percent}%")
        logger.info(f"Livestock Insurance: {livestock}/{total_respondents} = {livestock_percent}%")
        
        data = {
            'labels': ['Crop Insurance', 'Livestock Insurance'],
            'data': [crop_percent, livestock_percent],
            'colors': ['#ec4899', '#db2777']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in agri_insurance_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def savings_channels_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            savings_mobile_banking__isnull=True,
            savings_microfinance__isnull=True,
            savings_sacco__isnull=True,
            savings_group_friends__isnull=True,
            savings_family_friend__isnull=True,
            savings_secret_place__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        channels = {
            'Mobile Banking': SurveyData2021.objects.filter(savings_mobile_banking='Yes').count(),
            'Microfinance': SurveyData2021.objects.filter(savings_microfinance='Yes').count(),
            'SACCO': SurveyData2021.objects.filter(savings_sacco='Yes').count(),
            'Group/Friends': SurveyData2021.objects.filter(savings_group_friends='Yes').count(),
            'Family/Friend': SurveyData2021.objects.filter(savings_family_friend='Yes').count(),
            'Secret Place': SurveyData2021.objects.filter(savings_secret_place='Yes').count()
        }
        
        # Calculate percentages
        if total_respondents > 0:
            percentages = {k: round((v/total_respondents)*100, 2) for k, v in channels.items()}
        else:
            percentages = {k: 0 for k in channels.keys()}
        
        # Log the values for debugging
        for channel, percent in percentages.items():
            logger.info(f"{channel}: {channels[channel]}/{total_respondents} = {percent}%")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#164e63', '#083344']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in savings_channels_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def digital_savings_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            savings_mobile_banking__isnull=True,
            debit_card__isnull=True,
            credit_card__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        digital_products = {
            'Mobile Banking': SurveyData2021.objects.filter(savings_mobile_banking='Yes').count(),
            'Digital Wallet': SurveyData2021.objects.filter(debit_card='Yes').count(),
            'Credit Card': SurveyData2021.objects.filter(credit_card='Yes').count()
        }
        
        # Calculate percentages
        if total_respondents > 0:
            percentages = {k: round((v/total_respondents)*100, 2) for k, v in digital_products.items()}
        else:
            percentages = {k: 0 for k in digital_products.keys()}
        
        # Log the values for debugging
        for product, percent in percentages.items():
            logger.info(f"{product}: {digital_products[product]}/{total_respondents} = {percent}%")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#f43f5e', '#e11d48', '#be123c']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in digital_savings_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def account_types_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            bank_account_current__isnull=True,
            bank_account_savings__isnull=True,
            bank_account_everyday__isnull=True,
            postbank_account__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        accounts = {
            'Current Account': SurveyData2021.objects.filter(bank_account_current='Yes').count(),
            'Savings Account': SurveyData2021.objects.filter(bank_account_savings='Yes').count(),
            'Everyday Account': SurveyData2021.objects.filter(bank_account_everyday='Yes').count(),
            'Postbank Account': SurveyData2021.objects.filter(postbank_account='Yes').count()
        }
        
        # Calculate percentages
        if total_respondents > 0:
            percentages = {k: round((v/total_respondents)*100, 2) for k, v in accounts.items()}
        else:
            percentages = {k: 0 for k in accounts.keys()}
        
        # Log the values for debugging
        for account, percent in percentages.items():
            logger.info(f"{account}: {accounts[account]}/{total_respondents} = {percent}%")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#6366f1', '#4f46e5', '#4338ca', '#3730a3']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in account_types_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_GET
def banking_products_api(request):
    try:
        # Get total respondents excluding null values
        total_respondents = SurveyData2021.objects.exclude(
            debit_card__isnull=True,
            credit_card__isnull=True,
            bank_overdraft__isnull=True
        ).count()
        
        # Count only 'Yes' responses
        products = {
            'Debit Card': SurveyData2021.objects.filter(debit_card='Yes').count(),
            'Credit Card': SurveyData2021.objects.filter(credit_card='Yes').count(),
            'Overdraft': SurveyData2021.objects.filter(bank_overdraft='Yes').count()
        }
        
        # Calculate percentages
        if total_respondents > 0:
            percentages = {k: round((v/total_respondents)*100, 2) for k, v in products.items()}
        else:
            percentages = {k: 0 for k in products.keys()}
        
        # Log the values for debugging
        for product, percent in percentages.items():
            logger.info(f"{product}: {products[product]}/{total_respondents} = {percent}%")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#0ea5e9', '#0284c7', '#0369a1']
        }
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in banking_products_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
