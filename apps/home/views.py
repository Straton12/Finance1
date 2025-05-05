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
        if total == 0:
            return JsonResponse({'labels': [], 'data': []})
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
        return JsonResponse({'labels': [], 'data': []})


@csrf_exempt
@require_http_methods(["GET"])
def education_distribution(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'labels': [], 'data': []})
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
        return JsonResponse({'labels': [], 'data': []})


@csrf_exempt
@require_http_methods(["GET"])
def residence_type_distribution(request):
    try:
        total = SurveyData2016.objects.count()
        if total == 0:
            return JsonResponse({'labels': [], 'data': []})
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
        return JsonResponse({'labels': [], 'data': []})


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
        survey_data = SurveyData2021.objects.all()
        
        # Calculate total respondents
        total_respondents = survey_data.count()
        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Calculate pension statistics
        pension_stats = {
            'NSSF': round(survey_data.filter(pension_nssf='Yes').count() * 100 / total_respondents, 2),
            'MBAO': round(survey_data.filter(pension_mbao='Yes').count() * 100 / total_respondents, 2),
            'Other': round(survey_data.filter(pension_other='Yes').count() * 100 / total_respondents, 2)
        }

        # Financial exclusion statistics
        excluded = survey_data.filter(financially_excluded__iexact='Yes').count()
        included = total_respondents - excluded
        exclusion_percentage = round((excluded / total_respondents) * 100, 2) if total_respondents else 0

        # Return JSON response
        return JsonResponse({
            'pension': {
                'labels': list(pension_stats.keys()),
                'data': list(pension_stats.values()),
                'colors': ['#64748b', '#475569', '#334155']
            },
            'exclusion': {
                'labels': ['Excluded', 'Included'],
                'data': [excluded, included],
                'percentage': exclusion_percentage,
                'colors': ['#ec4899', '#db2777']
            }
        })

    except Exception as e:
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


@login_required(login_url="/login/")
def ui_maps(request):
    return render(request, 'home/ui-maps.html')


@csrf_exempt
@require_http_methods(["GET", "POST"])
def predict_view(request):
    if request.method == 'GET':
        return render(request, 'home/ui-maps.html')
    
    try:
        # Extract form data
        form_data = {
            'age': int(request.POST.get('age', 0)),
            'gender': request.POST.get('gender', '').lower(),
            'education_level': request.POST.get('education_level', '').lower(),
            'residence_type': request.POST.get('residence_type', '').lower(),
            'marital_status': request.POST.get('marital_status', '').lower(),
            'relationship_to_hh': request.POST.get('relationship_to_hh', '').lower(),
            'region': request.POST.get('region', '').lower(),
            'mobile_money': request.POST.get('mobile_money') == 'on',
            'bank_account': request.POST.get('bank_account') == 'on',
            'savings_account': request.POST.get('savings_account') == 'on',
            'loan': request.POST.get('loan') == 'on',
            'insurance': request.POST.get('insurance') == 'on',
            'pension': request.POST.get('pension') == 'on',
            'has_debit_card': request.POST.get('has_debit_card') == 'on',
            'has_credit_card': request.POST.get('has_credit_card') == 'on',
            'savings_microfinance': request.POST.get('savings_microfinance') == 'on',
            'savings_sacco': request.POST.get('savings_sacco') == 'on',
            'savings_group': request.POST.get('savings_group') == 'on'
        }

        # Get selected model
        model_choice = request.POST.get('model_choice', 'Decision Tree (Demographics + Behavior, SMOTE)')
        
        # Make prediction
        result = make_prediction(form_data, model_choice)
        
        if result['status'] == 'success':
            return JsonResponse(result)
        else:
            return JsonResponse({
                'status': 'error',
                'error': result.get('error', 'Unknown error occurred')
            }, status=500)
    
    except Exception as e:
        logger.error(f"Error in predict_view: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@require_GET
def financial_stats(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for financial stats: {total}")
        
        if total == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Count 'Yes' responses for each category (case-insensitive)
        stats = {
            'savings_mobile_banking': round((SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count() / total) * 100, 2),
            'bank_current': round((SurveyData2016.objects.filter(Q(bank_account_current__iexact='yes') | Q(bank_account_current__iexact='y')).count() / total) * 100, 2),
            'bank_savings': round((SurveyData2016.objects.filter(Q(bank_account_savings__iexact='yes') | Q(bank_account_savings__iexact='y')).count() / total) * 100, 2),
            'bank_everyday': round((SurveyData2016.objects.filter(Q(bank_account_everyday__iexact='yes') | Q(bank_account_everyday__iexact='y')).count() / total) * 100, 2),
            'postbank': round((SurveyData2016.objects.filter(Q(postbank_account__iexact='yes') | Q(postbank_account__iexact='y')).count() / total) * 100, 2)
        }
        
        print(f"2016 Financial Stats: {stats}")
        return JsonResponse(stats)
    except Exception as e:
        logger.error(f"Error in financial_stats: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


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


@require_GET
def banking_services_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for banking services: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        services_data = {
            'Mobile Banking': SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count(),
            'Current Account': SurveyData2016.objects.filter(Q(bank_account_current__iexact='yes') | Q(bank_account_current__iexact='y')).count(),
            'Savings Account': SurveyData2016.objects.filter(Q(bank_account_savings__iexact='yes') | Q(bank_account_savings__iexact='y')).count(),
            'Everyday Account': SurveyData2016.objects.filter(Q(bank_account_everyday__iexact='yes') | Q(bank_account_everyday__iexact='y')).count(),
            'Post Bank': SurveyData2016.objects.filter(Q(postbank_account__iexact='yes') | Q(postbank_account__iexact='y')).count()
        }
        
        print(f"Raw banking services counts: {services_data}")
        
        # Calculate percentages
        services_percentages = {
            key: round((value / total) * 100, 2)
            for key, value in services_data.items()
        }
        
        print(f"Banking services percentages: {services_percentages}")
        
        color_palette = [
            'rgba(99, 102, 241, 0.7)',
            'rgba(129, 140, 248, 0.7)',
            'rgba(165, 180, 252, 0.7)',
            'rgba(199, 210, 254, 0.7)',
            'rgba(224, 231, 255, 0.7)'
        ]
        
        data = {
            'labels': list(services_percentages.keys()),
            'data': list(services_percentages.values()),
            'colors': color_palette[:len(services_percentages)]
        }
        
        print(f"Final banking services response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in banking_services_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def digital_services_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for digital services: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        services_data = {
            'Mobile Money': SurveyData2016.objects.filter(Q(mobile_money_registered__iexact='yes') | Q(mobile_money_registered__iexact='y')).count(),
            'Mobile Banking': SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count(),
            'Digital Loans': SurveyData2016.objects.filter(Q(loan_digital_app__iexact='yes') | Q(loan_digital_app__iexact='y')).count()
        }
        
        print(f"Raw digital services counts: {services_data}")
        
        # Calculate percentages
        services_percentages = {
            key: round((value / total) * 100, 2)
            for key, value in services_data.items()
        }
        
        print(f"Digital services percentages: {services_percentages}")
        
        color_palette = [
            'rgba(99, 102, 241, 0.7)',
            'rgba(129, 140, 248, 0.7)',
            'rgba(165, 180, 252, 0.7)'
        ]
        
        data = {
            'labels': list(services_percentages.keys()),
            'data': list(services_percentages.values()),
            'colors': color_palette[:len(services_percentages)]
        }
        
        print(f"Final digital services response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in digital_services_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def loan_sources_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for loan sources: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        loan_data = {
            'Bank Loans': SurveyData2016.objects.filter(Q(loan_bank__iexact='yes') | Q(loan_bank__iexact='y')).count(),
            'SACCO Loans': SurveyData2016.objects.filter(Q(loan_sacco__iexact='yes') | Q(loan_sacco__iexact='y')).count(),
            'Mobile Loans': SurveyData2016.objects.filter(Q(loan_mobile_banking__iexact='yes') | Q(loan_mobile_banking__iexact='y')).count(),
            'MFI Loans': SurveyData2016.objects.filter(Q(loan_microfinance__iexact='yes') | Q(loan_microfinance__iexact='y')).count(),
            'Informal Loans': SurveyData2016.objects.filter(Q(loan_family_friend__iexact='yes') | Q(loan_family_friend__iexact='y')).count()
        }
        
        print(f"Raw loan source counts: {loan_data}")
        
        # Calculate percentages
        loan_percentages = {k: round((v / total) * 100, 2) for k, v in loan_data.items()}
        
        print(f"Loan source percentages: {loan_percentages}")
        
        color_palette = [
            'rgba(99, 102, 241, 0.7)',
            'rgba(129, 140, 248, 0.7)',
            'rgba(165, 180, 252, 0.7)',
            'rgba(199, 210, 254, 0.7)',
            'rgba(224, 231, 255, 0.7)'
        ]
        
        data = {
            'labels': list(loan_percentages.keys()),
            'data': list(loan_percentages.values()),
            'colors': color_palette[:len(loan_percentages)]
        }
        
        print(f"Final loan sources response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in loan_sources_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def credit_types_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for credit types: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        credit_data = {
            'Bank Loans': SurveyData2016.objects.filter(Q(loan_bank__iexact='yes') | Q(loan_bank__iexact='y')).count(),
            'Digital Loans': SurveyData2016.objects.filter(Q(loan_digital_app__iexact='yes') | Q(loan_digital_app__iexact='y')).count(),
            'SACCO Loans': SurveyData2016.objects.filter(Q(loan_sacco__iexact='yes') | Q(loan_sacco__iexact='y')).count(),
            'Chama/Group Loans': SurveyData2016.objects.filter(Q(loan_group_chama__iexact='yes') | Q(loan_group_chama__iexact='y')).count(),
            'Government Loans': SurveyData2016.objects.filter(Q(loan_govt__iexact='yes') | Q(loan_govt__iexact='y')).count(),
            'Employer Loans': SurveyData2016.objects.filter(Q(loan_employer__iexact='yes') | Q(loan_employer__iexact='y')).count()
        }
        
        print(f"Raw credit type counts: {credit_data}")
        
        # Calculate percentages
        credit_percentages = {k: round((v / total) * 100, 2) for k, v in credit_data.items()}
        
        print(f"Credit type percentages: {credit_percentages}")
        
        color_palette = [
            'rgba(99, 102, 241, 0.7)',
            'rgba(129, 140, 248, 0.7)',
            'rgba(165, 180, 252, 0.7)',
            'rgba(199, 210, 254, 0.7)',
            'rgba(224, 231, 255, 0.7)',
            'rgba(238, 242, 255, 0.7)'
        ]
        
        data = {
            'labels': list(credit_percentages.keys()),
            'data': list(credit_percentages.values()),
            'colors': color_palette[:len(credit_percentages)]
        }
        
        print(f"Final credit types response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in credit_types_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def insurance_types_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total insurance respondents: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        insurance_counts = {
            'Health Insurance': SurveyData2016.objects.filter(Q(insurance_health_other__iexact='yes') | Q(insurance_health_other__iexact='y')).count(),
            'Life Insurance': SurveyData2016.objects.filter(Q(insurance_life__iexact='yes') | Q(insurance_life__iexact='y')).count(),
            'Property Insurance': SurveyData2016.objects.filter(Q(insurance_home__iexact='yes') | Q(insurance_home__iexact='y')).count(),
            'Education Insurance': SurveyData2016.objects.filter(Q(insurance_education__iexact='yes') | Q(insurance_education__iexact='y')).count(),
            'NHIF': SurveyData2016.objects.filter(Q(insurance_nhif__iexact='yes') | Q(insurance_nhif__iexact='y')).count()
        }
        
        print(f"Raw insurance counts: {insurance_counts}")
        
        # Calculate percentages
        insurance_percentages = {k: round((v / total) * 100, 2) for k, v in insurance_counts.items()}
        
        print(f"Insurance percentages: {insurance_percentages}")
        
        data = {
            'labels': list(insurance_percentages.keys()),
            'data': list(insurance_percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#164e63']
        }
        
        print(f"Final insurance types response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in insurance_types_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def savings_methods_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for savings methods: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        savings_counts = {
            'Bank Savings': SurveyData2016.objects.filter(Q(bank_account_savings__iexact='yes') | Q(bank_account_savings__iexact='y')).count(),
            'SACCO Savings': SurveyData2016.objects.filter(Q(savings_sacco__iexact='yes') | Q(savings_sacco__iexact='y')).count(),
            'Mobile Savings': SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count(),
            'Group Savings': SurveyData2016.objects.filter(Q(savings_group_friends__iexact='yes') | Q(savings_group_friends__iexact='y')).count(),
            'Informal Savings': SurveyData2016.objects.filter(Q(savings_secret_place__iexact='yes') | Q(savings_secret_place__iexact='y')).count()
        }
        
        print(f"Raw savings counts: {savings_counts}")
        
        # Calculate percentages
        savings_percentages = {k: round((v / total) * 100, 2) for k, v in savings_counts.items()}
        
        print(f"Savings percentages: {savings_percentages}")
        
        data = {
            'labels': list(savings_percentages.keys()),
            'data': list(savings_percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#164e63', '#083344']
        }
        
        print(f"Final savings methods response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in savings_methods_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def savings_type_distribution_2016(request):
    """Return the distribution of formal vs informal savings for 2016."""
    try:
        total_respondents = SurveyData2016.objects.count()
        
        # Count formal savings (bank, SACCO, microfinance)
        formal_savings = SurveyData2016.objects.filter(
            Q(bank_account_savings__iexact='yes') |
            Q(savings_sacco__iexact='yes') |
            Q(savings_microfinance__iexact='yes') |
            Q(savings_mobile_banking__iexact='yes')
        ).distinct().count()
        
        # Count informal savings (secret place, group/friends, family/friend)
        informal_savings = SurveyData2016.objects.filter(
            Q(savings_secret_place__iexact='yes') |
            Q(savings_group_friends__iexact='yes') |
            Q(savings_family_friend__iexact='yes')
        ).distinct().count()
        
        # Calculate percentages
        formal_percent = round((formal_savings / total_respondents) * 100, 2)
        informal_percent = round((informal_savings / total_respondents) * 100, 2)
        
        # Log the data for debugging
        logger.info(f"Savings distribution - Formal: {formal_percent}%, Informal: {informal_percent}%")
        logger.info(f"Raw counts - Formal: {formal_savings}, Informal: {informal_savings}, Total: {total_respondents}")
        
        response_data = {
            'labels': ['Formal Savings', 'Informal Savings'],
            'data': [formal_percent, informal_percent],
            'colors': ['#f43f5e', '#e11d48']
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in savings_type_distribution_2016: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def digital_vs_traditional_loans_api(request):
    total_respondents = SurveyData2021.objects.count()
    if total_respondents == 0:
        return JsonResponse({'labels': [], 'data': [], 'colors': []})
    # Get total count of valid responses for each loan type
    digital_loans = SurveyData2021.objects.filter(loan_digital_app='Yes').count()
    mobile_loans = SurveyData2021.objects.filter(loan_mobile_banking='Yes').count()
    traditional_loans = SurveyData2021.objects.filter(loan_bank='Yes').count()

    # Calculate percentages based on total respondents
    digital_percent = round((digital_loans / total_respondents) * 100, 2)
    mobile_percent = round((mobile_loans / total_respondents) * 100, 2)
    traditional_percent = round((traditional_loans / total_respondents) * 100, 2)

    response_data = {
        "labels": ["Digital App Loans", "Mobile Banking Loans", "Traditional Bank Loans"],
        "data": [digital_percent, mobile_percent, traditional_percent],
        "colors": ["#FF6384", "#36A2EB", "#FFCE56"]
    }
    return JsonResponse(response_data)

@require_GET
def informal_lending_api(request):
    total_respondents = SurveyData2021.objects.count()
    if total_respondents == 0:
        return JsonResponse({'labels': [], 'data': [], 'colors': []})
    # Count responses for each informal lending source
    informal_sources = {
        'Shylock': SurveyData2021.objects.filter(loan_shylock='Yes').count(),
        'Family/Friends': SurveyData2021.objects.filter(loan_family_friend='Yes').count(),
        'Group/Chama': SurveyData2021.objects.filter(loan_group_chama='Yes').count(),
        'Shopkeeper': SurveyData2021.objects.filter(loan_shopkeeper_cash='Yes').count()
    }
    # Calculate percentages
    percentages = [round((count / total_respondents) * 100, 2) for count in informal_sources.values()]
    data = {
        'labels': list(informal_sources.keys()),
        'data': percentages,
        'colors': ['#f59e0b', '#d97706', '#b45309', '#92400e']
    }
    return JsonResponse(data)

@require_GET
def health_insurance_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total health insurance respondents: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        health_counts = {
            'NHIF': SurveyData2016.objects.filter(Q(insurance_nhif__iexact='yes') | Q(insurance_nhif__iexact='y')).count(),
            'Other Health Insurance': SurveyData2016.objects.filter(Q(insurance_health_other__iexact='yes') | Q(insurance_health_other__iexact='y')).count()
        }
        
        print(f"Raw health insurance counts: {health_counts}")
        
        # Calculate percentages
        health_percentages = {k: round((v / total) * 100, 2) for k, v in health_counts.items()}
        
        print(f"Health insurance percentages: {health_percentages}")
        
        data = {
            'labels': list(health_percentages.keys()),
            'data': list(health_percentages.values()),
            'colors': ['#06b6d4', '#0891b2']
        }
        
        print(f"Final health insurance response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in health_insurance_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def asset_insurance_api(request):
    total_respondents = SurveyData2021.objects.count()
    if total_respondents == 0:
        return JsonResponse({'labels': [], 'data': [], 'colors': []})
    assets = {
        'Motor': SurveyData2021.objects.filter(insurance_motor='Yes').count(),
        'Home': SurveyData2021.objects.filter(insurance_home='Yes').count(),
        'Life': SurveyData2021.objects.filter(insurance_life='Yes').count(),
        'Education': SurveyData2021.objects.filter(insurance_education='Yes').count()
    }
    percentages = [round((v / total_respondents) * 100, 2) for v in assets.values()]
    data = {
        'labels': list(assets.keys()),
        'data': percentages,
        'colors': ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af']
    }
    return JsonResponse(data)

@require_GET
def agri_insurance_api(request):
    total_respondents = SurveyData2021.objects.count()
    if total_respondents == 0:
        return JsonResponse({'labels': [], 'data': [], 'colors': []})
    crop = SurveyData2021.objects.filter(insurance_crop='Yes').count()
    livestock = SurveyData2021.objects.filter(insurance_livestock='Yes').count()
    crop_percent = round((crop / total_respondents) * 100, 2)
    livestock_percent = round((livestock / total_respondents) * 100, 2)
    data = {
        'labels': ['Crop Insurance', 'Livestock Insurance'],
        'data': [crop_percent, livestock_percent],
        'colors': ['#ec4899', '#db2777']
    }
    return JsonResponse(data)

@require_GET
def savings_channels_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for savings channels: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        channels = {
            'Mobile Banking': SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count(),
            'Microfinance': SurveyData2016.objects.filter(Q(savings_microfinance__iexact='yes') | Q(savings_microfinance__iexact='y')).count(),
            'SACCO': SurveyData2016.objects.filter(Q(savings_sacco__iexact='yes') | Q(savings_sacco__iexact='y')).count(),
            'Group/Friends': SurveyData2016.objects.filter(Q(savings_group_friends__iexact='yes') | Q(savings_group_friends__iexact='y')).count(),
            'Family/Friend': SurveyData2016.objects.filter(Q(savings_family_friend__iexact='yes') | Q(savings_family_friend__iexact='y')).count(),
            'Secret Place': SurveyData2016.objects.filter(Q(savings_secret_place__iexact='yes') | Q(savings_secret_place__iexact='y')).count()
        }
        
        print(f"Raw savings channel counts: {channels}")
        
        # Calculate percentages
        percentages = {k: round((v / total) * 100, 2) for k, v in channels.items()}
        
        print(f"Savings channel percentages: {percentages}")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#164e63', '#083344']
        }
        
        print(f"Final savings channels response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in savings_channels_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def digital_savings_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for digital savings: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        digital_savings = {
            'Mobile Banking': SurveyData2016.objects.filter(Q(savings_mobile_banking__iexact='yes') | Q(savings_mobile_banking__iexact='y')).count(),
            'Digital Wallet': SurveyData2016.objects.filter(Q(mobile_money_registered__iexact='yes') | Q(mobile_money_registered__iexact='y')).count(),
            'Credit Card': SurveyData2016.objects.filter(Q(credit_card__iexact='yes') | Q(credit_card__iexact='y')).count()
        }
        
        print(f"Raw digital savings counts: {digital_savings}")
        
        # Calculate percentages
        percentages = {k: round((v / total) * 100, 2) for k, v in digital_savings.items()}
        
        print(f"Digital savings percentages: {percentages}")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490']
        }
        
        print(f"Final digital savings response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in digital_savings_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def account_types_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for account types: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        accounts = {
            'Current Account': SurveyData2016.objects.filter(Q(bank_account_current__iexact='yes') | Q(bank_account_current__iexact='y')).count(),
            'Savings Account': SurveyData2016.objects.filter(Q(bank_account_savings__iexact='yes') | Q(bank_account_savings__iexact='y')).count(),
            'Everyday Account': SurveyData2016.objects.filter(Q(bank_account_everyday__iexact='yes') | Q(bank_account_everyday__iexact='y')).count(),
            'Postbank Account': SurveyData2016.objects.filter(Q(postbank_account__iexact='yes') | Q(postbank_account__iexact='y')).count()
        }
        
        print(f"Raw account type counts: {accounts}")
        
        # Calculate percentages
        percentages = {k: round((v / total) * 100, 2) for k, v in accounts.items()}
        
        print(f"Account type percentages: {percentages}")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490', '#155e75']
        }
        
        print(f"Final account types response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in account_types_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def banking_products_api(request):
    try:
        total = SurveyData2016.objects.count()
        print(f"Total respondents for banking products: {total}")
        
        if total == 0:
            return JsonResponse({'labels': [], 'data': [], 'colors': []})
            
        # Get counts with case-insensitive matching
        products = {
            'Debit Card': SurveyData2016.objects.filter(Q(debit_card__iexact='yes') | Q(debit_card__iexact='y')).count(),
            'Credit Card': SurveyData2016.objects.filter(Q(credit_card__iexact='yes') | Q(credit_card__iexact='y')).count(),
            'Overdraft': SurveyData2016.objects.filter(Q(bank_overdraft__iexact='yes') | Q(bank_overdraft__iexact='y')).count()
        }
        
        print(f"Raw banking product counts: {products}")
        
        # Calculate percentages
        percentages = {k: round((v / total) * 100, 2) for k, v in products.items()}
        
        print(f"Banking product percentages: {percentages}")
        
        data = {
            'labels': list(percentages.keys()),
            'data': list(percentages.values()),
            'colors': ['#06b6d4', '#0891b2', '#0e7490']
        }
        
        print(f"Final banking products response: {data}")
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in banking_products_api: {str(e)}")
        return JsonResponse({'labels': [], 'data': [], 'colors': []})

@require_GET
def pension_and_exclusion_stats_2016(request):
    """Return pension and financial exclusion statistics for 2016."""
    try:
        total_respondents = SurveyData2016.objects.count()
        
        # Get pension stats with case-insensitive matching
        nssf_users = SurveyData2016.objects.filter(pension_nssf__iexact='yes').count()
        mbao_users = SurveyData2016.objects.filter(pension_mbao__iexact='yes').count()
        other_pension = SurveyData2016.objects.filter(pension_other__iexact='yes').count()
        
        # Calculate percentages
        nssf_percent = round((nssf_users / total_respondents) * 100, 2)
        mbao_percent = round((mbao_users / total_respondents) * 100, 2)
        other_percent = round((other_pension / total_respondents) * 100, 2)
        
        # Get financial exclusion stats
        excluded = SurveyData2016.objects.filter(financially_excluded__iexact='yes').count()
        included = total_respondents - excluded
        
        # Calculate exclusion percentages
        excluded_percent = round((excluded / total_respondents) * 100, 2)
        included_percent = round(100 - excluded_percent, 2)
        
        # Log the data for debugging
        logger.info(f"Pension stats - NSSF: {nssf_percent}%, MBAO: {mbao_percent}%, Other: {other_percent}%")
        logger.info(f"Financial exclusion - Excluded: {excluded_percent}%, Included: {included_percent}%")
        
        response_data = {
            'pension': {
                'labels': ['NSSF', 'MBAO', 'Other Pension'],
                'data': [nssf_percent, mbao_percent, other_percent],
                'colors': ['#f43f5e', '#e11d48', '#be123c']
            },
            'exclusion': {
                'labels': ['Financially Included', 'Financially Excluded'],
                'data': [included_percent, excluded_percent],
                'colors': ['#ec4899', '#db2777']
            }
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in pension_and_exclusion_stats_2016: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def health_insurance_coverage_api(request):
    """Return health insurance coverage statistics."""
    try:
        total_respondents = SurveyData2016.objects.count()
        
        # Count users with different types of health insurance
        nhif_users = SurveyData2016.objects.filter(insurance_nhif__iexact='yes').count()
        other_health_insurance = SurveyData2016.objects.filter(insurance_health_other__iexact='yes').count()
        
        # Calculate percentages
        nhif_percent = round((nhif_users / total_respondents) * 100, 2)
        other_percent = round((other_health_insurance / total_respondents) * 100, 2)
        no_insurance_percent = round(100 - (nhif_percent + other_percent), 2)
        
        # Log the data for debugging
        logger.info(f"Health insurance coverage - NHIF: {nhif_percent}%, Other: {other_percent}%, None: {no_insurance_percent}%")
        logger.info(f"Raw counts - NHIF: {nhif_users}, Other: {other_health_insurance}, Total: {total_respondents}")
        
        response_data = {
            'labels': ['NHIF Coverage', 'Other Health Insurance', 'No Health Insurance'],
            'data': [nhif_percent, other_percent, no_insurance_percent],
            'colors': ['#06b6d4', '#0891b2', '#0e7490']
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error in health_insurance_coverage_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)





######################
#####################
##########################
###################