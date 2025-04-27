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
        age_labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
        
        hist, _ = np.histogram(age_data, bins=age_bins)
        total = len(age_data)
        
        # Convert counts to percentages
        percentages = [round((count / total) * 100, 2) if total > 0 else 0 for count in hist]
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
        # Get total records
        total_records = SurveyData2016.objects.count()
        logger.debug(f"Total records: {total_records}")

        if total_records == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get gender distribution
        gender_counts = SurveyData2016.objects.values('gender').annotate(count=Count('id'))
        
        # Convert to percentages
        gender_data = {
            'labels': [],
            'data': []
        }
        
        for item in gender_counts:
            if item['gender']:  # Skip null values
                percentage = (item['count'] / total_records) * 100
                gender_data['labels'].append(item['gender'])
                gender_data['data'].append(round(percentage, 2))
        
        logger.debug(f"Gender distribution data: {gender_data}")
        return JsonResponse(gender_data)
    
    except Exception as e:
        logger.error(f"Error in gender_distribution: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def digital_vs_traditional_loans_api(request):
    try:
        # Get total respondents excluding null responses for all loan types
        total_respondents = SurveyData2021.objects.exclude(
            Q(loan_digital_app__isnull=True) &
            Q(loan_mobile_banking__isnull=True) &
            Q(loan_bank__isnull=True)
        ).count()
        
        logger.debug(f"Total respondents (excluding nulls): {total_respondents}")
        
        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Count responses for each loan type
        digital_loans = SurveyData2021.objects.filter(
            loan_digital_app__iexact='yes'
        ).count()
        
        mobile_banking_loans = SurveyData2021.objects.filter(
            loan_mobile_banking__iexact='yes'
        ).count()
        
        traditional_bank_loans = SurveyData2021.objects.filter(
            loan_bank__iexact='yes'
        ).count()

        logger.debug(f"Digital loans: {digital_loans}")
        logger.debug(f"Mobile banking loans: {mobile_banking_loans}")
        logger.debug(f"Traditional bank loans: {traditional_bank_loans}")

        # Calculate percentages
        digital_percentage = round((digital_loans / total_respondents) * 100, 2)
        mobile_percentage = round((mobile_banking_loans / total_respondents) * 100, 2)
        traditional_percentage = round((traditional_bank_loans / total_respondents) * 100, 2)

        response_data = {
            'labels': ['Digital App Loans', 'Mobile Banking Loans', 'Traditional Bank Loans'],
            'data': [digital_percentage, mobile_percentage, traditional_percentage],
            'colors': ['#FF6384', '#36A2EB', '#FFCE56']
        }

        logger.debug(f"Response data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in digital_vs_traditional_loans_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def financial_exclusion_api(request):
    try:
        # Get total records
        total_records = SurveyData2016.objects.count()
        logger.debug(f"Total records: {total_records}")

        if total_records == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get financial exclusion counts
        excluded_count = SurveyData2016.objects.filter(financially_excluded=True).count()
        included_count = total_records - excluded_count

        # Calculate percentages
        excluded_percentage = round((excluded_count / total_records) * 100, 2)
        included_percentage = round((included_count / total_records) * 100, 2)

        response_data = {
            'labels': ['Financially Excluded', 'Financially Included'],
            'data': [excluded_percentage, included_percentage],
            'colors': ['#FF6384', '#36A2EB']
        }

        logger.debug(f"Financial exclusion data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in financial_exclusion_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def informal_lending_sources_api(request):
    try:
        # Get total respondents
        total_respondents = SurveyData2021.objects.count()
        logger.debug(f"Total respondents: {total_respondents}")

        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get counts for each informal lending source
        chama_count = SurveyData2021.objects.filter(loan_chama__iexact='yes').count()
        family_count = SurveyData2021.objects.filter(loan_family__iexact='yes').count()
        employer_count = SurveyData2021.objects.filter(loan_employer__iexact='yes').count()
        shylock_count = SurveyData2021.objects.filter(loan_shylock__iexact='yes').count()

        # Calculate percentages
        chama_percentage = round((chama_count / total_respondents) * 100, 2)
        family_percentage = round((family_count / total_respondents) * 100, 2)
        employer_percentage = round((employer_count / total_respondents) * 100, 2)
        shylock_percentage = round((shylock_count / total_respondents) * 100, 2)

        response_data = {
            'labels': ['Chama Loans', 'Family/Friends Loans', 'Employer Loans', 'Shylock Loans'],
            'data': [chama_percentage, family_percentage, employer_percentage, shylock_percentage],
            'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
        }

        logger.debug(f"Informal lending sources data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in informal_lending_sources_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def pension_types_api(request):
    try:
        # Get total respondents
        total_respondents = SurveyData2021.objects.count()
        logger.debug(f"Total respondents: {total_respondents}")

        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get counts for each pension type
        nssf_count = SurveyData2021.objects.filter(pension_nssf__iexact='yes').count()
        private_count = SurveyData2021.objects.filter(pension_private__iexact='yes').count()
        employer_count = SurveyData2021.objects.filter(pension_employer__iexact='yes').count()

        # Calculate percentages
        nssf_percentage = round((nssf_count / total_respondents) * 100, 2)
        private_percentage = round((private_count / total_respondents) * 100, 2)
        employer_percentage = round((employer_count / total_respondents) * 100, 2)

        response_data = {
            'labels': ['NSSF Pension', 'Private Pension', 'Employer Pension'],
            'data': [nssf_percentage, private_percentage, employer_percentage],
            'colors': ['#FF6384', '#36A2EB', '#FFCE56']
        }

        logger.debug(f"Pension types data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in pension_types_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def savings_behavior_api(request):
    try:
        # Get total respondents
        total_respondents = SurveyData2021.objects.count()
        logger.debug(f"Total respondents: {total_respondents}")

        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get counts for each savings type
        bank_count = SurveyData2021.objects.filter(savings_bank__iexact='yes').count()
        mobile_count = SurveyData2021.objects.filter(savings_mobile__iexact='yes').count()
        chama_count = SurveyData2021.objects.filter(savings_chama__iexact='yes').count()
        home_count = SurveyData2021.objects.filter(savings_home__iexact='yes').count()

        # Calculate percentages
        bank_percentage = round((bank_count / total_respondents) * 100, 2)
        mobile_percentage = round((mobile_count / total_respondents) * 100, 2)
        chama_percentage = round((chama_count / total_respondents) * 100, 2)
        home_percentage = round((home_count / total_respondents) * 100, 2)

        response_data = {
            'labels': ['Bank Savings', 'Mobile Money Savings', 'Chama Savings', 'Home Savings'],
            'data': [bank_percentage, mobile_percentage, chama_percentage, home_percentage],
            'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
        }

        logger.debug(f"Savings behavior data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in savings_behavior_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def insurance_coverage_api(request):
    try:
        # Get total respondents
        total_respondents = SurveyData2021.objects.count()
        logger.debug(f"Total respondents: {total_respondents}")

        if total_respondents == 0:
            return JsonResponse({'error': 'No data available'}, status=404)

        # Get counts for each insurance type
        health_count = SurveyData2021.objects.filter(insurance_health__iexact='yes').count()
        life_count = SurveyData2021.objects.filter(insurance_life__iexact='yes').count()
        property_count = SurveyData2021.objects.filter(insurance_property__iexact='yes').count()

        # Calculate percentages
        health_percentage = round((health_count / total_respondents) * 100, 2)
        life_percentage = round((life_count / total_respondents) * 100, 2)
        property_percentage = round((property_count / total_respondents) * 100, 2)

        response_data = {
            'labels': ['Health Insurance', 'Life Insurance', 'Property Insurance'],
            'data': [health_percentage, life_percentage, property_percentage],
            'colors': ['#FF6384', '#36A2EB', '#FFCE56']
        }

        logger.debug(f"Insurance coverage data: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in insurance_coverage_api: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500) 