# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path, include
from apps.home import views
from .views import age_distribution, gender_distribution, survey_data_view, residence_type_distribution, age_distribution1, gender_distribution1, residence_type_distribution1, predict_view
from rest_framework.routers import DefaultRouter
from .views import SurveyData2016ViewSet, SurveyData2021ViewSet


# Create a router and register the viewsets
router = DefaultRouter()
router.register(r'survey-2016', SurveyData2016ViewSet)
router.register(r'survey-2021', SurveyData2021ViewSet)


urlpatterns = [
    path('', views.index, name='home'),
    path('api/age-distribution/', views.age_distribution, name='age_distribution'),
    path('api/age-distribution1/', views.age_distribution1, name='age_distribution1'),
    path('api/gender-distribution/', views.gender_distribution, name='gender-distribution'),
    path('api/gender-distribution1/', views.gender_distribution1, name='gender-distribution1'),
    path('api/education-distribution/', views.education_distribution, name='education-distribution'),
    path('api/education-distribution1/', views.education_distribution1, name='education-distribution1'),
    path('survey-data/', views.survey_data_view, name='survey_data'),
    path('ui-tables/', views.ui_tables, name='ui-tables'),
    path('predict/', views.predict_view, name='predict'),
    path('api/financial-stats/', views.financial_stats, name='financial_stats'),
    path('api/financial-stats1/', views.financial_stats1, name='financial_stats1'),
    path('', include(router.urls)),
    path('api/residence-type-distribution/', residence_type_distribution, name='residence-type-distribution'),
    path('api/residence-type-distribution1/', residence_type_distribution1, name='residence-type-distribution1'),
    path('api/pension-exclusion-stats/', views.pension_and_exclusion_stats, name='pension_exclusion_stats'),
    path('api/pension-exclusion-stats1/', views.pension_and_exclusion_stats1, name='pension_exclusion_stats1'),
    path('api/survey-data-2016/', views.survey_data_2016_table, name='survey_data_2016_table'),
    path('api/survey-data-2021/', views.survey_data_2021_table, name='survey_data_2021_table'),
    
    # Insurance API endpoints
    path('api/insurance-types/', views.insurance_types_api, name='insurance_types_api'),
    path('api/health-insurance/', views.health_insurance_api, name='health_insurance_api'),
    path('api/asset-insurance/', views.asset_insurance_api, name='asset_insurance_api'),
    path('api/agri-insurance/', views.agri_insurance_api, name='agri_insurance_api'),
    path('api/insurance-types-2021/', views.insurance_types_2021_api, name='insurance_types_2021_api'),
    path('api/debug-insurance/', views.debug_insurance_data, name='debug_insurance_data'),
    
    # Other API endpoints
    path('api/banking-services/', views.banking_services_api, name='banking_services_api'),
    path('api/digital-services/', views.digital_services_api, name='digital_services_api'),
    path('api/loan-sources/', views.loan_sources_api, name='loan_sources_api'),
    path('api/savings-methods/', views.savings_methods_api, name='savings_methods_api'),
    path('api/credit-types/', views.credit_types_api, name='credit_types_api'),
    path('api/digital-vs-traditional-loans/', views.digital_vs_traditional_loans_api, name='digital_vs_traditional_loans_api'),
    path('api/informal-lending/', views.informal_lending_api, name='informal_lending_api'),
    
    # Add new savings behavior endpoints
    path('api/savings-channels/', views.savings_channels_api, name='savings_channels_api'),
    path('api/digital-savings/', views.digital_savings_api, name='digital_savings_api'),
    
    # This should be the last pattern as it's a catch-all
    re_path(r'^.*\.*', views.pages, name='pages'),
] + router.urls
