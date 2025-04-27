# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models
from django.contrib.auth.models import User


class SurveyData2016(models.Model):
    respondent_id = models.BigIntegerField(primary_key=True)
    age = models.IntegerField(null=True, blank=True)  # Allow null/blank
    gender = models.CharField(max_length=10, null=True, blank=True)
    education_level = models.CharField(max_length=50, null=True, blank=True)
    residence_type = models.CharField(max_length=20, null=True, blank=True)
    marital_status = models.CharField(max_length=50, null=True, blank=True)
    relationship_to_hh = models.CharField(max_length=50, null=True, blank=True)
    region = models.CharField(max_length=50, null=True, blank=True)
    population_weight = models.FloatField(null=True, blank=True)
    mobile_money_registered = models.CharField(max_length=10, null=True, blank=True)
    savings_mobile_banking = models.CharField(max_length=10, null=True, blank=True)
    bank_account_current = models.CharField(max_length=10, null=True, blank=True)
    bank_account_savings = models.CharField(max_length=10, null=True, blank=True)
    bank_account_everyday = models.CharField(max_length=10, null=True, blank=True)
    postbank_account = models.CharField(max_length=10, null=True, blank=True)
    bank_overdraft = models.CharField(max_length=10, null=True, blank=True)
    debit_card = models.CharField(max_length=10, null=True, blank=True)
    credit_card = models.CharField(max_length=10, null=True, blank=True)
    savings_microfinance = models.CharField(max_length=10, null=True, blank=True)
    savings_sacco = models.CharField(max_length=10, null=True, blank=True)
    savings_group_friends = models.CharField(max_length=10, null=True, blank=True)
    savings_family_friend = models.CharField(max_length=10, null=True, blank=True)
    savings_secret_place = models.CharField(max_length=10, null=True, blank=True)
    loan_bank = models.CharField(max_length=10, null=True, blank=True)
    loan_mobile_banking = models.CharField(max_length=10, null=True, blank=True)
    loan_sacco = models.CharField(max_length=10, null=True, blank=True)
    loan_microfinance = models.CharField(max_length=10, null=True, blank=True)
    loan_shylock = models.CharField(max_length=10, null=True, blank=True)
    loan_group_chama = models.CharField(max_length=10, null=True, blank=True)
    loan_govt = models.CharField(max_length=10, null=True, blank=True)
    loan_employer = models.CharField(max_length=10, null=True, blank=True)
    loan_family_friend = models.CharField(max_length=10, null=True, blank=True)
    loan_shopkeeper_cash = models.CharField(max_length=10, null=True, blank=True)
    loan_goods_credit = models.CharField(max_length=10, null=True, blank=True)
    loan_digital_app = models.CharField(max_length=10, null=True, blank=True)
    loan_agri_buyer_supplier = models.CharField(max_length=10, null=True, blank=True)
    loan_hire_purchase = models.CharField(max_length=10, null=True, blank=True)
    loan_mortgage = models.CharField(max_length=10, null=True, blank=True)
    insurance_motor = models.CharField(max_length=10, null=True, blank=True)
    insurance_home = models.CharField(max_length=10, null=True, blank=True)
    insurance_crop = models.CharField(max_length=10, null=True, blank=True)
    insurance_livestock = models.CharField(max_length=10, null=True, blank=True)
    insurance_nhif = models.CharField(max_length=10, null=True, blank=True)
    insurance_health_other = models.CharField(max_length=10, null=True, blank=True)
    insurance_life = models.CharField(max_length=10, null=True, blank=True)
    insurance_education = models.CharField(max_length=10, null=True, blank=True)
    insurance_other = models.CharField(max_length=10, null=True, blank=True)
    pension_nssf = models.CharField(max_length=10, null=True, blank=True)
    pension_mbao = models.CharField(max_length=10, null=True, blank=True)
    pension_other = models.CharField(max_length=10, null=True, blank=True)
    financially_excluded = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"Respondent {self.respondent_id}"


class SurveyData2021(models.Model):
    respondent_id = models.BigIntegerField(primary_key=True)
    age = models.IntegerField(null=True, blank=True)  # Allow null/blank
    gender = models.CharField(max_length=10, null=True, blank=True)
    education_level = models.CharField(max_length=50, null=True, blank=True)
    residence_type = models.CharField(max_length=20, null=True, blank=True)
    marital_status = models.CharField(max_length=50, null=True, blank=True)
    relationship_to_hh = models.CharField(max_length=50, null=True, blank=True)
    region = models.CharField(max_length=50, null=True, blank=True)
    population_weight = models.FloatField(null=True, blank=True)
    mobile_money_registered = models.CharField(max_length=10, null=True, blank=True)
    savings_mobile_banking = models.CharField(max_length=10, null=True, blank=True)
    bank_account_current = models.CharField(max_length=10, null=True, blank=True)
    bank_account_savings = models.CharField(max_length=10, null=True, blank=True)
    bank_account_everyday = models.CharField(max_length=10, null=True, blank=True)
    postbank_account = models.CharField(max_length=10, null=True, blank=True)
    bank_overdraft = models.CharField(max_length=10, null=True, blank=True)
    debit_card = models.CharField(max_length=10, null=True, blank=True)
    credit_card = models.CharField(max_length=10, null=True, blank=True)
    savings_microfinance = models.CharField(max_length=10, null=True, blank=True)
    savings_sacco = models.CharField(max_length=10, null=True, blank=True)
    savings_group_friends = models.CharField(max_length=10, null=True, blank=True)
    savings_family_friend = models.CharField(max_length=10, null=True, blank=True)
    savings_secret_place = models.CharField(max_length=10, null=True, blank=True)
    loan_bank = models.CharField(max_length=10, null=True, blank=True)
    loan_mobile_banking = models.CharField(max_length=10, null=True, blank=True)
    loan_sacco = models.CharField(max_length=10, null=True, blank=True)
    loan_microfinance = models.CharField(max_length=10, null=True, blank=True)
    loan_shylock = models.CharField(max_length=10, null=True, blank=True)
    loan_group_chama = models.CharField(max_length=10, null=True, blank=True)
    loan_govt = models.CharField(max_length=10, null=True, blank=True)
    loan_employer = models.CharField(max_length=10, null=True, blank=True)
    loan_family_friend = models.CharField(max_length=10, null=True, blank=True)
    loan_shopkeeper_cash = models.CharField(max_length=10, null=True, blank=True)
    loan_goods_credit = models.CharField(max_length=10, null=True, blank=True)
    loan_digital_app = models.CharField(max_length=10, null=True, blank=True)
    loan_agri_buyer_supplier = models.CharField(max_length=10, null=True, blank=True)
    loan_hire_purchase = models.CharField(max_length=10, null=True, blank=True)
    loan_mortgage = models.CharField(max_length=10, null=True, blank=True)
    insurance_motor = models.CharField(max_length=10, null=True, blank=True)
    insurance_home = models.CharField(max_length=10, null=True, blank=True)
    insurance_crop = models.CharField(max_length=10, null=True, blank=True)
    insurance_livestock = models.CharField(max_length=10, null=True, blank=True)
    insurance_nhif = models.CharField(max_length=10, null=True, blank=True)
    insurance_health_other = models.CharField(max_length=10, null=True, blank=True)
    insurance_life = models.CharField(max_length=10, null=True, blank=True)
    insurance_education = models.CharField(max_length=10, null=True, blank=True)
    insurance_other = models.CharField(max_length=10, null=True, blank=True)
    pension_nssf = models.CharField(max_length=10, null=True, blank=True)
    pension_mbao = models.CharField(max_length=10, null=True, blank=True)
    pension_other = models.CharField(max_length=10, null=True, blank=True)
    financially_excluded = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"Respondent {self.respondent_id}"
    
    
class Prediction(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    education_level = models.CharField(max_length=20)
    residence_type = models.CharField(max_length=10)
    prediction = models.BooleanField()
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.age} year old - {'Excluded' if self.prediction else 'Included'}"
