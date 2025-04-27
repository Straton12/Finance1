# apps/home/load.py
import csv
import os
from django.core.management.base import BaseCommand
from home.models import Survey2016, Data2021  # Adjust based on your models

class Command(BaseCommand):
    help = 'Load datasets from CSV files into the database'

    def handle(self, *args, **kwargs):
        # Path to the datasets
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(base_dir, 'Datasets')

        # Load 2016 dataset
        with open(os.path.join(dataset_dir, '2016_data.csv'), 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                Survey2016.objects.create(
                    respondent_id=row['respondent_id'],
                    age=row['age'],
                    gender=row['gender'],
                    education_level=row['education_level'],
                    residence_type=row['residence_type'],
                    marital_status=row['marital_status'],
                    relationship_to_hh=row['relationship_to_hh'],
                    region=row['region'],
                    population_weight=row['population_weight'],
                    mobile_money_registered=row['mobile_money_registered'],
                    savings_mobile_banking=row['savings_mobile_banking'],
                    bank_account_current=row['bank_account_current'],
                    bank_account_savings=row['bank_account_savings'],
                    bank_account_everyday=row['bank_account_everyday'],
                    postbank_account=row['postbank_account'],
                    bank_overdraft=row['bank_overdraft'],
                    debit_card=row['debit_card'],
                    credit_card=row['credit_card'],
                    savings_microfinance=row['savings_microfinance'],
                    savings_sacco=row['savings_sacco'],
                    savings_group_friends=row['savings_group_friends'],
                    savings_family_friend=row['savings_family_friend'],
                    savings_secret_place=row['savings_secret_place'],
                    loan_bank=row['loan_bank'],
                    loan_mobile_banking=row['loan_mobile_banking'],
                    loan_sacco=row['loan_sacco'],
                    loan_microfinance=row['loan_microfinance'],
                    loan_shylock=row['loan_shylock'],
                    loan_group_chama=row['loan_group_chama'],
                    loan_govt=row['loan_govt'],
                    loan_employer=row['loan_employer'],
                    loan_family_friend=row['loan_family_friend'],
                    loan_shopkeeper_cash=row['loan_shopkeeper_cash'],
                    loan_goods_credit=row['loan_goods_credit'],
                    loan_digital_app=row['loan_digital_app'],
                    loan_agri_buyer_supplier=row['loan_agri_buyer_supplier'],
                    loan_hire_purchase=row['loan_hire_purchase'],
                    loan_mortgage=row['loan_mortgage'],
                    insurance_motor=row['insurance_motor'],
                    insurance_home=row['insurance_home'],
                    insurance_crop=row['insurance_crop'],
                    insurance_livestock=row['insurance_livestock'],
                    insurance_nhif=row['insurance_nhif'],
                    insurance_health_other=row['insurance_health_other'],
                    insurance_life=row['insurance_life'],
                    insurance_education=row['insurance_education'],
                    insurance_other=row['insurance_other'],
                    pension_nssf=row['pension_nssf'],
                    pension_mbao=row['pension_mbao'],
                    pension_other=row['pension_other'],
                    financially_excluded=row['financially_excluded']
                )
            self.stdout.write(self.style.SUCCESS('2016 dataset loaded successfully'))

        # Load 2021 dataset
        with open(os.path.join(dataset_dir, '2021_data.csv'), 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                Data2021.objects.create(
                    respondent_id=row['respondent_id'],
                    age=row['age'],
                    gender=row['gender'],
                    education_level=row['education_level'],
                    residence_type=row['residence_type'],
                    marital_status=row['marital_status'],
                    relationship_to_hh=row['relationship_to_hh'],
                    region=row['region'],
                    population_weight=row['population_weight'],
                    mobile_money_registered=row['mobile_money_registered'],
                    savings_mobile_banking=row['savings_mobile_banking'],
                    bank_account_current=row['bank_account_current'],
                    bank_account_savings=row['bank_account_savings'],
                    bank_account_everyday=row['bank_account_everyday'],
                    postbank_account=row['postbank_account'],
                    bank_overdraft=row['bank_overdraft'],
                    debit_card=row['debit_card'],
                    credit_card=row['credit_card'],
                    savings_microfinance=row['savings_microfinance'],
                    savings_sacco=row['savings_sacco'],
                    savings_group_friends=row['savings_group_friends'],
                    savings_family_friend=row['savings_family_friend'],
                    savings_secret_place=row['savings_secret_place'],
                    loan_bank=row['loan_bank'],
                    loan_mobile_banking=row['loan_mobile_banking'],
                    loan_sacco=row['loan_sacco'],
                    loan_microfinance=row['loan_microfinance'],
                    loan_shylock=row['loan_shylock'],
                    loan_group_chama=row['loan_group_chama'],
                    loan_govt=row['loan_govt'],
                    loan_employer=row['loan_employer'],
                    loan_family_friend=row['loan_family_friend'],
                    loan_shopkeeper_cash=row['loan_shopkeeper_cash'],
                    loan_goods_credit=row['loan_goods_credit'],
                    loan_digital_app=row['loan_digital_app'],
                    loan_agri_buyer_supplier=row['loan_agri_buyer_supplier'],
                    loan_hire_purchase=row['loan_hire_purchase'],
                    loan_mortgage=row['loan_mortgage'],
                    insurance_motor=row['insurance_motor'],
                    insurance_home=row['insurance_home'],
                    insurance_crop=row['insurance_crop'],
                    insurance_livestock=row['insurance_livestock'],
                    insurance_nhif=row['insurance_nhif'],
                    insurance_health_other=row['insurance_health_other'],
                    insurance_life=row['insurance_life'],
                    insurance_education=row['insurance_education'],
                    insurance_other=row['insurance_other'],
                    pension_nssf=row['pension_nssf'],
                    pension_mbao=row['pension_mbao'],
                    pension_other=row['pension_other'],
                    financially_excluded=row['financially_excluded']
                )
            self.stdout.write(self.style.SUCCESS('2021 dataset loaded successfully'))