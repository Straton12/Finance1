from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from apps.home.models import SurveyData2016, SurveyData2021
from django.db import transaction

class Command(BaseCommand):
    help = 'Import survey data from CSV files into the database'

    def handle(self, *args, **kwargs):
        # Import 2016 data
        try:
            self.stdout.write('Importing 2016 data...')
            df_2016 = pd.read_csv('2016_data.csv')
            
            # Convert 'yes'/'no' to boolean strings
            for col in df_2016.columns:
                if df_2016[col].dtype == object:
                    df_2016[col] = df_2016[col].str.lower()
                    df_2016[col] = df_2016[col].map({'yes': 'yes', 'no': 'no'})

            # Bulk create records
            with transaction.atomic():
                SurveyData2016.objects.all().delete()  # Clear existing data
                records_2016 = []
                for _, row in df_2016.iterrows():
                    record = SurveyData2016(
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
                    records_2016.append(record)
                
                SurveyData2016.objects.bulk_create(records_2016, batch_size=1000)
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {len(records_2016)} records from 2016 data'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error importing 2016 data: {str(e)}'))

        # Import 2021 data
        try:
            self.stdout.write('Importing 2021 data...')
            df_2021 = pd.read_csv('2021_data.csv')
            
            self.stdout.write(f'Initial 2021 data shape: {df_2021.shape}')
            self.stdout.write(f'Columns in 2021 data: {df_2021.columns.tolist()}')
            
            # Generate sequential IDs for 2021 data
            # Start from a high number to avoid conflicts with 2016 IDs
            start_id = 1000000
            df_2021['respondent_id'] = range(start_id, start_id + len(df_2021))
            
            # Fill NaN values with appropriate defaults
            df_2021 = df_2021.fillna({
                'age': 0,
                'gender': 'unknown',
                'education_level': 'unknown',
                'residence_type': 'unknown',
                'marital_status': 'unknown',
                'relationship_to_hh': 'unknown',
                'region': 'unknown',
                'population_weight': 0
            })
            
            # Fill boolean columns with 'no'
            bool_columns = [col for col in df_2021.columns if col not in ['respondent_id', 'age', 'gender', 'education_level', 'residence_type', 'marital_status', 'relationship_to_hh', 'region', 'population_weight']]
            df_2021[bool_columns] = df_2021[bool_columns].fillna('no')

            # Bulk create records
            with transaction.atomic():
                SurveyData2021.objects.all().delete()  # Clear existing data
                records_2021 = []
                for _, row in df_2021.iterrows():
                    record = SurveyData2021(
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
                    records_2021.append(record)
                
                SurveyData2021.objects.bulk_create(records_2021, batch_size=1000)
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {len(records_2021)} records from 2021 data'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error importing 2021 data: {str(e)}')) 