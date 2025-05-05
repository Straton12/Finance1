import csv
from django.core.management.base import BaseCommand
from apps.home.models import SurveyData2021
from django.db import transaction


class Command(BaseCommand):
    help = 'Import 2021 survey data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        count = 0
        next_id = 1  # Start with ID 1

        try:
            with transaction.atomic():
                # Clear existing data
                SurveyData2021.objects.all().delete()
                
                with open(csv_file, 'r') as file:
                    csv_reader = csv.DictReader(file)
                    for row in csv_reader:
                        try:
                            # Use next_id if respondent_id is empty
                            respondent_id = int(row['respondent_id']) if row['respondent_id'].strip() else next_id
                            next_id += 1

                            survey_data = SurveyData2021(
                                respondent_id=respondent_id,
                                age=int(row['age']) if row['age'].strip() else None,
                                gender=row['gender'].lower(),
                                education_level=row['education_level'].lower(),
                                residence_type=row['residence_type'].lower(),
                                marital_status=row['marital_status'].lower(),
                                relationship_to_hh=row['relationship_to_hh'].lower(),
                                region=row['region'].lower(),
                                population_weight=float(row['population_weight']) if row['population_weight'].strip() else 0,
                                mobile_money_registered=row['mobile_money_registered'].lower() == 'yes',
                                savings_mobile_banking=row['savings_mobile_banking'].lower() == 'yes',
                                bank_account_current=row['bank_account_current'].lower() == 'yes',
                                bank_account_savings=row['bank_account_savings'].lower() == 'yes',
                                bank_account_everyday=row['bank_account_everyday'].lower() == 'yes',
                                postbank_account=row['postbank_account'].lower() == 'yes',
                                bank_overdraft=row['bank_overdraft'].lower() == 'yes',
                                debit_card=row['debit_card'].lower() == 'yes',
                                credit_card=row['credit_card'].lower() == 'yes',
                                savings_microfinance=row['savings_microfinance'].lower() == 'yes',
                                savings_sacco=row['savings_sacco'].lower() == 'yes',
                                savings_group_friends=row['savings_group_friends'].lower() == 'yes',
                                savings_family_friend=row['savings_family_friend'].lower() == 'yes',
                                savings_secret_place=row['savings_secret_place'].lower() == 'yes',
                                loan_bank=row['loan_bank'].lower() == 'yes',
                                loan_mobile_banking=row['loan_mobile_banking'].lower() == 'yes',
                                loan_sacco=row['loan_sacco'].lower() == 'yes',
                                loan_microfinance=row['loan_microfinance'].lower() == 'yes',
                                loan_shylock=row['loan_shylock'].lower() == 'yes',
                                loan_group_chama=row['loan_group_chama'].lower() == 'yes',
                                loan_govt=row['loan_govt'].lower() == 'yes',
                                loan_employer=row['loan_employer'].lower() == 'yes',
                                loan_family_friend=row['loan_family_friend'].lower() == 'yes',
                                loan_shopkeeper_cash=row['loan_shopkeeper_cash'].lower() == 'yes',
                                loan_goods_credit=row['loan_goods_credit'].lower() == 'yes',
                                loan_digital_app=row['loan_digital_app'].lower() == 'yes',
                                loan_agri_buyer_supplier=row['loan_agri_buyer_supplier'].lower() == 'yes',
                                loan_hire_purchase=row['loan_hire_purchase'].lower() == 'yes',
                                loan_mortgage=row['loan_mortgage'].lower() == 'yes',
                                insurance_motor=row['insurance_motor'].lower() == 'yes',
                                insurance_home=row['insurance_home'].lower() == 'yes',
                                insurance_crop=row['insurance_crop'].lower() == 'yes',
                                insurance_livestock=row['insurance_livestock'].lower() == 'yes',
                                insurance_nhif=row['insurance_nhif'].lower() == 'yes',
                                insurance_health_other=row['insurance_health_other'].lower() == 'yes',
                                insurance_life=row['insurance_life'].lower() == 'yes',
                                insurance_education=row['insurance_education'].lower() == 'yes',
                                insurance_other=row['insurance_other'].lower() == 'yes',
                                pension_nssf=row['pension_nssf'].lower() == 'yes',
                                pension_mbao=row['pension_mbao'].lower() == 'yes',
                                pension_other=row['pension_other'].lower() == 'yes',
                                financially_excluded=row['financially_excluded'].lower() == 'yes'
                            )
                            survey_data.save()
                            count += 1
                            if count % 1000 == 0:  # Print progress every 1000 records
                                self.stdout.write(f"Imported {count} records...")
                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f'Error loading row {count + 1}: {str(e)}'))
                            raise

            self.stdout.write(self.style.SUCCESS(f'Successfully imported {count} records from 2021 survey data'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to import data: {str(e)}'))
            raise
