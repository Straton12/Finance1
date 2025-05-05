import csv
from django.core.management.base import BaseCommand
from home.models import SurveyData2016


class Command(BaseCommand):
    help = 'Import 2016 survey data from CSV'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        with open(csv_file, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader, 1):
                try:
                    SurveyData2016.objects.create(
                        respondent_id=row.get('respondent_id'),
                        age=int(row['age']) if row.get('age') else None,
                        gender=row.get('gender'),
                        education_level=row.get('education_level'),
                        residence_type=row.get('residence_type'),
                        marital_status=row.get('marital_status'),
                        relationship_to_hh=row.get('relationship_to_hh'),
                        region=row.get('region'),
                        population_weight=float(row['population_weight']) if row.get(
                            'population_weight') else None,
                        mobile_money_registered=row.get(
                            'mobile_money_registered'),
                        savings_mobile_banking=row.get(
                            'savings_mobile_banking'),
                        bank_account_current=row.get('bank_account_current'),
                        bank_account_savings=row.get('bank_account_savings'),
                        bank_account_everyday=row.get('bank_account_everyday'),
                        postbank_account=row.get('postbank_account'),
                        bank_overdraft=row.get('bank_overdraft'),
                        debit_card=row.get('debit_card'),
                        credit_card=row.get('credit_card'),
                        savings_microfinance=row.get('savings_microfinance'),
                        savings_sacco=row.get('savings_sacco'),
                        savings_group_friends=row.get('savings_group_friends'),
                        savings_family_friend=row.get('savings_family_friend'),
                        savings_secret_place=row.get('savings_secret_place'),
                        loan_bank=row.get('loan_bank'),
                        loan_mobile_banking=row.get('loan_mobile_banking'),
                        loan_sacco=row.get('loan_sacco'),
                        loan_microfinance=row.get('loan_microfinance'),
                        loan_shylock=row.get('loan_shylock'),
                        loan_group_chama=row.get('loan_group_chama'),
                        loan_govt=row.get('loan_govt'),
                        loan_employer=row.get('loan_employer'),
                        loan_family_friend=row.get('loan_family_friend'),
                        loan_shopkeeper_cash=row.get('loan_shopkeeper_cash'),
                        loan_goods_credit=row.get('loan_goods_credit'),
                        loan_digital_app=row.get('loan_digital_app'),
                        loan_agri_buyer_supplier=row.get(
                            'loan_agri_buyer_supplier'),
                        loan_hire_purchase=row.get('loan_hire_purchase'),
                        loan_mortgage=row.get('loan_mortgage'),
                        insurance_motor=row.get('insurance_motor'),
                        insurance_home=row.get('insurance_home'),
                        insurance_crop=row.get('insurance_crop'),
                        insurance_livestock=row.get('insurance_livestock'),
                        insurance_nhif=row.get('insurance_nhif'),
                        insurance_health_other=row.get(
                            'insurance_health_other'),
                        insurance_life=row.get('insurance_life'),
                        insurance_education=row.get('insurance_education'),
                        insurance_other=row.get('insurance_other'),
                        pension_nssf=row.get('pension_nssf'),
                        pension_mbao=row.get('pension_mbao'),
                        pension_other=row.get('pension_other'),
                        financially_excluded=row.get('financially_excluded')
                    )
                    if i % 100 == 0:
                        self.stdout.write(f"Processed {i} records...")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f"Error on row {i}: {str(e)}"))

        self.stdout.write(self.style.SUCCESS(
            f'Successfully imported {i} records from {csv_file}'))
