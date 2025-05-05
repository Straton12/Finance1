from django.core.management.base import BaseCommand
from apps.home.models import SurveyData2016

class Command(BaseCommand):
    help = 'Check data values in SurveyData2016'

    def handle(self, *args, **kwargs):
        # Check savings_mobile_banking values
        savings_values = SurveyData2016.objects.values_list('savings_mobile_banking', flat=True).distinct()
        self.stdout.write('Unique values for savings_mobile_banking:')
        for value in savings_values:
            self.stdout.write(f'- {value}')
            self.stdout.write(f'  Count: {SurveyData2016.objects.filter(savings_mobile_banking=value).count()}')

        # Check loan_bank values
        loan_values = SurveyData2016.objects.values_list('loan_bank', flat=True).distinct()
        self.stdout.write('\nUnique values for loan_bank:')
        for value in loan_values:
            self.stdout.write(f'- {value}')
            self.stdout.write(f'  Count: {SurveyData2016.objects.filter(loan_bank=value).count()}') 