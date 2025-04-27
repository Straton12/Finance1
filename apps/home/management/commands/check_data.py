from django.core.management.base import BaseCommand
from apps.home.models import SurveyData2016

class Command(BaseCommand):
    help = 'Check data in SurveyData2016 table'

    def handle(self, *args, **kwargs):
        total_records = SurveyData2016.objects.count()
        self.stdout.write(f'Total records in SurveyData2016: {total_records}')
        
        if total_records > 0:
            # Sample some data
            sample = SurveyData2016.objects.all()[:5]
            self.stdout.write('\nSample records:')
            for record in sample:
                self.stdout.write(f'ID: {record.id}')
                self.stdout.write(f'Age: {record.age}')
                self.stdout.write(f'Gender: {record.gender}')
                self.stdout.write(f'Education: {record.education_level}')
                self.stdout.write(f'Residence: {record.residence_type}')
                self.stdout.write('---') 