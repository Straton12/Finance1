from django.core.management.base import BaseCommand
from apps.home.models import SurveyData2021

class Command(BaseCommand):
    help = 'Check savings-related data in SurveyData2021'

    def handle(self, *args, **kwargs):
        # Get total count
        total = SurveyData2021.objects.count()
        self.stdout.write(f'Total records in SurveyData2021: {total}')

        # Check savings_mobile_banking values
        mobile_banking = SurveyData2021.objects.filter(savings_mobile_banking='Yes').count()
        self.stdout.write(f'Mobile banking savings count: {mobile_banking}')

        # Check mobile_money_registered values
        digital_wallet = SurveyData2021.objects.filter(mobile_money_registered='Yes').count()
        self.stdout.write(f'Digital wallet count: {digital_wallet}')

        # Check credit_card values
        credit_card = SurveyData2021.objects.filter(credit_card='Yes').count()
        self.stdout.write(f'Credit card count: {credit_card}')

        # Check other savings fields
        microfinance = SurveyData2021.objects.filter(savings_microfinance='Yes').count()
        sacco = SurveyData2021.objects.filter(savings_sacco='Yes').count()
        group_friends = SurveyData2021.objects.filter(savings_group_friends='Yes').count()
        family_friend = SurveyData2021.objects.filter(savings_family_friend='Yes').count()
        secret_place = SurveyData2021.objects.filter(savings_secret_place='Yes').count()

        self.stdout.write(f'Microfinance savings count: {microfinance}')
        self.stdout.write(f'SACCO savings count: {sacco}')
        self.stdout.write(f'Group/Friends savings count: {group_friends}')
        self.stdout.write(f'Family/Friend savings count: {family_friend}')
        self.stdout.write(f'Secret place savings count: {secret_place}') 