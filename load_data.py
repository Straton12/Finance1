import os
import django
import pandas as pd
import numpy as np

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from apps.home.models import SurveyData2016, SurveyData2021

def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return None
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return value

def load_2016_data():
    print("Loading 2016 data...")
    df = pd.read_csv('2016_data.csv')
    for _, row in df.iterrows():
        if pd.isna(row['respondent_id']):
            continue
        data = {col: clean_value(row[col]) for col in row.index}
        try:
            SurveyData2016.objects.create(**data)
        except Exception as e:
            print(f"Error loading row with respondent_id {row['respondent_id']}: {str(e)}")
            continue
    print("2016 data loaded successfully!")

def load_2021_data():
    print("Loading 2021 data...")
    df = pd.read_csv('2021_data.csv')
    for _, row in df.iterrows():
        if pd.isna(row['respondent_id']):
            continue
        data = {col: clean_value(row[col]) for col in row.index}
        try:
            SurveyData2021.objects.create(**data)
        except Exception as e:
            print(f"Error loading row with respondent_id {row['respondent_id']}: {str(e)}")
            continue
    print("2021 data loaded successfully!")

if __name__ == '__main__':
    # Clear existing data
    SurveyData2016.objects.all().delete()
    SurveyData2021.objects.all().delete()
    
    # Load new data
    load_2016_data()
    load_2021_data() 