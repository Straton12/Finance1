

from rest_framework import serializers
from .models import SurveyData2016, SurveyData2021

class SurveyData2016Serializer(serializers.ModelSerializer):
    class Meta:
        model = SurveyData2016
        fields = '__all__'  # Include all fields

class SurveyData2021Serializer(serializers.ModelSerializer):
    class Meta:
        model = SurveyData2021
        fields = '__all__'  # Include all fields