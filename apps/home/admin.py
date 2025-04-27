# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin
from django.apps import apps

# Register your models here.
# Get all models from the 'home' app
app_models = apps.get_app_config('home').get_models()

# Register each model dynamically
for model in app_models:
    admin.site.register(model)
