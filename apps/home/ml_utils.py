import joblib
from django.conf import settings
import os

def load_model():
    """Load the model only when needed"""
    model_path = os.path.join(settings.BASE_DIR, 'apps/home/ml_models/voting_ensemble_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_feature_selector():
    """Load the feature selector only when needed"""
    selector_path = os.path.join(settings.BASE_DIR, 'apps/home/ml_models/feature_selector.pkl')
    if os.path.exists(selector_path):
        return joblib.load(selector_path)
    return None

def load_selected_features():
    """Load the selected features only when needed"""
    features_path = os.path.join(settings.BASE_DIR, 'apps/home/ml_models/selected_features.pkl')
    if os.path.exists(features_path):
        return joblib.load(features_path)
    return [] 