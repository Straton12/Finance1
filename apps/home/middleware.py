# Create a new file middleware.py
from django.utils.deprecation import MiddlewareMixin
import joblib

class ModelLoaderMiddleware(MiddlewareMixin):
    def __init__(self, get_response):
        super().__init__(get_response)
        self.models_loaded = False
        
    def __call__(self, request):
        if not self.models_loaded:
            try:
                # Load models once
                request.model = joblib.load('home/models/decision_tree_model.pkl')
                request.feature_selector = joblib.load('home/models/feature_selector.pkl')
                request.selected_features = joblib.load('home/models/selected_features.pkl')
                self.models_loaded = True
            except Exception as e:
                print(f"Error loading models: {e}")
        
        return self.get_response(request)