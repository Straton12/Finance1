import joblib
import os

# Get absolute path to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir)


def load_models():
    """Load models from the same directory as this script"""
    models = {
        'decision_tree': joblib.load(os.path.join(MODEL_DIR, 'decision_tree_model.pkl')),
        'voting': joblib.load(os.path.join(MODEL_DIR, 'voting_ensemble_model.pkl')),
        'selector': joblib.load(os.path.join(MODEL_DIR, 'feature_selector.pkl')),
        'features': joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
    }
    return models


ml_components = load_models()
