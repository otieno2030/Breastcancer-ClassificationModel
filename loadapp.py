import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

model_path = 'breast_cancer_model_v2.joblib'

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        rfc_selected = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Make sure the path is correct.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
