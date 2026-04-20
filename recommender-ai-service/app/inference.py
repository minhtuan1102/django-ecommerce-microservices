import os
import numpy as np
from django.conf import settings

# Load TensorFlow only when needed to save memory in microservices
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            model_path = os.path.join(settings.BASE_DIR, 'app', 'ml_models', 'model_best.keras')
            if os.path.exists(model_path):
                _model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    return _model

def predict_next_action(sequence):
    model = get_model()
    if model is None:
        return None
    
    try:
        # Assuming sequence is a list of action indices
        X = np.array(sequence).reshape(1, len(sequence), 1).astype('float32')
        prediction = model.predict(X, verbose=0)
        action_idx = np.argmax(prediction, axis=1)[0]
        return int(action_idx)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None
