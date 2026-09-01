from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import numpy as np



def evaluate_model(model, X_test_scaled, y_test):
    """this functions evaluate models"""

    prediction = model.predict(X_test_scaled)

    y_pred = np.argmax(prediction, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)   

    return {
        "accuracy":accuracy,
        "Classification report":report}
