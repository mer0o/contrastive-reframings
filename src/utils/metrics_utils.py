import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix



def get_metrics_from_predictions(y_true, y_pred):
    """
    Calculate various metrics from the true and predicted labels.

    This function calculates the following metrics:
    - F1 Score
    - Accuracy
    - Precision
    - Recall
    - Confusion Matrix components

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    
    metrics = {
        'acc': accuracy,
        'prec': precision,
        'recall': recall,
        'f1': f1,
        'confusion': confusion,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
    }

    return metrics