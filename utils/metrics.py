import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy import stats


def calculate_pearson_correlation(x, y):
    """Compute Pearson correlation without np.corrcoef; returns value in [-1, 1]."""
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        covariance += dx * dy
        variance_x += dx * dx
        variance_y += dy * dy
    epsilon = 1e-10
    variance_x = max(variance_x, epsilon)
    variance_y = max(variance_y, epsilon)
    correlation = covariance / (variance_x ** 0.5 * variance_y ** 0.5)
    return max(min(correlation, 1.0), -1.0)


def get_metrics(output, target, metrics=['accuracy'], is_binary=False):
    """Compute requested metrics (Acc, BAC, MSE, R2, etc.) from model output and target; returns dict of metric names to values."""
    results = {}
    if len(target.shape) > 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if is_binary:
        if output.shape[-1] == 1:
            y_pred_proba = output.squeeze(-1)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred_proba = output[:, 1]
            y_pred = np.argmax(output, axis=1)
        y_true = target.astype(int)
    else:
        if len(output.shape) > 1 and output.shape[-1] > 1:
            y_pred = np.argmax(output, axis=1)
            y_true = target.astype(int)
        else:
            y_pred = output.flatten()
            y_true = target.flatten()
    for metric in metrics:
        metric_upper = metric.upper()
        
        if metric_upper in ['ACCURACY', 'ACC']:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        elif metric_upper in ['BALANCED_ACCURACY', 'BAC']:
            results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        elif metric_upper in ['COHEN_KAPPA', 'COHEN\'S KAPPA']:
            results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        elif metric_upper in ['F1', 'WEIGHTED-F1']:
            results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        elif metric_upper in ['PRECISION']:
            results['precision'] = precision_score(y_true, y_pred, average='weighted')
        
        elif metric_upper in ['RECALL']:
            results['recall'] = recall_score(y_true, y_pred, average='weighted')
        
        elif metric_upper in ['AUROC'] and is_binary:
            try:
                results['auroc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results['auroc'] = 0.5
        elif metric_upper in ['AUPRC'] and is_binary:
            try:
                results['auprc'] = average_precision_score(y_true, y_pred_proba)
            except ValueError:
                results['auprc'] = 0.5
        elif metric_upper in ['CORR']:
            results['CORR'] = float(calculate_pearson_correlation(y_pred, y_true))
            
            

        
        elif metric_upper in ['R2']:
            try:
                results['R2'] = r2_score(y_true, y_pred)
            except Exception as e:
                print(f"Error computing R2: {e}")
                results['R2'] = -1.0
        elif metric_upper in ['RMSE']:
            results['RMSE'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric_upper in ['MSE']:
            results['MSE'] = float(mean_squared_error(y_true, y_pred))
        elif metric_upper in ['MAE']:
            results['MAE'] = float(mean_absolute_error(y_true, y_pred))
    return results
