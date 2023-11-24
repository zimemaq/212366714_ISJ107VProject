import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def get_metrics_df(results_df, threshold=0.50):

    yhat_default = np.where(results_df["Score"] >= 0.5, 1, 0)
    auc_score_default = roc_auc_score(results_df["Target variable"], yhat_default)
    ap_score_default = average_precision_score(results_df["Target variable"], yhat_default)
    results_df['Classification_default'] = yhat_default
    results_df['Default threshold'] = 0.5

    yhat = np.where(results_df["Score"] >= threshold, 1, 0)
    auc_score = roc_auc_score(results_df["Target variable"], yhat)
    ap_score = average_precision_score(results_df["Target variable"], yhat)
    results_df['Classification'] = yhat
    results_df['Updated threshold'] = threshold

    metrics_df = pd.DataFrame(
        {
            "Metric name": ["Area under the curve", "Average Precision", "Threshold"],
            "Score": [
                auc_score,
                ap_score,
                threshold
            ],
            "Cut-off score": [0.8, 0.01, ''],
        }
    )

    metrics_df_default = pd.DataFrame(
                {
                    "Metric name": ["Area under the curve", "Average Precision", "Threshold"],
                    "Score": [
                        auc_score_default,
                        ap_score_default,
                        0.50
                    ],
                    "Cut-off score": [0.8, 0.01, ''],
                }
            )

    return results_df, metrics_df, metrics_df_default