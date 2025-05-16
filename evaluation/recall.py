
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import pickle 
import json

def recall_at_precision_thresholds(y_true, y_scores, precision_thresholds=[0.6, 0.7, 0.8]):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    recall_at_precision = {}

    for pt in precision_thresholds:
        # Find all recall values where precision >= pt
        valid_recalls = recalls[precisions >= pt]
        if len(valid_recalls) > 0:
            recall_at_precision[pt] = max(valid_recalls)
        else:
            recall_at_precision[pt] = 0.0  # or np.nan if you want to indicate not achievable

    return recall_at_precision

for task in ['icu']: #  'hypoglycemia', 
    if task == 'decomp' or task == 'sepsis':
        dataset = 'mc_med'
        TIME = 90
    elif task == 'hyperkalemia' or task == 'hypoglycemia':
        dataset = 'ehrshot'
        TIME = 60
    elif task == 'mortality':
        dataset = 'mimic'
        TIME = 12
    elif task == 'icu':
        dataset = 'mimic'
        TIME = 6.0

    with open(f'stability_results/{dataset}_{task}/qwen32b.json', 'r') as file:
        data = json.load(file)

    predictions = data['all_predictions']

    y_score, y_true = [], []
    for prediction in predictions:
        if round(prediction['TimeUpto'], ndigits=2) == TIME and not (prediction['PredictedProb'] is None):
            y_score.append(prediction['PredictedProb'])
            y_true.append(prediction['TrueLabel'])
        

    x = recall_at_precision_thresholds(y_true, y_score, [0.6, 0.7, 0.8])
    print(x)

        
    