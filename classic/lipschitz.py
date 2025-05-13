from itertools import combinations
import pandas as pd
import numpy as np
import joblib
from scipy.stats import linregress, kurtosis, skew
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score

TASK = 'mortality'
I = 0
MODEL = 'xgb'
METHOD = 'median'
BASE = '12.0'

df = pd.read_csv(f'results/{MODEL}_{TASK}_{METHOD}_{I}_smooth.csv', index_col=0)
df = df.dropna(thresh=5)

df[BASE] = df['probs']
df = df.drop(columns='probs')

MIN = df.columns[1]
df[MIN] = df.apply(lambda x: x[BASE] if np.isnan(x[MIN]) else x[MIN], axis=1)

for col_idx in range(2, len(df.columns)):
    current_col = df.columns[col_idx]
    prev_col = df.columns[col_idx - 1]
    df[current_col] = df[current_col].combine_first(df[prev_col])

times = df.columns[1:]
times = np.array([float(t) for t in times])
probs = df.iloc[:, 1:].values

idx_pairs = [(i, j) for i, j in combinations(range(len(times)), 2) if abs(times[i] - times[j]) <= 0.167]
i_idx = np.array([i for i, _ in idx_pairs])
j_idx = np.array([j for _, j in idx_pairs])

delta_times = np.abs(times[i_idx] - times[j_idx])
delta_probs = np.abs(probs[:, i_idx] - probs[:, j_idx]) 

lipschitz_constants = delta_probs / delta_times 
max_lipschitz_per_row = lipschitz_constants.max(axis=1)
df['lipschitz'] = max_lipschitz_per_row
# df = df.to_csv(f'results/{MODEL}_{TASK}_{METHOD}_{I}_smooth.csv')

# [df['lipschitz'] > 0.0]
x = df['lipschitz'].mean()
print(x)