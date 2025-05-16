import pandas as pd
import numpy as np
from itertools import combinations

for task in ['sepsis']:
    df = pd.read_csv(f'{task}.csv')
    df = df.fillna(0)
    df = df.dropna(thresh=4)
    # df = df.drop(columns=['4.0'])
    df.columns = [df.columns[0]] + [float(c) for c in df.columns[1:]]
    
    times = df.columns[1:]
    print(df.columns)
    times = np.array([float(t) for t in times])
    probs = df.iloc[:, 1:].values

    idx_pairs = [(i, j) for i, j in combinations(range(len(times)), 2) if abs(times[i] - times[j]) <= 60.0]
    i_idx = np.array([i for i, _ in idx_pairs])
    j_idx = np.array([j for _, j in idx_pairs])

    delta_times = np.abs(times[i_idx] - times[j_idx])
    delta_probs = np.abs(probs[:, i_idx] - probs[:, j_idx]) 

    lipschitz_constants = delta_probs / delta_times 
    cleaned_rows = [row[~np.isnan(row)] for row in delta_probs]
    row_means = np.array([
        np.mean(row) if len(row) > 0 else np.nan
        for row in cleaned_rows
    ])
    row_means
    df['lipschitz'] = row_means
    print(task)
    print(df['lipschitz'].dropna().mean())
    # df.to_csv(f'/home/seewon/ehrshot/results/{MODEL}_{TASK}_{METHOD}_{I}_smooth2.csv', index=False)