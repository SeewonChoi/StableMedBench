import pandas as pd
import pickle

def count_flips(row):
    # Drop NaNs, convert to 0/1 based on threshold
    rounded = row.dropna().apply(lambda x: 1 if x >= 0.5 else 0)
    # Compute differences and count changes
    return (rounded.diff().fillna(0) != 0).sum()


for task in ['icu']:
    df = pd.read_csv(f'{task}.csv')

    # Apply to each row
    df['flip_count'] = df.apply(count_flips, axis=1)
    print(task)
    print(df['flip_count'].mean())