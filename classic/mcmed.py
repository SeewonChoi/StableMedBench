import numpy as np
import pandas as pd

from dataset import extract_stats

TIME = 1.5

def combine_classic(df):
    events = df.groupby('event')['CSN'].nunique().sort_values(ascending=False)
    events = events.reset_index()
    events = events['event'].unique()[:30]

    df = df[df['event'].isin(events)]

    df_pivoted = None
    for event in events:
        df_i = df[df['event'] == event].copy()
        unique_vals = df_i['value'].unique()
        print(event)

        if len(unique_vals) <= 1:
            df2 = df_i.groupby('CSN')['event'].count().rename(f'{event}_count').reset_index()
        else:
            df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
            df_i = df_i.dropna()
            df2 = df_i.groupby('CSN')['value'].apply(list).apply(extract_stats).reset_index()
            df2.columns = ['CSN'] + [f'{event}_{col}' for col in df2.columns[1:]]
        
        if df_pivoted is None:
            df_pivoted = df2
        else:
            df_pivoted = df_pivoted.merge(df2, on='CSN', how='outer')

    df_pivoted = df_pivoted.loc[:, df_pivoted.isnull().mean() < 0.9]
    
    return  df_pivoted

def decomp_loader():
    label_df = pd.read_parquet("data/decompensation.parquet")
    demo_df = pd.read_parquet('data/decompensation_demo.parquet', columns=['CSN', 'Age', 'Gender_ind', 'Race_ind', 'Ethnicity_ind'])
    df = pd.read_parquet("data/decompensation_data.parquet")

    df = df.merge(label_df, on='CSN')
    df = df[df['time_arrive'] <= df['Time'] - TIME]
    df = df[['CSN', 'event', 'value']]

    df = combine_classic(df)
    df = df.merge(label_df[['CSN', 'Label']], on='CSN')
    df = df.merge(demo_df, on='CSN')

    X = df.drop(columns=['CSN', 'Label'])
    y = df[['Label']]
    ids = df[['CSN']]
    
    print(f"Original class distribution: {np.bincount(y['Label'].astype(int))}")

    return X, y, ids


def sepsis_loader():
    label_df = pd.read_parquet("data/eSOFA.parquet")
    demo_df = pd.read_parquet('data/eSOFA_demo.parquet', columns=['CSN', 'Age', 'Gender_ind', 'Race_ind', 'Ethnicity_ind'])
    df = pd.read_parquet("data/eSOFA_data.parquet")

    df = df.merge(label_df, on='CSN')
    df = df[df['time'] <= df['Time'] - TIME]
    df = df[['CSN', 'event', 'value']]

    df = combine_classic(df)
    df = df.merge(label_df[['CSN', 'Label']], on='CSN')
    df = df.merge(demo_df, on='CSN')

    X = df.drop(columns=['CSN', 'Label'])
    y = df[['Label']]
    ids = df[['CSN']]

    print(f"Original class distribution: {np.bincount(y['Label'].astype(int))}")
    
    return X, y, ids