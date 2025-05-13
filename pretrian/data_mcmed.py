import torch

import os
import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split

DATA_DIR = '/home/mkeoliya/projects/mc-med/mc-med-1.0.0/data'

COLS_NAME = ['event', 'time', 'value']
COLS = {
    "labs": ['Component_name', 'Order_time', 'Component_value'],
    "numerics": ['Measure', 'Time', 'Value'],
    "orders": ['Procedure_ID', 'Order_time'],
}

def read_csv_fn(fn):
    cols = COLS[fn]
    df = pd.read_csv(os.path.join(DATA_DIR, f'{fn}.csv'), usecols=['CSN'] + cols)
    df = df.rename(columns={c: COLS_NAME[i] for i, c in enumerate(cols)})
    df = df.dropna()
    df['event'] = df['event'].astype(str)
    return df

def try_parse_float(x):
    if isinstance(x, str):
        try:
            x = x.lstrip('<>=').strip()
            x = x.replace(',', ':')
            if ':' in x:
                try:
                    num, denom = x.split(':')
                    return float(num) / float(denom)
                except (ValueError, ZeroDivisionError):
                    return x
            return float(x)
        except ValueError:
            return x
    return x

def bucket_eventval(event, val, d):
    buckets = d[event]
    ind = np.searchsorted(buckets, val, side='right')
    if ind == len(buckets):
        eventval = f"{event}|{buckets[ind-1]}-"
    else:
        eventval = f"{event}|{buckets[ind-1]}-{buckets[ind]}"
    return eventval


def bucket_ind(event, val, d):
    buckets = d[event]
    ind = np.searchsorted(buckets, val, side='right')
    return ind


def create_df():
    visits_df = pd.read_csv(os.path.join(DATA_DIR, 'visits.csv'), usecols=['CSN', 'Arrival_time']) # 'Age', 'Gender', 'Race', 'Ethnicity'

    vitals_df = read_csv_fn('numerics')
    with open('next_token/numerics_buckets.pkl', 'rb') as f:
        buckets = pickle.load(f)
    vitals_df['eventval'] = vitals_df.apply(lambda x: bucket_eventval(x['event'], x['value'], buckets), axis=1)
    vitals_df['buckets'] = vitals_df.apply(lambda x: bucket_ind(x['event'], x['value'], buckets), axis=1)

    labs_df = read_csv_fn('labs')
    labs_df['value'] = labs_df['value'].replace([None], 0.0).apply(try_parse_float)
    labs_df['value'] = labs_df['value'].apply(lambda x: 0.0 if isinstance(x, str) and 'pos' in x.lower() else x)
    labs_df['value'] = labs_df['value'].apply(lambda x: 1.0 if isinstance(x, str) and (any(sub in x.lower() for sub in ['neg', 'not', 'none', 'auto'])) else x)

    with open('next_token/labs_buckets.pkl', 'rb') as f:
        buckets = pickle.load(f)
    labs_df['eventval'] = labs_df.apply(lambda x: bucket_eventval(x['event'], x['value'], buckets), axis=1)
    labs_df['buckets'] = labs_df.apply(lambda x: bucket_ind(x['event'], x['value'], buckets), axis=1)

    orders_df = read_csv_fn('orders')
    orders_df['value'] = 0
    orders_df['buckets'] = 0
    orders_df['eventval'] = orders_df['event']

    df = pd.concat([labs_df, vitals_df, orders_df])
    df = df.merge(visits_df, on='CSN', how='left')

    df['time'] = df['time'].apply(lambda x: str(int(x[:4]) - 500) + x[4:])
    df['Arrival_time'] = df['Arrival_time'].apply(lambda x: str(int(x[:4]) - 500) + x[4:])
    df['time'] = pd.to_datetime(df['time']) - pd.to_datetime(df['Arrival_time'])
    df['time'] = df['time'].dt.total_seconds() / 3600
    df = df.drop(columns=['Arrival_time'])

    df = df.sort_values(['CSN', 'time']).groupby('CSN').agg(list).reset_index(drop=True)
    return df

def sepsis_loader(batch_size, SickDataset, context_length = 1024, seed=1234, filter=False):
    if os.path.exists("data/eventval.parquet"):
        df = pd.read_parquet("data/eventval.parquet", columns=['time', 'eventval'])
        df = df.sample(10)
        print("READING")
    else:
        df = create_df()
        print("DONE")
    
    if filter:
        print("FILTER")
        df2 = pd.read_parquet("data/decomp_data.parquet")

        events = list(df2['event'].unique())
        print(events)

        df['times2'] = df.apply(lambda x: [t for (t, e) in zip(x['time'], x['eventval']) if e.startswith(tuple(events))], axis=1)
        print("TIME")
    
        df['events2'] = df.apply(lambda x: [e for (t, e) in zip(x['time'], x['eventval']) if e.startswith(tuple(events))], axis=1)
        print("EVENT")

        df = df.drop(columns=['time', 'eventval'])
        df = df.rename(columns={'times2': 'time', 'events2':'eventval'})

    df = df[df['time'].apply(len)>0]
    df = df[df['eventval'].apply(len) <= context_length]
    # df = df[df['eventval'].apply(len) >= 10]
    
    train_inputs, temp_inputs = train_test_split(
        df, test_size=0.4, random_state=seed
    )

    val_inputs, test_inputs = train_test_split(
        temp_inputs, test_size=0.5, random_state=seed
    )

    train = SickDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader