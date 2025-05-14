import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split

COLS_NAME = ['event', 'time', 'value']
COLS = {
    "labs": ['Component_name', 'Order_time', 'Component_value'],
    "numerics": ['Measure', 'Time', 'Value'],
    "orders": ['Procedure_ID', 'Order_time'],
}

class SickDataset(Dataset):
    def __init__(self, data):
        self.samples = data[['eventval', 'time']]
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        label = torch.tensor(1)
        event = list(sample['eventval'])
        time = list(sample['time'])
        return event, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        labels = torch.stack(labels, dim=0)
        return events, times, labels

def read_csv_fn(fn):
    cols = COLS[fn]
    df = pd.read_csv(f'{fn}.csv', usecols=['CSN'] + cols)
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
    visits_df = pd.read_csv('visits.csv', usecols=['CSN', 'Arrival_time']) # 'Age', 'Gender', 'Race', 'Ethnicity'

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

def sepsis_loader(batch_size, context_length = 1024, seed=1234):
    if os.path.exists("data/eventval.parquet"):
        df = pd.read_parquet("data/eventval.parquet", columns=['time', 'eventval'])
        df = df.sample(10)
        print("READING")
    else:
        df = create_df()
        print("DONE")

    df = df[df['eventval'].apply(len) <= context_length]
    
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