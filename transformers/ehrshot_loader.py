import torch

import os
import pandas as pd
import random
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

HYPERKALEMIA = ['LOINC/LG7931-1', 'LOINC/LP386618-5', 'LOINC/LG10990-6', 'LOINC/6298-4', 'LOINC/2823-3']
HYPOGLYCEMIA = ['SNOMED/33747003', 'LOINC/LP416145-3', 'LOINC/14749-6']

class EHRSHOTDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'start', 'Label', 'age_str', 'gender_str', 'race_str', 'ethnicity_str']]
        for _, d in data.iterrows():
            if d['Label'] == 1: 
                self.samples.append(d)
                labels.append(d['Label'])
            else: 
                if random.random() < 1.0: 
                    self.samples.append(d)
                    labels.append(d['Label'])
        
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)

        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        print(f"Original: {counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[self.index_map[idx]] 
        label = float(sample['Label'])
        event = [sample['age_str'], sample['gender_str'], sample['race_str'], sample['ethnicity_str']] + list(sample['eventval'])
        time = list(sample['start'])
        return event, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        labels = torch.tensor(labels)
        return events, times, labels


def create_df_limit(TIME, df, label_df, demo_df, context_length):
    df = df.merge(label_df[['patient_id', 'Time', 'start_time']], on='patient_id')

    df['start'] = (df['start'] - df['start_time']).dt.total_seconds() / 3600
    df['Time'] = (df['Time'] - df['start_time']).dt.total_seconds() / 3600

    df = df[df['Time'] >= TIME]
    df = df[df['start'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['Time', 'start_time'])
    df = df.sort_values(['patient_id', 'start']).groupby('patient_id').agg(list).reset_index()

    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['start'] = df['start'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

    df = df.merge(label_df[['patient_id', 'Label']], on='patient_id', how='left')
    df = df.merge(demo_df, on='patient_id')

    return df

def hypoglycemia_loader(TIME, batch_size, context_length=1000, seed=1234):
    TASK = 'hypoglycemia'
    df = pd.read_parquet(f"data/lab_{TASK}/{TASK}2.parquet", columns=['patient_id', 'code', 'start', 'eventval'])
    df = df[~df['code'].isin(HYPOGLYCEMIA)]
    df = df.drop(columns=['code'])
    
    label_df = pd.read_parquet(f"data/lab_{TASK}/{TASK}_label2.parquet", columns=['patient_id', 'value', 'start', 'Time'])
    label_df = label_df.rename(columns={'value': 'Label', 'start': 'start_time'})
    demo_df = pd.read_parquet(f"data/lab_{TASK}/{TASK}_demo.parquet",
                              columns=['patient_id', 'gender_str', 'ethnicity_str', 'race_str', 'age_str'])
    
    df = create_df_limit(TIME, df, label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = EHRSHOTDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = EHRSHOTDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = EHRSHOTDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def hyperkalemia_loader(TIME, batch_size, context_length=1000, seed=1234):
    df = pd.read_parquet("data/lab_hyperkalemia/hyperkalemia2.parquet", columns=['patient_id', 'code', 'start', 'eventval'])
    df = df[~df['code'].isin(HYPOGLYCEMIA)]
    df = df.drop(columns=['code'])

    label_df = pd.read_parquet("data/lab_hyperkalemia/hyperkalemia_label2.parquet", columns=['patient_id', 'value', 'start', 'Time'])
    label_df = label_df.rename(columns={'value': 'Label', 'start': 'start_time'})
    demo_df = pd.read_parquet("data/lab_hyperkalemia/hyperkalemia_demo.parquet",
                              columns=['patient_id', 'gender_str', 'ethnicity_str', 'race_str', 'age_str'])
    
    df = create_df_limit(TIME, df, label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = EHRSHOTDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = EHRSHOTDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = EHRSHOTDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=EHRSHOTDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader