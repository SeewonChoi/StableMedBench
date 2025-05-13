import torch

import os
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

DATA_DIR = '/project/ClinicalAI/working_data/seewon/hf_ehr/'
TIME = 1.0

def create_df_limit(df, label_df, demo_df, context_length):
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

def hypoglycemia_loader(batch_size, SickDataset, context_length=1000, seed=1234):
    TASK = 'hypoglycemia'
    df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}2.parquet"), columns=['patient_id', 'start', 'eventval'])
    label_df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}_label2.parquet"), columns=['patient_id', 'value', 'start', 'Time'])
    label_df = label_df.rename(columns={'value': 'Label', 'start': 'start_time'})
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}_demo.parquet"),
                              columns=['patient_id', 'gender_str', 'ethnicity_str', 'race_str', 'age_str'])
    
    df = create_df_limit(df, label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def hyperkalemia_loader(batch_size, SickDataset, context_length=1000, seed=1234):
    df = pd.read_parquet(os.path.join(DATA_DIR, "data/lab_hyperkalemia/hyperkalemia2.parquet"), columns=['patient_id', 'start', 'eventval'])
    label_df = pd.read_parquet(os.path.join(DATA_DIR, "data/lab_hyperkalemia/hyperkalemia_label2.parquet"), columns=['patient_id', 'value', 'start', 'Time'])
    label_df = label_df.rename(columns={'value': 'Label', 'start': 'start_time'})
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, "data/lab_hyperkalemia/hyperkalemia_demo.parquet"),
                              columns=['patient_id', 'gender_str', 'ethnicity_str', 'race_str', 'age_str'])
    
    df = create_df_limit(df, label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader