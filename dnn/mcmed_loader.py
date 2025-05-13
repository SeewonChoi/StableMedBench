import torch

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_stats import plot_event_count

DATA_DIR = '/home/mkeoliya/projects/mc-med'
TIME = 1.5

def decomp_loader(batch_size, SickDataset, model, context_length=1024, plot=False, seed=1234):
    df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation_data.parquet"))
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation_demo.parquet"), 
                              columns=['CSN', 'Age_str', 'Race_str', 'Ethnicity_str', 'Gender_str'])
    
    label_df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation.parquet"))
    label_df['Time'] = label_df.apply(lambda x: x['Trigger_time'] if x['Label'] else x['Sample_time'], axis=1)
    label_df = label_df[label_df['Time'] >= TIME]
    
    df = df.merge(label_df[['CSN', 'Time']], on='CSN')

    df = df[df['time_arrive'] <= (df['Time'] - TIME)]
    df = df.groupby('CSN').agg(list).reset_index()

    df = df.merge(label_df[['CSN', 'Label']], on='CSN')
    df = df.merge(demo_df, on='CSN', how='left')

    print(df['value'].apply(len).max())
    df['value'] = df['value'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['event'] = df['event'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['time_arrive'] = df['time_arrive'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

    if plot: 
        plot_event_count(df)
    
    df_train, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    df_test, df_val = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(df_train, sample=False, backbone=model)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(df_val, backbone=model)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(df_test, backbone=model)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def sepsis_loader(time, batch_size, SickDataset, model, context_length=1000, seed=1234, plot=False):
    df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA_data.parquet"))
    label_df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA.parquet"))
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA_demo.parquet"),
                              columns=['CSN', 'Age_str', 'Race_str', 'Ethnicity_str', 'Gender_str'])

    label_df['Time'] = label_df.apply(lambda x: x['Trigger_time'] if x['Label'] else x['Sample_time'], axis=1)
    label_df = label_df[(label_df['Time'] >= TIME)]
    df = df.merge(label_df[['CSN', 'Time']], on='CSN')

    df = df[df['time'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['Time'])
    df = df.sort_values(['CSN', 'time']).groupby('CSN').agg(list).reset_index()

    df['value'] = df['value'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['event'] = df['event'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['time'] = df['time'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

    df = df.merge(label_df[['CSN', 'Label']], on='CSN', how='left')
    df = df.merge(demo_df, on='CSN')

    if plot:
        plot_event_count(df)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs, sample=True, backbone=model)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs, backbone=model)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs, backbone=model)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader