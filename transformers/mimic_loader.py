import torch
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class MIMICDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'time', 'Label', 'demo_str']]
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
        event = list(sample['eventval'])
        time = list(sample['time'])
        demo = list(sample['demo_str'])
        return event, demo, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        demos = [item[1] for item in batch]
        times = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        labels = torch.tensor(labels)
        return events, demos, times, labels

def mortality_loader(TIME, batch_size, context_length=1000, seed=1234, stay=True):
    label_df = pd.read_csv("data/mortality/mortality_labels.csv", usecols=['subject_id', 'stay_id', 'Label', 'intime', 'Time'])
    demo_df = pd.read_parquet('data/mortality/demographics.parquet', columns=['race_str', 'gender_str', 'ethnicity_str', 'age_str'])
    demo_df['subject_id'] = pd.to_numeric(demo_df.index)
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['ethnicity_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str', 'ethnicity_str'])
    
    df = None
    for i in range(0, 37):
        df_i = pd.read_parquet(f'data/mortality/{i}_final.parquet', columns=['subject_id', 'stay_id', 'time', 'eventval'])
        # df_i = df_i.dropna(subset=['time', 'eventval'])
        df_i = df_i.merge(label_df[['subject_id', 'stay_id', 'Time', 'intime']], on='subject_id')
        if stay:
            df_i = df_i[df_i['stay_id_x'] == df_i['stay_id_y']]
            df_i = df_i.drop(columns=['stay_id_x', 'stay_id_y'])
        else:
            df_i = df_i[(df_i['stay_id_x'] == df_i['stay_id_y']) | (df_i['stay_id_x'].isna())]
            df_i = df_i.drop(columns=['stay_id_x', 'stay_id_y'])
        df_i['time'] = pd.to_datetime(df_i['time'])
        df_i['intime'] = pd.to_datetime(df_i['intime'])
        df_i['Time'] = pd.to_datetime(df_i['Time'])

        df_i['time'] = (df_i['time'] - df_i['intime']).dt.total_seconds() / 3600
        df_i['Time'] = (df_i['Time'] - df_i['intime']).dt.total_seconds() / 3600

        df_i[df_i['Time'] >= TIME]
        df_i = df_i[df_i['time'] <= (df_i['Time'] - TIME)]
        df_i = df_i.drop(columns=['Time', 'intime'])
        df_i = df_i.sort_values(['subject_id', 'time']).groupby('subject_id').agg(list).reset_index()

        df_i['eventval'] = df_i['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
        df_i['time'] = df_i['time'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

        df_i = df_i.merge(label_df[['subject_id', 'Label']], on='subject_id', how='left')
        df_i = df_i.merge(demo_df, on='subject_id', how='left')

        if df is None: df = df_i
        else: df = pd.concat([df, df_i])
    df = df.drop(columns='subject_id')

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = MIMICDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = MIMICDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = MIMICDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def icu_loader(TIME, batch_size, context_length=1000, seed=1234):
    med_df = pd.read_csv(f"data/transfer/med.csv", usecols=['stay_id', 'charttime', 'eventval'])
    num_df = pd.read_csv(f"data/transfer/numerics.csv", usecols=['stay_id', 'charttime', 'eventval'])
    df = pd.concat([med_df, num_df]).sort_values(['stay_id', 'charttime'])
    
    label_df = pd.read_csv("data/transfer/label.csv", usecols=['stay_id', 'Label', 'intime', 'Time'])
    demo_df = pd.read_csv('data/transfer/demo.csv', usecols=['stay_id', 'gender_str', 'race_str', 'age_str'])
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str'])

    df = df.merge(label_df[['stay_id', 'Time', 'intime']], on='stay_id')
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])

    df['charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600

    df[df['Time'] >= TIME]
    df = df[df['charttime'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['Time', 'intime'])
    df = df.sort_values(['stay_id', 'charttime']).groupby('stay_id').agg(list).reset_index()

    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['charttime'] = df['charttime'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df = df.rename(columns={'charttime': 'time'})

    df = df.merge(label_df[['stay_id', 'Label']], on='stay_id', how='left')
    df = df.merge(demo_df, on='stay_id')

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = MIMICDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = MIMICDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = MIMICDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=MIMICDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader