import torch
import os
import pandas as pd

from sklearn.model_selection import train_test_split

#from hf_ehr.data.tokenization import CLMBRTokenizer
#from hf_ehr.config import Event

DATA_DIR = 'data'

class SickDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            ecg_data: numpy array of shape (total_samples, n_leads)
            labels: numpy array with timestamps of abnormal events
            window_size: number of samples in each snippet
        """
        self.samples = data[['eventval', 'time']]
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        return list(sample['eventval']), sample['time']

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        return events, times

def split_chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def read_df(context_length):
    df_final = None
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, f'demographics.parquet'), columns=['patient_id', 'start', 'code_only'])
    demo_df = demo_df.dropna()
    demo_df = demo_df.rename(columns={'code_only' : 'eventval'})
    demo_df['time'] = 0
    for i in range(0, 34):
        df = pd.read_parquet(os.path.join(DATA_DIR, f'output_{i}.parquet'), columns=['patient_id', 'start', 'eventval'])
        df = df.sort_values(['patient_id', 'start'])
        df['start'] = pd.to_datetime(df['start'])
        df['time'] = (df['start'] - df.groupby('patient_id')['start'].transform('min')).dt.total_seconds() / 3600
        df = df.dropna()

        df = pd.concat([demo_df, df])
        df = df.groupby('patient_id').agg(list)

        rows = []

        for patient_id, row in df.iterrows():
            event_chunks = split_chunks(row['eventval'], context_length)
            time_chunks = split_chunks(row['time'], context_length)

            for event_chunk, time_chunk in zip(event_chunks, time_chunks):
                rows.append({'patient_id': patient_id, 'eventval': event_chunk, 'time': time_chunk})

        df = pd.DataFrame(rows)

        if df_final is None:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])
    return df_final

def sick_loader(batch_size, context_length=1000, seed=1234):
    df = read_df(context_length)
    
    train_inputs, temp_inputs = train_test_split(
        df, test_size=0.4, random_state=seed
    )

    val_inputs, test_inputs = train_test_split(
        temp_inputs, test_size=0.5, random_state=seed
    )
    
    train = SickDataset(train_inputs)
    val = SickDataset(val_inputs)
    test = SickDataset(test_inputs)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=SickDataset.collate_fn, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, collate_fn=SickDataset.collate_fn, shuffle=True)
    test_loader =torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=SickDataset.collate_fn, shuffle=True)

    return train_loader, val_loader, test_loader