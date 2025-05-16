from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from models import CustomGPT, CustomMamba

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/ed_eventval_tokenizer.json")
tokenizer.eos_token = '[EOS]'
tokenizer.sep_token = '[SEP]'
tokenizer.bos_token = '[BOS]'
tokenizer.pad_token = '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.mask_token = '[MASK]'

class TimeSeriesDataset(Dataset):
    def __init__(self, input_ids_all, times_all):
        self.input_ids_all = input_ids_all
        self.times_all = times_all

    def __len__(self):
        return len(self.input_ids_all)

    def __getitem__(self, idx):
        return self.input_ids_all[idx], self.times_all[idx]

def collate_fn(batch):
    input_ids, times = zip(*batch)
    input_ids, times, attention_masks = get_inputs(list(input_ids), list(times))
    return input_ids, times, attention_masks

def icu_loader(i):
    med_df = pd.read_csv(f"data/transfer/med.csv", usecols=['stay_id', 'charttime', 'eventval'])
    num_df = pd.read_csv(f"data/transfer/numerics.csv", usecols=['stay_id', 'charttime', 'eventval'])
    df = pd.concat([med_df, num_df]).sort_values(['stay_id', 'charttime'])
    
    label_df = pd.read_csv("data/transfer/label.csv", usecols=['stay_id', 'Label', 'intime', 'Time'])

    demo_df = pd.read_csv('data/transfer/demo.csv', usecols=['stay_id', 'gender_str', 'race_str', 'age_str'])
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str'])

    test_ids = pd.read_csv('transfer.csv')
    test_ids = test_ids[i:i+10]

    df = df[df['stay_id'].isin(test_ids['stay_id'])]
    demo_df = demo_df[demo_df['stay_id'].isin(test_ids['stay_id'])]

    df = df.merge(label_df[['stay_id', 'Time', 'intime']], on='stay_id')
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])

    df['charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600

    df = df[df['Time'] >= 6.0]
    df['time'] = df['Time'] - df['charttime']

    df = df.drop(columns=['Time', 'intime'])
    return df, label_df, demo_df, test_ids

def get_inputs(inputs, times):
    max_len = max(len(i) for i in inputs) - 3
    tokens = tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
    sequences = tokens["input_ids"].to(device)   
    attention_masks = tokens["attention_mask"].to(device) 

    times = [F.pad(torch.tensor(r, dtype=torch.float32), (3, max_len - len(r))).round(decimals=2)  for r in times]
    times = torch.stack(times, dim=0).to(device)  

    return sequences, times, attention_masks


if __name__ == "__main__":
    model_name = 'gpt'

    device =torch.device("cuda:%d" % 0) if torch.cuda.is_available() else torch.device('cpu')

    if model_name == "gpt":
        model = CustomGPT(vocab_size=len(tokenizer)).to(device)
    elif model_name == "mamba":
        model = CustomMamba(vocab_size=len(tokenizer)).to(device)
    context_length = 1024

    lmhead_state_dict = torch.load("checkpoints/icu_gpt_1234.pth", weights_only=True, map_location=device)
    model.load_state_dict(state_dict=lmhead_state_dict, strict=True)

    for i in range(60, 11810, 10):
        # Load data
        df_all, label_df, demo_df, probs_df = icu_loader(i)
        df_i = None
        TIMES = np.arange(3.0, 9.0, 0.1)
        for n, TIME in enumerate(TIMES):
            df = df_all[df_all['time'] >= TIME]
            if n > 0:
                valid_ids = df_all[(df_all['time'] > TIMES[n-1]) & df_all['time'] <= TIME]['stay_id'].unique()
                df = df[df['stay_id'].isin(valid_ids)]
            
            df = df.sort_values(['stay_id', 'charttime']).groupby('stay_id').agg(list).reset_index()

            # df = df[df['eventval'].apply(len) >= 5]
            df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
            df['charttime'] = df['charttime'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

            df = df.merge(label_df[['stay_id', 'Label']], on='stay_id', how='left')
            df = df.merge(demo_df, on='stay_id', how='left')

            valid_ids = df['stay_id'].unique()
            input_ids = list((df['demo_str'] + df['eventval']).values)
            times = list(df['charttime'].values) 

            input_ids, times, attention_masks = get_inputs(input_ids, times)

            output = model(input_ids, times, attention_masks)
            output = torch.sigmoid(output).squeeze(-1)

            # probs_df[TIME] = (torch.stack(probs_all, dim=-1).flatten()).detach().cpu().numpy()
            if df_i is None:
                df_i = pd.DataFrame({
                'ids': valid_ids,
                f'{TIME}': output.cpu().detach().numpy()
                })
            else:
                print(valid_ids.shape)
                print(output.shape)
                df_temp = pd.DataFrame({
                'ids': valid_ids,
                    f'{TIME}': output.cpu().detach().numpy()
                })
                df_i = df_i.merge(df_temp, on='ids', how='outer')
            
        df_i.to_csv(f'transfer_stability/transfer_prob_{i}.csv', index=False)
