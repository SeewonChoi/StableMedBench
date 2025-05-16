from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

from models import CustomGPT, CustomMamba

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/eventval_tokenizer.json")
tokenizer.eos_token = '[EOS]'
tokenizer.sep_token = '[SEP]'
tokenizer.bos_token = '[BOS]'
tokenizer.pad_token = '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.mask_token = '[MASK]'

def create_df_limit(label_df, ids, stay=True):
    df = None
    for i in range(0, 37):
        df_i = pd.read_parquet(f'data/mortality/{i}_final.parquet', columns=['subject_id', 'stay_id', 'time', 'eventval'])
        df_i = df_i[df_i['subject_id'].isin(ids)]

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

        if df is None: df = df_i
        else: df = pd.concat([df, df_i])
    return df

def mortality_loader():
    label_df = pd.read_csv("data/mortality/mortality_labels.csv", usecols=['subject_id', 'stay_id', 'Label', 'intime', 'Time'])
    
    demo_df = pd.read_parquet('data/mortality/demographics.parquet', columns=['race_str', 'gender_str', 'ethnicity_str', 'age_str'])
    demo_df['subject_id'] = pd.to_numeric(demo_df.index)
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['ethnicity_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str', 'ethnicity_str'])
    
    test_ids = pd.read_csv('mortality.csv')
    test_ids = test_ids[:10]
    label_df = label_df[label_df['subject_id'].isin(test_ids['subject_id'])]
    demo_df = demo_df[demo_df['subject_id'].isin(test_ids['subject_id'])]

    df = create_df_limit(label_df, test_ids['subject_id'])

    df = df[df['Time'] >= 12.0]

    df['delta'] = df['Time'] - df['time']
    df = df.drop(columns=['Time', 'intime'])
    return df, label_df, demo_df, test_ids


def get_inputs(inputs, times):
    max_len = max(len(i) for i in inputs) - 4
    tokens = tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
    sequences = tokens["input_ids"].to(device)   
    attention_masks = tokens["attention_mask"].to(device) 

    times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len - len(r))).round(decimals=2)  for r in times]
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

    lmhead_state_dict = torch.load("checkpoints/mortality_gpt_1234.pth", weights_only=True, map_location=device)
    model.load_state_dict(state_dict=lmhead_state_dict, strict=True)

    for i in range(20, 6231, 10):
        # Load data
        df_all, label_df, demo_df, probs_df = mortality_loader()

        df_i = None
        TIMES = np.arange(9.0, 12.0, 0.1)
        for n, TIME in enumerate(TIMES):
            df = df_all[df_all['delta'] >= TIME]
            if n > 0:
                valid_ids = df_all[(df_all['delta'] > TIMES[n-1]) & df_all['delta'] <= TIME]['subject_id'].unique()
                df = df[df['subject_id'].isin(valid_ids)]

            df = df.drop(columns='delta')

            df = df.sort_values(['subject_id', 'time']).groupby('subject_id').agg(list).reset_index()

            df = df[df['eventval'].apply(len) >= 5]
            df['eventval'] = df['eventval'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)
            df['time'] = df['time'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)

            df = df.merge(label_df[['subject_id', 'Label']], on='subject_id', how='left')
            df = df.merge(demo_df, on='subject_id', how='left')
            
            valid_ids = df['subject_id'].unique()

            input_ids = list((df['demo_str'] + df['eventval']).values)
            times = list(df['time'].values) 

            input_ids, times, attention_masks = get_inputs(input_ids, times)
            output = model(input_ids, times, attention_masks)
            output = torch.sigmoid(output).squeeze(-1)

            if df_i is None:
                df_i = pd.DataFrame({
                'ids': valid_ids,
                f'{TIME}': output.cpu().detach().numpy()
                })
            else:
                df_temp = pd.DataFrame({
                'ids': valid_ids,
                f'{TIME}': output.cpu().detach().numpy()
                })
                df_i = df_i.merge(df_temp, on='ids', how='outer')
            
        df_i.to_csv(f'mortality_stability/mortality_prob_{i}.csv', index=False)
