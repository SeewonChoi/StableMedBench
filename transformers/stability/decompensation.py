from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F

import os
import pandas as pd
import numpy as np

from models import CustomGPT, CustomMamba
# from models_nesy import CustomGPTLinear
from collections import OrderedDict

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/eventval_tokenizer.json")
tokenizer.eos_token = '[EOS]'
tokenizer.sep_token = '[SEP]'
tokenizer.bos_token = '[BOS]'
tokenizer.pad_token = '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.mask_token = '[MASK]'

DATA_DIR = '/project/ClinicalAI/working_data/seewon/mc-med/'

def decomp_loader(i):
    df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation_data.parquet"))
    label_df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation.parquet"))
    
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, "data/decompensation_demo.parquet"), 
                              columns=['CSN', 'Age_str', 'Race_str', 'Ethnicity_str', 'Gender_str'])

    test_ids = pd.read_csv('decomp.csv')
    test_ids = test_ids[i*10 : (i+1):10]
    df = df[df['CSN'].isin(test_ids['CSN'])]
    label_df = label_df[label_df['CSN'].isin(test_ids['CSN'])]
    demo_df = demo_df[demo_df['CSN'].isin(test_ids['CSN'])]
    
    label_df['Time'] = label_df.apply(lambda x: x['Trigger_time'] if x['Label'] else x['Sample_time'], axis=1)
    df = df.merge(label_df[['CSN', 'Time']], on='CSN')

    df = df[df['Time'] > 1.5]
    df['delta'] = df['Time'] - df['time_arrive']
    df = df.drop(columns=['Time'])
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
    elif model_name == 'gpt_linear':
        model = CustomGPTLinear(vocab_size=len(tokenizer)).to(device)
    
    context_length = 1024

    lmhead_state_dict = torch.load(f"checkpoints/decomp_{model_name}_1234.pth", weights_only=True, map_location=device)
    new_state_dict = OrderedDict()
    for key, value in lmhead_state_dict.items():
        if ('wte' not in key):
            new_key = key.replace('model.transformer', 'model')
            new_state_dict[new_key] = value
    model.load_state_dict(state_dict=new_state_dict, strict=False)

    for i in range(0, 2256, 10):
        # Load data
        df_all, label_df, demo_df, probs_df = decomp_loader(i)

        df_i = None
        TIMES = np.arange(0.5, 4.0, 0.1)

        for n, TIME in enumerate(TIMES):
            df = df_all[df_all['delta'] >= TIME]
            if n > 0:
                valid_ids = df_all[(df_all['delta'] > TIMES[n-1]) & df_all['delta'] <= TIME]['CSN'].unique()
                df = df[df['CSN'].isin(valid_ids)]

            df = df.drop(columns='delta')

            df = df.sort_values(['CSN', 'time_arrive']).groupby('CSN').agg(list).reset_index()

            df = df[df['eventval'].apply(len) >= 5]
            df['eventval'] = df['eventval'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)
            df['time_arrive'] = df['time_arrive'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)

            df = df.merge(label_df[['CSN', 'Label']], on='CSN', how='left')
            df = df.merge(demo_df, on='CSN', how='left')
            
            valid_ids = df['CSN'].unique()
            if len(valid_ids) == 0: continue

            df['demo_str'] = df.apply(lambda sample: [sample['Age_str'], sample['Gender_str'], sample['Race_str'], sample['Ethnicity_str']], axis=1)
            input_ids = list((df['demo_str'] + df['eventval']).values)
            times = list(df['time_arrive'].values) 

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
            
        if not df_i is None:
            df_i.to_csv(f'decomp_stability/{model_name}_prob_{i}.csv', index=False)
