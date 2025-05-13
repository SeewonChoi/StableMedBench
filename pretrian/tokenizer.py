from tokenizers import Tokenizer, models
import os
import pandas as pd

OUTPUT_DIR = "data"
DATA_DIR = '/home/mkeoliya/projects/mc-med/mc-med-1.0.0/data'

COLS_NAME = ['event']
COLS = {
    "labs": ['Component_name'],
    "numerics": ['Measure'],
    "orders": ['Procedure_ID'],
    "rads": ['Study']
}

def read_csv_fn(fn):
    cols = COLS[fn]
    df = pd.read_csv(os.path.join(DATA_DIR, f'{fn}.csv'), usecols=['CSN'] + cols)
    df = df.rename(columns={c: COLS_NAME[i] for i, c in enumerate(cols)})
    return df

def create_from_data(FNAME, exclude_demo=False):
    labs_df = read_csv_fn('labs')
    vitals_df = read_csv_fn('numerics')
    orders_df = read_csv_fn('orders')
    df = pd.concat([labs_df, vitals_df, orders_df])

    if exclude_demo:
        vocab_list = list(set(df['event'].astype(str).unique()))
    else:
        vocab_list = list(set().union(*df['input'])) + list(df['age'].unique()) + list(df['sex'].unique()) + list(df['race'].unique())
    vocab_list = [x for x in vocab_list if x is not None]
    vocab_list = list(set(vocab_list))
    
    special_list = ['[UNK]', '[BOS]', '[EOS]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    vocab_list = special_list + vocab_list

    # Create a dictionary mapping each token to a unique index
    vocab = {token: idx for idx, token in enumerate(vocab_list)}

    # Create a WordLevel tokenizer with the custom vocab and an unknown token
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    tokenizer.save(os.path.join(OUTPUT_DIR, f'{FNAME}_tokenizer.json'))

def create_from_df(FNAME, exclude_demo=False):
    df = pd.read_parquet('/home/mkeoliya/projects/mc-med/data/decomp_data.parquet')

    vocab_list = list(set(df['eventval'].astype(str).unique()))
    vocab_list = [x for x in vocab_list if x is not None]
    vocab_list = list(set(vocab_list))
    
    special_list = ['[UNK]', '[BOS]', '[EOS]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    vocab_list = special_list + vocab_list

    # Create a dictionary mapping each token to a unique index
    vocab = {token: idx for idx, token in enumerate(vocab_list)}

    # Create a WordLevel tokenizer with the custom vocab and an unknown token
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    tokenizer.save(os.path.join(OUTPUT_DIR, f'{FNAME}_tokenizer.json'))

create_from_df("decomp", exclude_demo=True)