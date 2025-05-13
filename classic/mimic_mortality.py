import pandas as pd
import numpy as np
from scipy.stats import linregress, kurtosis, skew

TIME = 12.0

def remove_rows(label_df, time, stay=True):
    df_final = None
    for i in range(0, 37):
        df = pd.read_parquet(f"data/mortality/{i}_final.parquet")
        df = df.merge(label_df[['subject_id', 'stay_id', 'intime', 'outtime', 'deathtime']], on='subject_id')
        
        if stay:
            # df = df[(df['stay_id_x'].isna()) | (df['stay_id_x'] == df['stay_id_y'])]
            df = df[(df['stay_id_x'] == df['stay_id_y'])]

        df['time'] = pd.to_datetime(df['time'])
        # df['intime'] = pd.to_datetime(df['intime'])
        df['outtime'] = pd.to_datetime(df['outtime'])
        df['deathtime'] = pd.to_datetime(df['deathtime'])

        df = df[((~df['deathtime'].isna()) & (df['time'] <= (df['deathtime'] - time))) | 
                ((df['deathtime'].isna()) & (df['time'] <= (df['outtime'] - time)))]
        
        df = df.drop(columns=['intime', 'outtime', 'deathtime'])
        if df_final is None:
            df_final = df
        else:
            df_final = pd.concat([df_final, df], ignore_index=True)
    return df

def mortality_loader():
    label = pd.read_csv(f"data/mortality/mortality_labels.csv")
    demo = pd.read_parquet(f"data/demographics.parquet", columns=['age', 'ethnicity', 'race_ind', 'gender_ind'])
    demo['subject_id'] = pd.to_numeric(demo.index)
    demo = demo[demo['subject_id'].isin(label['subject_id'])]
    
    N = pd.Timedelta(hours=TIME)
    print(N)
    data = remove_rows(label, N)
    data = run_get_classic(data)

    data = data.merge(label[['subject_id', 'y_true']], on='subject_id')
    data = data.merge(demo, on='subject_id')

    X = data.drop(columns=['y_true', 'subject_id'])
    y = data[['y_true']]
    ids = data[['subject_id']]
    print(data['y_true'].value_counts())
    
    return X, y, ids

def run_get_classic(df, start=0, end=30):
    codes = df[['subject_id', 'itemid']].groupby('itemid')['subject_id'].nunique().sort_values(ascending=False).reset_index()
    codes = codes['itemid'].unique()[:30]
    
    df = df[df['itemid'].isin(codes)]
    df_grouped = None
    for c in codes[start:end]:
        print(c)
        df_i = df[df['itemid'] == c].copy()
        unique_vals = df_i['value'].dropna().unique()

        if len(unique_vals) <= 1:
            df2 = df_i.groupby('subject_id')['itemid'].count().rename(f'{c}_count').reset_index()
        else:
            df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
            df_i = df_i.dropna()
            df2 = df_i.groupby('subject_id')['value'].apply(list).apply(extract_stats).reset_index()
            df2.columns = ['subject_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='subject_id', how='outer')
        print(len(df_grouped))

    df_grouped = df_grouped.loc[:, df_grouped.isnull().mean() < 0.9]
    count_columns = [col for col in df_grouped.columns if '_count' in col]
    df_grouped[count_columns] = df_grouped[count_columns].fillna(0)

    return df_grouped

def extract_stats(xs):
    xs = np.array(xs, dtype=float)
    x = np.arange(len(xs))
    slope, _, _, _, _ = linregress(x, xs)

    return pd.Series({
        'min': xs.min(),
        'max': xs.max(),
        'mean': xs.mean(),
        'std': xs.std(),
        'median': np.median(xs),
        'skew': skew(xs),
        'kurtosis': kurtosis(xs),
        'slope': slope,
        'qr_25': np.quantile(xs, 0.25),
        'qr_75': np.quantile(xs, 0.75),
    })