import pandas as pd

from dataset import extract_stats

TIME = 4.0

def remove_rows(df, label_df):
    df['charttime'] = pd.to_datetime(df['charttime'])
    df = df.merge(label_df[['stay_id', 'intime', 'Time']], on='stay_id')
    df['charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600
    
    df = df[df['Time'] >= TIME]
    df = df[df['charttime'] <= df['Time'] - TIME]
    df = df.drop(columns=['intime', 'charttime', 'Time'])
    return df

def icu_loader():
    label = pd.read_csv(f"data/transfer/label.csv", usecols=['stay_id', 'intime', 'outcome_icu_transfer_12h', 'Time'])
    label['intime'] = pd.to_datetime(label['intime'])

    demo = pd.read_csv(f"data/transfer/demo.csv", usecols=['stay_id', 'gender_ind', 'race_ind', 'age'])
    demo = demo[demo['stay_id'].isin(label['stay_id'])]

    vital_df = pd.read_csv(f"data/transfer/numerics.csv")
    vital_df = remove_rows(vital_df, label)
    # vital_df = vital_df[:200]

    med_df = pd.read_csv(f"data/transfer/med.csv", index_col=0, usecols=['stay_id', 'group', 'charttime'])
    med_df = remove_rows(med_df, label)

    df_final = vital_df[['stay_id']]
    vital_df = vital_df.groupby('stay_id').agg(lambda x: x.dropna().tolist())
    for c in vital_df.columns[1:]:
        df_c = vital_df[c]
        df_c = df_c[df_c.apply(len) > 0].apply(extract_stats).reset_index()
        df_c.columns = ['stay_id'] + [f'{c}_{col}' for col in df_c.columns[1:]]
        df_final = df_final.merge(df_c, on='stay_id', how='outer')

    med_df = med_df.groupby(['stay_id', 'group']).size().reset_index()
    meds = med_df.groupby('group')['stay_id'].nunique().sort_values(ascending=False).reset_index()['group'].unique()[:22]
    for m in meds:
        med_i = med_df[med_df['group']== m].drop(columns=['group'])
        med_i.columns = ['stay_id', f'{m}_count']
        df_final = df_final.merge(med_i, on='stay_id', how='outer')
    
    df_final = df_final.loc[:, df_final.isnull().mean() < 0.9]
    count_columns = [col for col in df_final.columns if '_count' in col]
    df_final[count_columns] = df_final[count_columns].fillna(0)

    df_final = df_final.merge(label[['stay_id', 'outcome_icu_transfer_12h']], on='stay_id')
    df_final = df_final.merge(demo, on='stay_id')

    X = df_final.drop(columns=['outcome_icu_transfer_12h', 'stay_id'])
    y = df_final[['outcome_icu_transfer_12h']]
    ids = df_final[['stay_id']]
    print(df_final['outcome_icu_transfer_12h'].value_counts())
    
    return X, y, ids