import pandas as pd

from dataset import extract_stats

TIME = 1.0

def hypoglycemia_loader():
    return lab_task_loader('hypoglycemia')

def hyperkalemia_loader():
    return lab_task_loader('hyperkalemia')

def remove_rows(df, label_df, time):
    df = df.merge(label_df[['patient_id', 'Time', 'start_time']], on='patient_id')
    df['start'] = pd.to_datetime(df['start'])

    # df = df[df['Time'] - df['start_time'] >= time]
    df = df[df['start'] >= df['start_time']]
    df = df[df['start'] <= df['Time'] - time]
    df = df.drop(columns=['start', 'Time', 'start_time'])
    return df

def lab_task_loader(task_type, task_prefix='lab_', multi=False):
    data = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}2.parquet", columns=['code', 'value', 'patient_id', 'start'])
    demo = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}_demo.parquet", columns=['patient_id', 'age', 'gender_ind', 'race_ind', 'ethnicity_ind'])
    label = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}_label2.parquet")
    label['Time'] = label.apply(lambda x: x['prediction_time'] if x['value'] else x['Sample_time'], axis=1)
    label = label.rename(columns={'start': 'start_time'})

    if multi:
        if 'value_multi' in label.columns:
            label['value'] = label['value_multi']
            label = label.drop(columns=['value_multi'])
    else:
        if 'value_multi' in label.columns:
            label = label.drop(columns=['value_multi'])
    
    N = pd.Timedelta(hours=TIME)
    print(N)
    data = remove_rows(data, label, N)
    data = run_get_classic(data)

    data = data.merge(label[['patient_id', 'value']], on='patient_id')
    data = data.merge(demo, on='patient_id')

    X = data.drop(columns=['value', 'patient_id'])
    y = data[['value']]
    ids = data[['patient_id']]
    print(data['value'].value_counts())
    
    return X, y, ids


def run_get_classic(df, start=0, end=30):
    codes = df[~df['code'].str.contains('CARE|Visit|Domain|Race|Ethnicity|Gender|CMS|OMOP|Medicare')]
    codes.groupby('code')['patient_id'].nunique().sort_values(ascending=False).reset_index()
    codes = codes['code'].unique()
    
    df = df[df['code'].isin(codes)]
    df_grouped = None
    for c in codes[start:end]:
        print(c)
        df_i = df[df['code'] == c].copy()
        unique_vals = df_i['value'].dropna().unique()

        if len(unique_vals) <= 1:
            df2 = df_i.groupby('patient_id')['code'].count().rename(f'{c}_count').reset_index()
        else:
            df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
            df_i = df_i.dropna()
            df2 = df_i.groupby('patient_id')['value'].apply(list).apply(extract_stats).reset_index()
            df2.columns = ['patient_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='patient_id', how='outer')
        print(len(df_grouped))

    df_grouped = df_grouped.loc[:, df_grouped.isnull().mean() < 0.9]
    count_columns = [col for col in df_grouped.columns if '_count' in col]
    df_grouped[count_columns] = df_grouped[count_columns].fillna(0)

    return df_grouped