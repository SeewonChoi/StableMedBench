import pandas as pd
import numpy as np
import joblib
from scipy.stats import linregress, kurtosis, skew
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
import pickle 
from itertools import combinations

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

def run_stats(df, codes):
    df_grouped = None
    for c in codes:
        print(c)
        df_i = df[df['itemid'] == c].copy()

        df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
        df_i = df_i.dropna()
        df2 = df_i.groupby('subject_id')['value'].apply(list).apply(extract_stats).reset_index()
        df2.columns = ['subject_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='subject_id', how='outer')
        print(len(df_grouped))
    return df_grouped

def run_count(df, codes):
    df_grouped = None
    for c in codes:
        print(c)
        df_i = df[df['itemid'] == c].copy()
        df2 = df_i.groupby('subject_id')['itemid'].count().rename(f'{c}_count').reset_index()
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='subject_id', how='outer')
        print(len(df_grouped))
    count_columns = [col for col in df_grouped.columns if '_count' in col]
    df_grouped[count_columns] = df_grouped[count_columns].fillna(0)
    return df_grouped

def impute(X_test, imputer):
    features = X_test.columns
    X_test = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=features)
    return X_test

def compute_results(y_pred, y_pred_prob, y_test, metrics):
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)

    # Save metrics
    metrics['acc'].append(acc)
    metrics['auc'].append(auc)
    metrics['f1'].append(f1)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['auprc'].append(auprc)

def get_model_features(model):
    features = model.feature_names_in_
    features = list(filter(lambda x: not ('ind' in x or 'Age' in x), features))

    count_features = list(filter(lambda x: 'count' in x, features))
    stat_features = list(filter(lambda x: not 'count' in x, features))

    stat_codes = list(set(['_'.join(x.split('_')[:-1]) for x in stat_features]))
    count_codes = list(set(['_'.join(x.split('_')[:-1]) for x in count_features]))
    return stat_codes, count_codes

def get_stability(method, task, model_name, i):
    metrics = { 'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []}
    
    if model_name == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(f'results/{task}/{model_name}_{task}_{method}_{i}.json')
    else:
        model = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}.pkl')

    imputer = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}_imputer.pkl')
    stat_codes, count_codes = get_model_features(model)

    label_ids = pd.read_csv(f'results/{task}/{model_name}_{task}_{method}_{i}.csv')
    label_df = pd.read_csv(f"data/mortality/mortality_labels.csv", usecols=['subject_id', 'stay_id', 'outtime', 'deathtime', 'y_true'])
    demo_test = pd.read_parquet(f"data/demographics.parquet", columns=['age', 'ethnicity', 'race_ind', 'gender_ind'])
    demo_test['subject_id'] = pd.to_numeric(demo_test.index)

    df = None
    for j in range(0, 37):
        df_i = pd.read_parquet(f"data/mortality/{j}_final.parquet")
        df_i = df_i[df_i['subject_id'].isin(label_ids['subject_id'])]
        if df is None: df = df_i
        else: df = pd.concat([df, df_i], ignore_index=True)
    df = df[(df['itemid'].isin(stat_codes)) | (df['itemid'].isin(count_codes))]

    label_df = label_df[label_df['subject_id'].isin(label_ids['subject_id'])]
    demo_test = demo_test[demo_test['subject_id'].isin(label_ids['subject_id'])]

    df = df.merge(label_df[['subject_id', 'outtime', 'deathtime']], on='subject_id')

    df['time'] = pd.to_datetime(df['time'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    df['deathtime'] = pd.to_datetime(df['deathtime'])

    df['Time'] = df.apply(lambda x: x['outtime'] if (x['deathtime'] is pd.NaT) else x['deathtime'], axis=1)
    df = df[df['time'] <= (df['Time'])]
    df['time'] = df['Time'] - df['time']
    df['time'] = df['time'].dt.total_seconds() / 3600
    df = df.drop(columns=['outtime', 'deathtime', 'Time'])

    TIMES = df[(df['time'] <= 30) & (df['time'] >= 1)].sort_values('time')['time'].unique()

    for n, TIME in enumerate(TIMES):
        df_i = df[df['time'] >= TIME]
        if n == 0:
            valid_ids = df_i[(df_i['time'] >= TIME) & (df_i['time'] <= 12.0)]['subject_id'].unique()
        else:
            valid_ids = df_i[(df_i['time'] >= TIMES[n-1]) & (df_i['time'] <= TIME)]['subject_id'].unique()
        df_i = df_i[df_i['subject_id'].isin(valid_ids)]
        df_i = df_i.drop(columns=['time'])

        df_count = run_count(df_i, count_codes)
        df_stat = run_stats(df_i, stat_codes)
        data = df_count.merge(df_stat, on='subject_id', how='outer')

        count_columns = [col for col in data.columns if '_count' in col]
        data[count_columns] = data[count_columns].fillna(0)

        data = data.merge(label_df[['subject_id', 'y_true']], on='subject_id')
        data = data.merge(demo_test, on='subject_id')

        X = data.drop(columns=['y_true', 'subject_id'])
        y = data[['y_true']]
        ids = data[['subject_id']]
        print(data['y_true'].value_counts())

        X = X[model.feature_names_in_]
        X = impute(X, imputer)

        predictions = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        ids[TIME] = probs
        label_ids = label_ids.merge(ids, on='subject_id', how='outer')
        compute_results(predictions, probs, y, metrics)

    print(label_ids.iloc[:, 2:].diff(axis=1).abs().max(axis=1).max())
    with open(f'results/{model_name}_{task}_{method}_{i}_smooth.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    label_ids.to_csv(f'results/{model_name}_{task}_{method}_{i}_smooth.csv')


def get_lipschitz():
    df = pd.read_csv(f'results/{MODEL}_{TASK}_{METHOD}_{I}_smooth.csv', index_col=0)
    df = df.dropna(thresh=5)

    df[BASE] = df['probs']
    df = df.drop(columns='probs')

    MIN = df.columns[1]
    df[MIN] = df.apply(lambda x: x[BASE] if np.isnan(x[MIN]) else x[MIN], axis=1)

    for col_idx in range(2, len(df.columns)):
        current_col = df.columns[col_idx]
        prev_col = df.columns[col_idx - 1]
        df[current_col] = df[current_col].combine_first(df[prev_col])

    times = df.columns[1:]
    times = np.array([float(t) for t in times])
    probs = df.iloc[:, 1:].values

    idx_pairs = [(i, j) for i, j in combinations(range(len(times)), 2) if abs(times[i] - times[j]) <= 0.167]
    i_idx = np.array([i for i, _ in idx_pairs])
    j_idx = np.array([j for _, j in idx_pairs])

    delta_times = np.abs(times[i_idx] - times[j_idx])
    delta_probs = np.abs(probs[:, i_idx] - probs[:, j_idx]) 

    lipschitz_constants = delta_probs / delta_times 
    max_lipschitz_per_row = lipschitz_constants.max(axis=1)
    df['lipschitz'] = max_lipschitz_per_row
    # df = df.to_csv(f'results/{MODEL}_{TASK}_{METHOD}_{I}_smooth.csv')

    # [df['lipschitz'] > 0.0]
    x = df['lipschitz'].mean()
    print(x)



if __name__ == "__main__":
    get_stability()

    TASK = 'mortality'
    I = 0
    MODEL = 'xgb'
    METHOD = 'median'
    BASE = '12.0'

    get_lipschitz()