import json
import pandas as pd

task = 'sepsis'
dataset = 'mc_med'

# Open and read the JSON file
with open(f'stability_results/{dataset}_{task}/qwen32b.json', 'r') as file:
    data = json.load(file)

predictions = data['all_predictions']

all_probs = {}
all_times = {}
correct_labels = {}
for prediction in predictions:
    csn = prediction['CSN']
    if not csn in correct_labels.keys():
        correct_labels[csn] = prediction['TrueLabel']
    if csn in all_probs.keys():
        all_probs[csn] = all_probs[csn] + [(prediction['TimeUpto'], prediction['PredictedProb'])]
    else:
        all_probs[csn] = [(prediction['TimeUpto'], prediction['PredictedProb'])]
    

filtered_probs = {}
for k in all_probs.keys():
    data_k = all_probs[k]
    if len(all_probs[k]) < 3: continue
    data_k = sorted(data_k, key=lambda x: x[0])
    data_k = [(t[0] - data_k[0][0], t[1]) for t in data_k]
    filtered_probs[k] = data_k

rows = []
for key, tuples in filtered_probs.items():
    row = {'key': key}
    for t, v in tuples:
        row[t] = v
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# df = df.drop(columns=[None])
df = df[['key'] + sorted(df.columns[1:])]
# Save to CSV
df.to_csv(f'{task}.csv', index=False)

print(df)