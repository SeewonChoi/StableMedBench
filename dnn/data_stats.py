import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_event_count(df):
    df = df[['eventval', 'Label']]

    df['event_length'] = df['eventval'].apply(len)
    bin_edges = range(0, df['event_length'].max() + 10, 10)
    df['length_bin'] = pd.cut(df['event_length'], bins=bin_edges, right=False)

    grouped = df.groupby(['length_bin', 'Label']).size().unstack(fill_value=0)
    tick_labels = [interval.left for interval in grouped.index]
    x = range(len(grouped.index))
    tick_step = 5

    labels = grouped.columns
    plt.figure(figsize=(12, 6))
    plt.bar(x, grouped[labels[0]], width=0.4, align='center', label=labels[0], alpha=0.7)
    plt.bar(x, grouped[labels[1]], width=0.25, align='center', label=labels[1], alpha=0.7)

    plt.xticks(
        ticks=np.arange(0, len(tick_labels), tick_step),
        labels=[tick_labels[i] for i in range(0, len(tick_labels), tick_step)]
    )
    plt.xlabel('# Events')
    plt.ylabel('# Encounters')
    plt.title('Number of Events per Patient')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0)
    plt.savefig('output.png')

    avg_lengths = df.groupby('Label')['event_length'].mean()
    print(avg_lengths)

    grouped_normalized = grouped.divide(grouped.sum())
    labels = grouped_normalized.columns
    x = range(len(grouped_normalized.index))
    tick_labels = [interval.left for interval in grouped_normalized.index]

    plt.figure(figsize=(12, 6))
    plt.bar(x, grouped_normalized[labels[0]], width=0.4, align='center', label=labels[0], alpha=0.7)
    plt.bar(x, grouped_normalized[labels[1]], width=0.25, align='center', label=labels[1], alpha=0.7)
    plt.xticks(
        ticks=np.arange(0, len(tick_labels), tick_step),
        labels=[tick_labels[i] for i in range(0, len(tick_labels), tick_step)]
    )
    plt.xlabel('# Events')
    plt.ylabel('Fraction of Encounters')
    plt.title('Proportional Number of Events')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0)
    plt.savefig('normal.png')