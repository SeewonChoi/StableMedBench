from torcheval.metrics.functional import binary_auprc
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from collections import defaultdict
import logging

from config_loader import get_config

# compute the lipschitz constant for each CSN, following the paper
def _compute_lipschitz_constant(entries, c=60):
    # Sort entries by TimeUpto
    entries.sort(key=lambda x: x["TimeUpto"])
    
    # Extract times and predicted probabilities
    times = [entry["TimeUpto"] for entry in entries]
    probs = [entry["PredictedProb"] for entry in entries]
    
    # Initialize max_lipschitz constant
    max_lipschitz_constant = 0.0
    max_t1 = 0
    max_t2 = 0
    
    # Compute Lipschitz constant for each pair of times where the difference is less than c
    # We will use a sliding window approach to compute the Lipschitz constant
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if probs[i] is None or probs[j] is None:
                continue
            if abs(times[i] - times[j]) <= c:
                # Compute Lipschitz constant
                delta_prob = abs(probs[i] - probs[j])
                delta_time = abs(times[i] - times[j])
                if delta_time > 0:  # Avoid division by zero
                    lipschitz_constant = (delta_prob / delta_time) * 60 # Scale to hourly measure
                    max_lipschitz_constant = max(max_lipschitz_constant, lipschitz_constant)
                    # Store the times for which the max Lipschitz constant was found
                    if lipschitz_constant == max_lipschitz_constant:
                        max_t1 = times[i]
                        max_t2 = times[j]
    return max_lipschitz_constant, max_t1, max_t2

def compute_lipschitz_constants_for_all_csns(csns, predicted_probs, times):
    
    grouped_results = defaultdict(list)
    for i in range(len(csns)):
        grouped_results[csns[i]].append({
            "TimeUpto": times[i],
            "PredictedProb": predicted_probs[i],
        })
    
    print("Grouped results by CSN:", grouped_results)
    filtered_grouped_results = {csn: entries for csn, entries in grouped_results.items() if len(entries) >= 3}

    lipschitz_results = {}
    for csn, entries in filtered_grouped_results.items():
        lipschitz_constant, max_t1, max_t2 = _compute_lipschitz_constant(entries)
        lipschitz_results[csn] = lipschitz_constant
        logging.info(f"CSN: {csn}, Lipschitz Constant: {lipschitz_constant:.4f} Max times: {max_t1}, {max_t2}")
        # plot times vs predicted probabilities along with the Lipschitz constant, mark the points with a circle
        # and also join the points with a line
        import matplotlib.pyplot as plt
        import numpy as np
        times = [entry["TimeUpto"] for entry in entries]
        probs = [entry["PredictedProb"] for entry in entries]
        plt.figure(figsize=(10, 5))
        plt.plot(times, probs, marker='o', linestyle='-', label=f"CSN: {csn}")
        plt.xlabel("TimeUpto")
        plt.ylabel("Predicted Probability")
        plt.title(f"Predicted Probability vs TimeUpto, Lipschitz Constant: {lipschitz_constant:.4f}")
        plt.legend()
        plt.show()
        # Save the plot
        plt.savefig(f"plots/lipschitz_constant_{csn}.png")
        plt.close()
    
    # compute the average Lipschitz constant over all CSNs
    average_lipschitz_constant = sum(lipschitz_results.values()) / len(lipschitz_results)
    return lipschitz_results, average_lipschitz_constant


def evaluate_predictions_stability(csns: list, true_labels: list, predicted_labels: list, predicted_probs: list, times: list) -> dict:
    """Calculates precision, recall, and F1 score."""
    print("Starting evaluation...")

    config = get_config()
    POSITIVE_LABEL = config.POSITIVE_LABEL

    # Ensure labels are in a format compatible with sklearn (0/1)
    # Assuming POSITIVE_LABEL in config is the value for the positive class.
    y_true = [1 if label == POSITIVE_LABEL else 0 for label in true_labels]
    y_pred = [1 if prediction else 0 for prediction in predicted_labels]

    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch between number of true labels and predicted labels.")

    if len(y_true) == 0:
        print("Warning: No samples to evaluate.")
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "num_samples": 0}


    # Calculate metrics. Use zero_division=0 to avoid warnings/errors if no positive predictions are made.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Let's calculate the AUROC using the predicted probabilities, filtering away the "None" values 
    # and ensuring the predicted probabilities are in the same format as y_true.
    y_true_filtered = [y for y, pr in zip(y_true, predicted_probs) if pr is not None]
    y_pred_probs = [pr for pr in predicted_probs if pr is not None]
    auroc = roc_auc_score(y_true_filtered, y_pred_probs)

    # Let's also compute the AUPRC (Precision-Recall Curve) using torchmetrics
    y_true_tensor = torch.tensor(y_true_filtered)
    y_pred_probs_tensor = torch.tensor(y_pred_probs)
    auprc = binary_auprc(y_pred_probs_tensor, y_true_tensor).item()

    # average the Lipschitz constant over all CSNs
    lipschitz_results, avg_lipschitz = compute_lipschitz_constants_for_all_csns(csns, predicted_probs, times)
    print(f"Average Lipschitz Constant over all CSNs: {avg_lipschitz:.4f}")

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
        "auprc": auprc,
        "num_samples": len(y_true),
        "num_positive_true": sum(y_true),
        "num_positive_predicted": sum(y_pred),
        "avg_lipschitz_constant": avg_lipschitz,
    }

    print("Evaluation complete.")
    print(f"Results: {metrics}")

    return metrics

def evaluate_predictions(true_labels: list, predicted_labels: list, predicted_probs: list) -> dict:
    """Calculates precision, recall, and F1 score."""
    print("Starting evaluation...")

    config = get_config()
    POSITIVE_LABEL = config.POSITIVE_LABEL

    # Ensure labels are in a format compatible with sklearn (0/1)
    # Assuming POSITIVE_LABEL in config is the value for the positive class.
    y_true = [1 if label == POSITIVE_LABEL else 0 for label in true_labels]
    y_pred = [1 if prediction else 0 for prediction in predicted_labels]

    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch between number of true labels and predicted labels.")

    if len(y_true) == 0:
        print("Warning: No samples to evaluate.")
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "num_samples": 0}


    # Calculate metrics. Use zero_division=0 to avoid warnings/errors if no positive predictions are made.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Let's calculate the AUROC using the predicted probabilities, filtering away the "None" values 
    # and ensuring the predicted probabilities are in the same format as y_true.
    y_true_filtered = [y for y, pr in zip(y_true, predicted_probs) if pr is not None]
    y_pred_probs = [pr for pr in predicted_probs if pr is not None]
    auroc = roc_auc_score(y_true_filtered, y_pred_probs)

    # Let's also compute the AUPRC (Precision-Recall Curve) using torchmetrics
    y_true_tensor = torch.tensor(y_true_filtered)
    y_pred_probs_tensor = torch.tensor(y_pred_probs)
    auprc = binary_auprc(y_pred_probs_tensor, y_true_tensor).item()

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
        "auprc": auprc,
        "num_samples": len(y_true),
        "num_positive_true": sum(y_true),
        "num_positive_predicted": sum(y_pred),
    }

    print("Evaluation complete.")
    print(f"Results: {metrics}")

    return metrics

if __name__ == "__main__":
    # Example usage
    sample_true = [True, False, True, True, False, False]
    sample_pred = [True, True, False, True, False, True]
    sample_probs = [0.1, 0.5, 0.7, 0.8, 0.1, 0.9]  # Example probabilities
    metrics = evaluate_predictions(sample_true, sample_pred, sample_probs)
    print(metrics)

    sample_true_int = [1, 0, 1, 1, 0, 0]
    sample_pred_int = [1, 1, 0, 1, 0, 1]
    sample_probs_int = [0.9, 0.1, 0.8, None, 0.2, 0.9]  # Example probabilities
    metrics_int = evaluate_predictions(sample_true_int, sample_pred_int, sample_probs_int)
    print(metrics_int)