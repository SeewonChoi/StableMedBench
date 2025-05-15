import logging
import json
import os
import time
import importlib.util
import sys
import argparse
from config_loader import load_config, set_config, get_config

from finetune import main_training_pipeline
from inference import run_inference
from evaluate import evaluate_predictions, evaluate_predictions_stability

def setup_logging():
    config = get_config()
    """Sets up logging to file and console."""
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_LEVEL.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


def load_config(config_file_path):
    """Loads configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    if spec is None:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module  # Make it available in sys.modules
    spec.loader.exec_module(config_module)
    return config_module

def main():
    """Orchestrates the prediction task."""

    # set up config
    parser = argparse.ArgumentParser(description="Main script for LLM tasks.")
    parser.add_argument("--config", type=str, required=True, help="Path to the project configuration file.", default="/home/mkeoliya/projects/arpa-h/llm/tasks/mc_med_sepsis/configs/llama_inf.py")
    parser.add_argument("--num_inf_rows", type=int, default=None, help="Limit the number of test samples to process.")
    parser.add_argument("--num_finetune_rows", type=int, default=None, help="Limit the number of test samples to process.")
    parser.add_argument("--stability", action="store_true", help="Enable stability mode.")
    args = parser.parse_args()

    # config stuff
    config = load_config(args.config)
    config.NUM_INF_ROWS = args.num_inf_rows
    config.NUM_FINETUNE_ROWS = args.num_finetune_rows
    config.STABILITY = args.stability
    set_config(config)

    # start logging
    setup_logging()
    logging.info(f"Configuration loaded from {args.config}")
    logging.info("Starting prediction workflow.")

    finetuned_adapter_path = None
    finetune_duration = 0
    model_used_for_inference = config.BASE_MODEL_ID

    # --- Step 1: Finetune the model ---
    if not config.SKIP_FINETUNING:
        if not os.path.exists(config.FINETUNED_ADAPTER_DIR):
            logging.info(f"Finetuned adapter directory {config.FINETUNED_ADAPTER_DIR} does not exist. Creating it.")
            logging.info("--- Finetuning Step ---")
            start_time = time.time()
            try:
                finetuned_adapter_path = main_training_pipeline("qlora")
                logging.info(f"Finetuning completed. Adapter saved to {finetuned_adapter_path}")
                model_used_for_inference = f"{config.BASE_MODEL_ID} + PEFT Adapter ({finetuned_adapter_path})"
                logging.info(f"Model used for inference: {model_used_for_inference}")
            except Exception as e:
                logging.error(f"Error during finetuning: {e}", exc_info=True)
                logging.critical("Workflow stopped due to finetuning error.")
                return
            finetune_duration = time.time() - start_time
            logging.info(f"Finetuning duration: {finetune_duration:.2f} seconds")
        else:
            finetuned_adapter_path = config.FINETUNED_ADAPTER_DIR
    else:
        logging.info("--- Finetuning Skipped ---")
        logging.info(f"Running inference directly on base model: {config.BASE_MODEL_ID}")


    # --- Step 2: Run Inference on Test Data ---

    # if config.RESULTS_FILE already exists, skip inference
    if os.path.exists(config.RESULTS_FILE):
        logging.info(f"Results file {config.RESULTS_FILE} already exists. Skipping inference.")
        with open(config.RESULTS_FILE, 'r') as f:
            results = json.load(f)
        all_preds = results['all_predictions']
        true_labels = [p['TrueLabel'] for p in all_preds]
        predicted_labels_bool = [result['PredictedLabel'] for result in all_preds]
        csns = [p['CSN'] for p in all_preds]
        predicted_probs = [p['PredictedProb'] for p in all_preds]
        times = [p['TimeUpto'] for p in all_preds]

        extracted_texts = [p['ExtractedText'] for p in results['sample_predictions']]
        test_prompts = [p['TestPrompt'] for p in results['sample_predictions']]
        inference_duration = results['inference_duration_sec']

        logging.info("Inference results loaded from existing file.")
    else:
        logging.info("--- Inference Step ---")
        start_time = time.time()
        try:
            if args.stability:
                true_labels, predicted_labels_bool, extracted_texts, csns, test_prompts, predicted_probs, times = run_inference(config.BASE_MODEL_ID, finetuned_adapter_path, stability=args.stability)
            else:
                true_labels, predicted_labels_bool, extracted_texts, csns, test_prompts, predicted_probs = run_inference(config.BASE_MODEL_ID, finetuned_adapter_path, stability=args.stability)
            logging.info("Inference completed.")
        except Exception as e:
            logging.error(f"Error during inference: {e}", exc_info=True)
            logging.critical("Workflow stopped due to inference error.")
            return
        inference_duration = time.time() - start_time
        logging.info(f"Inference duration: {inference_duration:.2f} seconds")


    # --- Step 3: Evaluate Results ---
    logging.info("--- Evaluation Step ---")
    try:
        if args.stability:
            evaluation_metrics = evaluate_predictions_stability(csns, true_labels, predicted_labels_bool, predicted_probs, times)
        else:
            evaluation_metrics = evaluate_predictions(true_labels, predicted_labels_bool, predicted_probs)
        logging.info("Evaluation completed.")
        logging.info(f"Final Metrics: Precision={evaluation_metrics['precision']:.4f}, Recall={evaluation_metrics['recall']:.4f}, F1={evaluation_metrics['f1_score']:.4f}, AUPRC={evaluation_metrics['auprc']:.4f}, AUROC={evaluation_metrics['auroc']:.4f}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}", exc_info=True)
        logging.critical("Workflow completed with evaluation error.")
        return


    # --- Step 4: Log and Save Results ---
    results = {
        "config": {k: v for k, v in globals().items() if not k.startswith('_') and k.isupper()}, # Log relevant config
        "inference_duration_sec": inference_duration,
        "evaluation_metrics": evaluation_metrics,
        "finetuning_performed": not config.SKIP_FINETUNING,
        "finetune_duration_sec": finetune_duration if not config.SKIP_FINETUNING else "N/A",
        # Optional: Log sample predictions for inspection
        "sample_predictions": [{
            "CSN": csns[i],
            "TrueLabel": true_labels[i],
            "PredictedLabel": predicted_labels_bool[i],
            "TimeUpto": times[i] if args.stability else None,
            "PredictedProb": predicted_probs[i],
            "TestPrompt": test_prompts[i],
            "ExtractedText": extracted_texts[i]
        } for i in range(min(100, len(csns)))], # Log first 50 samples
        "all_predictions": [{
            "CSN": csns[i],
            "TrueLabel": true_labels[i],
            "PredictedLabel": predicted_labels_bool[i],
            "TimeUpto": times[i] if args.stability else None,
            "PredictedProb": predicted_probs[i],
            # "TestPromptLength": len(test_prompts[i]),
            # "GeneratedTextLength": len(extracted_texts[i]),
        } for i in range(len(csns))] # Log all samples
    }

    try:
        with open(config.RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {config.RESULTS_FILE}")
    except Exception as e:
        logging.error(f"Error saving results file: {e}", exc_info=True)


    logging.info("Sepsis prediction workflow finished.")

if __name__ == "__main__":
    main()