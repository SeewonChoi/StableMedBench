import os
import re
import torch
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from peft import PeftModel
from vllm.lora.request import LoRARequest

from config_loader import get_config
from data_processing import load_data, format_data_for_inference

def run_inference(base_model_id: str, adapter_path: str | None, stability=False):
    config = get_config()
    
    """Runs inference on the test data using vLLM."""
    model_to_load = adapter_path if adapter_path and os.path.exists(adapter_path) else base_model_id
    print(f"Starting inference process using model: {model_to_load}")

    # Load tokenizer: Use adapter path if available, otherwise use the base model id
    tokenizer_path = adapter_path if adapter_path and os.path.exists(adapter_path) else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    # Initialize vLLM
    # vLLM can load PEFT adapters directly or load a merged model.
    # Loading the adapter with the base model is generally preferred.
    print(f"Initializing vLLM with base model {base_model_id}" # String part 1
          f"{' and adapter ' + adapter_path if adapter_path and os.path.exists(adapter_path) else ''}" # Conditional part 2 using f-string
          )
    
    if adapter_path and os.path.exists(adapter_path):
        llm = LLM(
            model=base_model_id,
            enable_lora=True, # Enable LoRA for vLLM
            # peft_adapter=adapter_path, # Point vLLM to the PEFT adapter directory
            tokenizer=tokenizer_path, # Use the tokenizer saved with the adapter
            tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
            max_model_len=config.VLLM_MAX_MODEL_LEN,
            dtype=config.VLLM_DTYPE,
        )
    else:
        # If no adapter path is provided, load the base model directly
        # This is useful for testing the base model without any finetuning
        print("No adapter path provided. Loading base model directly.")
        # Note: vLLM does not support loading PEFT models directly, so we need to load the base model
        # and then apply the adapter weights manually if needed.
        llm = LLM(
            model=base_model_id,
            tokenizer=tokenizer_path, # Use the tokenizer saved with the adapter
            tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
            max_model_len=config.VLLM_MAX_MODEL_LEN,
            dtype=config.VLLM_DTYPE,
        )
    print("vLLM initialized.")

    # Load and format test data
    print("Loading and formatting test data for inference...")
    test_df = load_data(config.TEST_FILE)

    # keep just the first X rows
    if config.NUM_INF_ROWS:
        print(f"Limiting test data to {config.NUM_INF_ROWS} rows for quick testing.")
        test_df = test_df.head(config.NUM_INF_ROWS) # Uncomment for testing with a smaller subset
        
    test_prompts = format_data_for_inference(test_df, config.PROMPT_TEMPLATE_FILE, tokenizer)
    true_labels = test_df[config.LABEL_COLUMN].tolist()
    csns = test_df[config.CSN_COLUMN].tolist()
    if stability:
        times = test_df[config.TIMEUPTO_COLUMN].tolist()
    print(f"Prepared {len(test_prompts)} prompts for inference.")

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=config.VLLM_TEMPERATURE,
        top_p=config.VLLM_TOP_P,
        max_tokens=config.VLLM_MAX_TOKENS,
        stop=config.VLLM_STOP_SEQUENCES,
        # Add logprobs=1 to potentially get probabilities if needed for analysis,
        # but not required for simple prediction extraction.
        # logprobs=1,
    )
    print("Sampling parameters configured.")

    # Run inference
    print("Running inference with vLLM...")
    outputs = llm.generate(test_prompts, sampling_params)
    print("Inference finished.")

    # Process outputs and extract predictions
    predicted_labels = []
    extracted_texts = []
    predicted_probs = []
    print("Processing vLLM outputs...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        extracted_texts.append(generated_text)

        # Simple parsing logic: Look for "Prediction: Yes" or "Prediction: No"
        # Make this case-insensitive and robust
        prediction = False # Default to predicting False (No sepsis)
        if "Prediction: Yes" in generated_text or "Yes" in generated_text:
            prediction = True
        # If both "Yes" and "No" appear, the *first* one is likely the intended prediction.
        # This simple check assumes "Yes" appearing anywhere means Yes, unless "No" appears *before* it.
        # A more robust regex or sequence check could be implemented.
        # Given the prompt format, the first word after "Prediction:" should be "Yes" or "No".
        # Let's refine the parsing slightly to look right after the tag.
        tag = "Prediction:"
        if tag in generated_text:
            after_tag = generated_text.split(tag, 1)[1].strip().lower()
            if after_tag.startswith("yes"):
                 prediction = True
            elif after_tag.startswith("no"):
                 prediction = False # Explicitly set to False if No is found first
        
        # Let's extract the probability of the prediction if available, "Risk: 0.95"
        # This is a simple regex to find the confidence score.
        confidence_match = re.search(r"Risk:\s*([0-9]*\.?[0-9]+)", generated_text)
        if confidence_match:
            confidence_score = float(confidence_match.group(1))
            predicted_probs.append(confidence_score)
        else:
            predicted_probs.append(None) # No confidence score found

        predicted_labels.append(prediction)
    print("Output processing complete.")

    # Map boolean predictions back to the string format if needed for logging,
    # but evaluation functions typically expect 0/1 or boolean.
    # Let's keep them as booleans for evaluation.

    # Optionally, return the extracted texts for debugging
    if stability:
        return true_labels, predicted_labels, extracted_texts, test_df[config.CSN_COLUMN].tolist(), test_prompts, predicted_probs, times
    return true_labels, predicted_labels, extracted_texts, test_df[config.CSN_COLUMN].tolist(), test_prompts, predicted_probs


if __name__ == "__main__":
    # Example usage: Run inference after finetuning
    # Ensure the finetuned_adapter directory exists with adapter weights and tokenizer
    config = get_config()
    FINETUNED_ADAPTER_DIR = config.FINETUNED_ADAPTER_DIR
    BASE_MODEL_ID = config.BASE_MODEL_ID
    if not os.path.exists(FINETUNED_ADAPTER_DIR):
        print(f"Finetuned adapter directory not found at {FINETUNED_ADAPTER_DIR}.")
        print(f"Running inference on base model: {BASE_MODEL_ID}")
        true_labels, predicted_labels, extracted_texts, csns = run_inference(BASE_MODEL_ID, None)
    else:
        print(f"Running inference on finetuned model from: {FINETUNED_ADAPTER_DIR}")
        true_labels, predicted_labels, extracted_texts, csns = run_inference(BASE_MODEL_ID, FINETUNED_ADAPTER_DIR)
        print("\n--- Sample Results ---")
        for i in range(min(5, len(csns))):
            print(f"CSN: {csns[i]}")
            print(f"True Label: {true_labels[i]}")
            print(f"Predicted Label: {predicted_labels[i]}")
            print(f"Extracted Text: {extracted_texts[i]}")
            print("-" * 20)