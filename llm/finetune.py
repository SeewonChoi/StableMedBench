import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
# datasets.Dataset is not explicitly used in the refactored main flow but might be used by your data_processing
# from datasets import Dataset

# Your existing imports - ensure these paths are correct
from data_processing import load_data, format_data_for_finetuning
from evaluate import evaluate_predictions
from config_loader import load_config, set_config, get_config # We'll simulate config loading for this example

# New imports for Linear Probe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
# DataLoader and TensorDataset are not strictly needed if get_llm_embeddings handles batching internally for sklearn
# from torch.utils.data import DataLoader, TensorDataset


def get_llm_embeddings(model, tokenizer, texts, max_length, device, batch_size=8):
    """
    Helper function to get embeddings from the LLM for linear probing.
    Extracts the hidden state of the last non-padding token.
    """
    model.eval()  # Set model to evaluation mode
    all_embeddings = []
    
    print(f"Using embedding batch size: {batch_size}")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]
        
        # Get the embeddings of the last non-padding token for each sequence
        if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
            # Sum attention mask to get sequence lengths (number of non-padding tokens)
            sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
        else:
            # Fallback if attention_mask is not available (e.g. some model configs)
            # This assumes no padding or that all sequences are of max_length
            # This part might need adjustment based on actual tokenizer output for specific models without attention_mask
            print("Warning: Attention mask not found. Using last token of max_length sequence for embeddings.")
            sequence_lengths = (torch.ones(last_hidden_states.shape[0], device=last_hidden_states.device) * (inputs.input_ids.shape[1] - 1)).long()

        # Gather the embeddings of the last token for each sequence
        # Ensure sequence_lengths are on the same device as last_hidden_states
        sequence_lengths = sequence_lengths.to(last_hidden_states.device)
        pooled_embeddings = last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]
        
        all_embeddings.append(pooled_embeddings.cpu().to(torch.float32).numpy())
        
    np_embeddings = np.concatenate(all_embeddings, axis=0)

    # save embeddings to a file
    np.save("embeddings.npy", np_embeddings)
    print(f"Embeddings saved to 'embeddings.npy' with shape: {np_embeddings.shape}")

    return np_embeddings


def train_linear_probe(base_model_id, train_file, prompt_template_str, max_seq_length, linear_probe_output_dir, config_params):
    """Trains a linear probe on top of the frozen LLM."""
    print(f"Starting linear probe training for model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # For causal LMs, if taking last token embedding, left padding is often preferred.
        # However, get_llm_embeddings dynamically finds the last non-pad token via attention_mask,
        # so tokenizer's padding_side default ('right') is usually fine.
        # tokenizer.padding_side = 'left' # Optional, set if you have specific reasons
        print(f"Tokenizer pad token set to EOS token: {tokenizer.pad_token_id}. Padding side: {tokenizer.padding_side}")

    # Load base model (not in 4-bit for cleaner embedding extraction if possible)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if config_params.get('BF16_FOR_EMBEDDINGS', config_params['BF16']) else \
                      torch.float16 if config_params.get('FP16_FOR_EMBEDDINGS', config_params['FP16']) else \
                      torch.float32,
        device_map="auto"
    )
    model.eval()  # Ensure model is in eval mode
    print("Base model loaded for linear probe.")

    train_df = load_data(train_file) # Expects columns like 'text' (EHR) and 'label' ('Yes'/'No')
    # shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # TODO: comment me out
    # limit to LIMIT_X rows for testing
    if config_params['LIMIT_X'] is not None:
        train_df = train_df.head(config_params['LIMIT_X'])
    
    formatted_texts = []
    labels = []
    # Ensure 'text_column' and 'label_column' are correctly specified or match defaults in load_data/train_df
    # Defaulting to 'text' and 'label' for this example
    text_column = config_params.get('TEXT_COLUMN_NAME', 'Text')
    label_column = config_params.get('LABEL_COLUMN_NAME', 'Label')

    print(f"Formatting texts for linear probe using text column: '{text_column}' and label column: '{label_column}'")
    for _, row in train_df.iterrows():
        input_text = prompt_template_str.format(ehr_record=row[text_column])
        formatted_texts.append(input_text)
        # Ensure labels are consistently 'yes'/'no' or convert appropriately
        labels.append(1 if row[label_column] else 0)

    print(f"Number of samples for linear probe: {len(formatted_texts)}")
    if not formatted_texts:
        print("Error: No data to process for linear probe. Check train_file and column names.")
        return None

    device = next(model.parameters()).device # Get the device the model (or part of it) is on
    print(f"Extracting embeddings using device: {device}")
    
    embedding_batch_size = config_params.get('EMBEDDING_BATCH_SIZE', 8)
    X = get_llm_embeddings(model, tokenizer, formatted_texts, max_seq_length, device, batch_size=embedding_batch_size)
    y = np.array(labels)

    del model # Clean up LLM from memory
    torch.cuda.empty_cache()
    print("LLM removed from memory after embedding extraction.")

    print("Training linear classifier...")
    # Stratify by y if classes are imbalanced for train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")
    print("Positive class distribution in training set:", np.bincount(y_train))
    print("Positive class distribution in validation set:", np.bincount(y_val))
    
    classifier = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)
    print("Linear classifier trained.")

    # get predicted_probs
    y_pred, y_pred_probs = classifier.predict(X_val), classifier.predict_proba(X_val)[:, 1]
    
    accuracy = classifier.score(X_val, y_val)
    print(f"Linear probe validation accuracy: {accuracy:.4f}")

    # Evaluate predictions
    # evaluate_predictions(csns: list, true_labels: list, predicted_labels: list, predicted_probs: list)
    metrics = evaluate_predictions(
        csns=None,  # CSN not applicable for linear probe
        true_labels=y_val.tolist(),
        predicted_labels=y_pred.tolist(),
        predicted_probs=y_pred_probs.tolist()  # No probabilities for Logistic Regression in this context
    )

    if not os.path.exists(linear_probe_output_dir):
        os.makedirs(linear_probe_output_dir)
    classifier_path = os.path.join(linear_probe_output_dir, "linear_probe_classifier.joblib")
    joblib.dump(classifier, classifier_path)
    tokenizer.save_pretrained(linear_probe_output_dir) # Save tokenizer for consistent preprocessing at inference
    print(f"Linear probe classifier and tokenizer saved to {linear_probe_output_dir}")

    return linear_probe_output_dir


def finetune_qlora_model(config_params):
    """Performs qLoRA finetuning on the training data."""
    print("Starting qLoRA finetuning process...")

    BASE_MODEL_ID = config_params['BASE_MODEL_ID']
    TRAIN_FILE = config_params['TRAIN_FILE']
    PROMPT_TEMPLATE_FILE = config_params['PROMPT_TEMPLATE_FILE']
    FINETUNED_ADAPTER_DIR = config_params['FINETUNED_ADAPTER_DIR']
    LORA_R = config_params['LORA_R']
    LORA_ALPHA = config_params['LORA_ALPHA']
    LORA_DROPOUT = config_params['LORA_DROPOUT']
    TARGET_MODULES = config_params['TARGET_MODULES']
    TRAIN_EPOCHS = config_params['TRAIN_EPOCHS']
    TRAIN_BATCH_SIZE = config_params['TRAIN_BATCH_SIZE']
    GRADIENT_ACCUMULATION_STEPS = config_params['GRADIENT_ACCUMULATION_STEPS']
    LEARNING_RATE = config_params['LEARNING_RATE']
    FP16 = config_params['FP16']
    BF16 = config_params['BF16']
    MAX_SEQ_LENGTH = config_params['MAX_SEQ_LENGTH']
    PACKING = config_params['PACKING']
    TEXT_COLUMN = config_params.get('TEXT_COLUMN', 'text')
    LABEL_COLUMN = config_params.get('LABEL_COLUMN', 'label')


    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad token set to EOS token: {tokenizer.pad_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=True,
        # torch_dtype=torch.bfloat16 if BF16 else torch.float16 if FP16 else torch.float32,
        device_map="auto"
    )
    print("Base model loaded in 4-bit for qLoRA.")

    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    print("PEFT model created with LoRA configuration.")
    model.print_trainable_parameters()

    print("Loading and formatting training data for qLoRA...")
    # Ensure load_data provides the correct column names or adapt here
    train_df = load_data(TRAIN_FILE) 
    # keep only the first LIMIT_X rows for training
    if config_params['NUM_FINETUNE_ROWS'] is not None:
        train_df = train_df.head(config_params['NUM_FINETUNE_ROWS'])
    
    # The format_data_for_finetuning function is key here.
    # It needs to use PROMPT_TEMPLATE_FILE to create sequences where the model
    # learns to generate "Reasoning: ... Prediction: Yes/No".
    # It should take columns like `ehr_record_column=TEXT_COLUMN_NAME` and `label_column=LABEL_COLUMN_NAME`
    # from train_df.
    print(f"Using PROMPT_TEMPLATE_FILE for qLoRA: {PROMPT_TEMPLATE_FILE}")
    print(f"Formatting data for QLoRA using text column: '{TEXT_COLUMN}' and label column: '{LABEL_COLUMN}'")

    train_dataset = format_data_for_finetuning(
        df=train_df, # Pass the dataframe
        prompt_template_file=PROMPT_TEMPLATE_FILE, # Path to the template file
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        packing=PACKING,
    )
    print("Training data formatted for qLoRA.")
    if not train_dataset:
        print("Error: qLoRA training dataset is empty. Check data processing.")
        return None
    print(f"qLoRA training dataset size: {len(train_dataset)}")


    # Data collator for Causal LM (mlm=False)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    # Note: `group_texts` was part of older DataCollatorForLanguageModeling versions or specific collators.
    # If `PACKING` is True, `format_data_for_finetuning` must handle the packing into constant length sequences.
    # If not, `MAX_SEQ_LENGTH` will be applied by the tokenizer or collator for individual sequences.

    training_args_params = {
        "output_dir": FINETUNED_ADAPTER_DIR,
        "num_train_epochs": TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_dir": f"{FINETUNED_ADAPTER_DIR}/logs",
        "logging_steps": config_params.get('LOGGING_STEPS', 10),
        "save_steps": config_params.get('SAVE_STEPS', 100),
        "save_total_limit": config_params.get('SAVE_TOTAL_LIMIT', 2),
        "fp16": FP16,
        "bf16": BF16,
        "optim": config_params.get('OPTIMIZER', "paged_adamw_8bit"),
        "report_to": config_params.get('REPORT_TO', "none"),
        "remove_unused_columns": False, # Set to True if your dataset from format_data has extra columns not needed by model
        # `max_seq_length` in TrainingArguments is often for the trainer to enforce,
        # but pre-tokenization to MAX_SEQ_LENGTH is more common with custom datasets.
        # If packing is False, tokenizer in format_data_for_finetuning should handle truncation.
    }
    if not PACKING:
         training_args_params["max_seq_length"] = MAX_SEQ_LENGTH


    training_args = TrainingArguments(**training_args_params)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting qLoRA training...")
    trainer.train()
    print("qLoRA training finished.")

    print(f"Saving finetuned qLoRA adapter to {FINETUNED_ADAPTER_DIR}")
    trainer.model.save_pretrained(FINETUNED_ADAPTER_DIR)
    print("Finetuned qLoRA adapter saved.")

    tokenizer.save_pretrained(FINETUNED_ADAPTER_DIR)
    print("Tokenizer saved with qLoRA adapter.")
    
    del model
    del trainer
    torch.cuda.empty_cache()

    return FINETUNED_ADAPTER_DIR


def main_training_pipeline(training_mode: str):
    """
    Main pipeline to select and run training.
    training_mode: 'qlora' or 'linear_probe'
    """
    print(f"Selected training mode: {training_mode}")

    # Load configuration
    config_params = get_config()
    # convert module to dict 
    config_params = {k: v for k, v in config_params.__dict__.items() if not k.startswith('__')}

    # Create necessary output directories
    if training_mode == "qlora":
        os.makedirs(config_params['FINETUNED_ADAPTER_DIR'], exist_ok=True)
        os.makedirs(f"{config_params['FINETUNED_ADAPTER_DIR']}/logs", exist_ok=True)
    elif training_mode == "linear_probe":
        os.makedirs(config_params['LINEAR_PROBE_OUTPUT_DIR'], exist_ok=True)


    if training_mode == "qlora":
        print("\n--- Strategy for QLoRA Fine-tuning for Sepsis Prediction (Reasoning + Yes/No) ---")
        print("1. Training Data (`TRAIN_FILE`): Should contain EHR records and a binary label (e.g., 'Yes', 'No').")
        print(f"   Ensure it has columns named '{config_params['TEXT_COLUMN']}' (for EHR) and '{config_params['LABEL_COLUMN']}' (for labels).")
        print("2. Prompt Template (`PROMPT_TEMPLATE_FILE`): This file is crucial. It defines the full input/output structure.")
        print("   The model learns to complete this template. For your goal, it should be like:")
        print("   \"EHR Record: {ehr_record}\\n\\nQuestion: Will the patient get sepsis in the next 2 hours? Please provide your reasoning and then state your prediction as 'Prediction: Yes' or 'Prediction: No'.\\n\\nAnswer: Reasoning: [Model learns to generate this] Prediction: {label}\"")
        print("3. `format_data_for_finetuning` (in `data_processing.py`): Must use this template to combine EHR records and labels into the full text sequences for Causal LM training. The LLM learns to predict the tokens for the 'Reasoning: ... Prediction: {label}' part.")
        print("4. No Golden CoT: The model will learn to generate plausible-sounding text for 'Reasoning' that statistically aligns with the EHR data and the correct final 'Prediction: Yes/No'.")
        print("5. Inference: Provide the EHR and question (up to 'Answer:'). The model will generate the rest.\n")
        
        finetuned_path = finetune_qlora_model(config_params)
        if finetuned_path:
            print(f"QLoRA finetuning complete. Adapter saved at: {finetuned_path}")
            return finetuned_path
        else:
            print("QLoRA finetuning failed or was skipped.")
            raise RuntimeError("QLoRA finetuning failed. Check logs for details.")


if __name__ == "__main__":
    # Choose the training mode:
    # mode_to_run = "qlora"
    mode_to_run = "linear_probe"
    
    print(f"Running example for training mode: {mode_to_run}")
    print("Please ensure your 'TRAIN_FILE' and other paths/configs are correctly set up above.")
    print("This script assumes `data_processing.py` (with `load_data` and `format_data_for_finetuning`) is in the same directory or Python path.")

    config = load_config('/home/mkeoliya/projects/arpa-h/llm/tasks/mc_med_sepsis/configs/qwen4b_inf.py')
    set_config(config)

    main_training_pipeline(training_mode=mode_to_run, LIMIT_X=1000)