import pandas as pd
import os
from datasets import Dataset
from transformers import AutoTokenizer
import torch
import logging

from config_loader import get_config

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a parquet file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def load_prompt_template(file_path: str) -> str:
    """Loads the prompt template from a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")
    with open(file_path, 'r') as f:
        template = f.read().strip()
    # Split into system and user parts
    parts = template.split("SYSTEM:")
    system_prompt = parts[1].split("USER:")[0].strip()
    user_template = "USER:" + parts[1].split("USER:")[1].strip()
    return system_prompt, user_template

def format_data_for_finetuning(df: pd.DataFrame, prompt_template_file: str, tokenizer: AutoTokenizer, max_length: int, packing: bool) -> Dataset:
    """Formats data into instruction-following prompts for finetuning."""
    config = get_config()
    system_prompt, user_template = load_prompt_template(prompt_template_file)

    def create_prompt(row):
        ehr_text = row[config.TEXT_COLUMN]
        remaining_ehr = row[config.REMAINING_TEXT_COLUMN]
        label = row[config.LABEL_COLUMN]
        # Format label to match expected output format
        target_output = f"Prediction: {label}, Risk: 0.99" if label == "Yes" else f"Prediction: {label}, Risk: 0.01"

        user_prompt_text = user_template.format(ehr_text=ehr_text)

        # Format using the model's chat template if available
        # This is crucial for finetuning chat/instruct models effectively
        try:
            chat_format_input = tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_text}
            ], tokenize=False, add_generation_prompt=False) # No generation prompt needed for training input

            chat_format_output = target_output + tokenizer.eos_token # Add EOS token to the target output

            # Combine input and output for causal language modeling finetuning
            # The model learns to predict the output given the input prompt.
            full_text = chat_format_input + chat_format_output

        except Exception as e:
            print(f"Warning: Could not apply chat template for {tokenizer.clean_up_tokenization(tokenizer.eos_token)}. Falling back to simple concatenation. Error: {e}")
            # Fallback for models without a built-in chat template or errors
            full_text = f"{system_prompt}\n{user_prompt_text} {target_output}{tokenizer.eos_token}"
            full_text = f"{system_prompt}\n{user_prompt_text} Prediction: {target_output}{tokenizer.eos_token}"

        return {"text": full_text}

    # Apply the formatting
    df_prompts = df.apply(create_prompt, axis=1).tolist()
    dataset = Dataset.from_pandas(pd.DataFrame(df_prompts))

    # Tokenize the dataset
    # We don't need to tokenize here if using TrainingArguments and a DataCollator
    # The DataCollator handles padding and truncation during training batches.
    # However, if packing is used, tokenize beforehand.

    if packing:
         def tokenize_function(examples):
             return tokenizer(examples["text"], truncation=True, max_length=max_length)

         tokenized_dataset = dataset.map(
             tokenize_function,
             batched=True,
             remove_columns=["text"],
             desc="Tokenizing dataset",
         )
         # Packing will be handled by the data collator (e.g., DataCollatorForLanguageModeling)
         # with `group_texts=True`.
         return tokenized_dataset
    else:
        # If not packing, the Trainer's DataCollator will handle tokenization and padding.
        # We just need the 'text' column.
        return dataset.map(lambda examples: {}, remove_columns=dataset.column_names, keep_original_columns=["text"])


def format_data_for_inference(df: pd.DataFrame, prompt_template_file: str, tokenizer: AutoTokenizer) -> list[str]:
    """Formats data into prompts for inference."""
    system_prompt, user_template = load_prompt_template(prompt_template_file)

    config = get_config()
    TEXT_COLUMN, CSN_COLUMN = config.TEXT_COLUMN, config.CSN_COLUMN

    prompts = []
    num_truncated = 0
    for _, row in df.iterrows():
        ehr_text = row[TEXT_COLUMN]
        user_prompt_text = user_template.format(ehr_text=ehr_text)

        # Format using the model's chat template if available
        try:
            chat_format_prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_text}
            ], tokenize=False, add_generation_prompt=True, enable_thinking=config.ENABLE_THINKING) # Add generation prompt for inference


        except Exception as e:
             print(f"Warning: Could not apply chat template for {tokenizer.clean_up_tokenization(tokenizer.eos_token)}. Falling back to simple concatenation. Error: {e}")
             # Fallback
             chat_format_prompt = f"###Question: {system_prompt}\n{user_prompt_text} ###Answer:"
        
        # if number of tokens is too long, truncate it 
        if len(tokenizer(chat_format_prompt)["input_ids"]) > (config.VLLM_MAX_MODEL_LEN - config.VLLM_MAX_TOKENS):
            num_truncated += 1
            # Truncate the prompt to fit within the model's max length
            chat_format_prompt = tokenizer(chat_format_prompt, truncation=True, max_length=config.VLLM_MAX_MODEL_LEN - config.VLLM_MAX_TOKENS)["input_ids"]
            # Convert back to string for vLLM
            chat_format_prompt = tokenizer.decode(chat_format_prompt, skip_special_tokens=True)
            logging.warning(f"Prompt truncated to fit model's max length: {len(chat_format_prompt)} tokens for CSN {row[CSN_COLUMN]}")

        prompts.append(chat_format_prompt)
    
    # count the median length of the prompts
    prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    median_length = int(sum(prompt_lengths) / len(prompt_lengths))
    logging.info(f"Median prompt length: {median_length} tokens.")
    logging.info(f"Max prompt length: {max(prompt_lengths)} tokens.")

    if num_truncated > 0:
        logging.warning(f"Truncated {num_truncated} prompts due to length issues.")
    return prompts