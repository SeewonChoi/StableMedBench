import os

# FINETUNING?
SKIP_FINETUNING = False

# Limit rows
NUM_INF_ROWS = None  # Set to None to use all rows, or specify a number for testing
NUM_FINETUNE_ROWS = None  # Set to None to use all rows, or specify a number for testing

# Mode
STABILITY = False  # Set to True for stability mode, False for normal mode

# --- File Paths ---
TASK_DIR = "/home/mkeoliya/projects/arpa-h/llm/tasks/mc_med_decomp/"
TRAIN_FILE = os.path.join(TASK_DIR, "data", "val.parquet")
TEST_FILE = os.path.join(TASK_DIR, "data", "test.parquet")
PROMPT_TEMPLATE_FILE = os.path.join(TASK_DIR, "prompts", "prompt_template.txt")
RESULTS_DIR = os.path.join(TASK_DIR, "runs", "qwen4b_qlora")
FINETUNED_ADAPTER_DIR = os.path.join(RESULTS_DIR, "finetuned_adapter")
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results directory if it doesn't exist
LOG_FILE = os.path.join(RESULTS_DIR, "file.log")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")


# --- Model Configuration ---
# Choose a base model. Examples:
# "meta-llama/Llama-2-7b-hf" (base model) or "meta-llama/Llama-2-7b-chat-hf" (chat model)
# "mistralai/Mistral-7B-v0.1" (base model) or "mistralai/Mistral-7B-Instruct-v0.2" (instruct model)
# Use instruct/chat models if your prompt format is conversational.
# Base models require more careful prompt engineering for instruction following.
BASE_MODEL_ID = "Qwen/Qwen3-4B"
# BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2" # Alternative good choice

# --- Finetuning (qLoRA) Configuration ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Target modules for LoRA. Depends on the model architecture.
# Common choices for Llama/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# A common practice for efficiency/effectiveness is to target attention layers (q, k, v, o).
# Finding the exact layer names might require inspecting the model's config or structure.
TARGET_MODULES = ["q_prof", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# For other models, you might need different names, e.g., 'query', 'key', 'value', 'dense' etc.

# Training arguments
TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 4 # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 3 # Adjust based on GPU memory/desired effective batch size
LEARNING_RATE = 3e-4
FP16 = False # Use FP16 if available and it fits memory. BF16 often better if supported.
BF16 = True # Use BF16 if your GPU supports it (e.g., A100, H100, some newer consumer cards)
MAX_SEQ_LENGTH = 6000 # Maximum tokens in combined prompt and completion
PACKING = True # Use packing for efficiency if using a recent transformers version and Dataset

# --- Inference (vLLM) Configuration ---
# vLLM parameters
VLLM_TENSOR_PARALLEL_SIZE = 1 # Number of GPUs to use for inference (1 for single GPU)
VLLM_MAX_MODEL_LEN = MAX_SEQ_LENGTH # Same as finetuning max length
VLLM_DTYPE = "auto" # or "float16", "bfloat16", "half"

# vLLM Sampling Parameters
ENABLE_THINKING = False # Enable thinking for the model
VLLM_TEMPERATURE = 0.7
VLLM_TOP_P = 0.80
VLLM_TOP_K = 20
VLLM_MIN_P = 0
VLLM_PRESENCE_PENALTY = 1.5
VLLM_MAX_TOKENS = 2048 # Max tokens for the LLM's response (should be small for Yes/No)
# Stop sequences: Ensure the model stops generating tokens after predicting "Yes" or "No".
# Add "\n" just in case, although it should ideally stop immediately after the word.
VLLM_STOP_SEQUENCES = [] # Stop generation after these tokens

# --- Data Processing Configuration ---
# Column names in the parquet files
CSN_COLUMN = "CSN"
TEXT_COLUMN = "Text"
LABEL_COLUMN = "Label"
REMAINING_TEXT_COLUMN = "Remaining_Text" # Column for remaining text after the first 6000 tokens

# --- Evaluation Configuration ---
POSITIVE_LABEL = True # The value in 'Label' that indicates sepsis

# --- Logging Configuration ---
LOGGING_LEVEL = "INFO" # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"