#!/bin/bash

# run experiments for MIMIC-ICU
CUDA_VISIBLE_DEVICES=3 python main.py --config /home/mkeoliya/projects/arpa-h/llm/tasks/mimic_icu/configs/qwen32b_inf.py
CUDA_VISIBLE_DEVICES=3 python main.py --config /home/mkeoliya/projects/arpa-h/llm/tasks/mimic_icu/configs/qwen32b_stability.py --stability --num_inf_rows 3000
