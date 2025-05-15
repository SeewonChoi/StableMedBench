# Running LLMs on StableMedBench

Running LLMs on StableMedBench requires the following steps:

1. Generate prompt-worthy text of the EHR data, from the gen_dataset.py scripts in each `task/{task_name}` directory. E.g. to generate the prompts for MC-MED sepsis,  run `python gen_dataset.py` in the `tasks/mc_med_sepsis` directory. This will create files called `full.parquet` (for accuracy metrics) and `stability.parquet` (for stability metrics) in the `tasks/mc_med_sepsis` directory. These files contain the EHR data in a format that can be used to generate prompts for the LLMs.

2. For each `task`, run the `configs/qwen32b_inf.py` file to run the LLMs to get the predictions. This file contains the configuration for the LLMs, including the model name, sequence length, batch size, and column names. You can modify this file to change the configuration as needed. 

3. Run the `configs/qwen32b_stability.py` file to run the LLMs to get the stability metrics. 

4. We have provided the raw LLM outputs (including 10, anonymized sample prompts and generations) in the `stability_results` directory. 