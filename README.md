# StableMedBench

### Dataset Access

The datasets used in this benchmark are protected, hence, need to be downloaded from respective sources.

* [EHRShot](https://redivis.com/datasets/53gc-8rhx41kgt)

* [MC-MED](https://physionet.org/content/mc-med/1.0.0/)

* MIMIC: [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/), and [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)

### Benchmark

For each TASK, run relevant files under `process_data` and `benchmarks`, the dataset would be created under `data/{TASK}`. 
Set `DATA_DIR` to `data/{TASK}` for each task. 

### Models

* For classical models XGBoost and Random Forest, run  `classical/trainer --task {TASK}`.

  Additionally for stability, run `classical/stability --task {TASK}`


* For transformers GPT2, GPT2-AR and Mamba, first create a tokenizer with files under `tokenizer`.

  Then, run `python trainer_binary.py --task {TASK}`.

  Optionally, to pre-train the model, use `python pretrain/trainer.py`, 
  and modify the loader in `pretrain/trainer.py` to load the dataset you want to pre-train on.


* For LLMs, refer to the README.md in the `llm` directory. Note that we ran experiments on an Nvidia A100 80GB GPU, and the code is not optimized for other GPUs. Physionet policies for MIMIC dataset prevent using API providers such as OpenAI or Claude naively, refer [here](https://physionet.org/news/post/gpt-responsible-use) for details. 