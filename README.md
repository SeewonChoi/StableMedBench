# StableMedBench

### Dataset Access

The datasets used in this benchmark are protected, hence, need to be downloaded from respective sources.

* [EHRShot](https://redivis.com/datasets/53gc-8rhx41kgt)

* [MC-MED](https://physionet.org/content/mc-med/1.0.0/)

* MIMIC: [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/), and [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)

### Benchmark

For each TASK, run relevant files under `process_data` and `benchmarks`, the dataset would be created under `data/{TASK}`

### Models

* For classical models XGBoost and Random Forest, run  `classical/trainer --task {TASK}`.

  Additionally for stability, run `classical/stability --task {TASK}`


* For transformers GPT2 and Mamba-130M, run

  Optionally, to pre-train the model, 


* 