# Evaluation

Evaluation of the finetuned LLM is a crucial part of the finetuning process. In this part, the test dataset will be used to get answers from both the finetuned model and the same model that has not been finetuned. We will then compare these models' answers on the "ground truth answers" (NOTE! ground truth answers are also created by an LLM in the [question and answer generation part](../c_qa/README.md)).

Evaluation will be conducted with basic benchmarks used in natural language processing, including BLEU, ROUGE and more modern metrics such as BERTScore and A3CU.

This process is divided into two steps. First, answers will be created by running SLURM scripts, individually for each model (finetuned and non-finetuned variant). Then, we will head to LUMI web UI to open a Marimo notebook to run evaluations on the answers.

BE NOTED! The amount of data collected in [data gather phase](../a_data/data_gather.py#55) (by default 2000) may not be sufficient enough to have an effect on how the finetuned LLM behaves. When using more data in finetuning, it will alter the model's output behavior and thus result in (hopefully better) scoring across metrics.
Also, when having less data overall, less data will be used to actually test the model 
- For example with default settings used throughout this tutorial, 1800 question-answer pairs (90%) will be used to actually finetune the model in training, 180 question-answer pairs (9%) will be used to test the model during the finetuning phase and 20 question-answer pairs (1%) will be used in this evaluation part. As can be seen, the amount of data (that has not been used in the finetuning phase) in the actual evaluation is not much.
- Increasing the data amount to even 10000 in the data gather phase can improve the results, but downloading the data will take more time.

# How to
**Run the following model inference scripts to get answers from the finetuned and non-finetuned models**
`sbatch src/e_eval/model_test_inference.sh meta-llama/Llama-3.1-8B-Instruct climate-llm-finetuning/data/ 512 16 n base_8b`

`sbatch src/e_eval/model_test_inference.sh climate-llm-finetuning/ft_data_8b/meta-llama_Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-finetuned/merged/ climate-llm-finetuning/data/ 512 16 n finetuned_8b`

**Script arguments explained:**
- `meta-llama/Llama-3.1-8B-Instruct` - model to use (either model name as stated in HuggingFace or the path to the saved model)
- `climate-llm-finetuning/data/` - main folder where dataset folder is located
- `512` - model output token amount
- `16` - batch size
- `n` - whether to resume answer creation (this is a "leftover argument", by default one doesn't need to resume answer creation so just use **n** here)
- `base_8b` - how to name the json file where the answers are stored (should infer which model was used (finetuned or non-finetuned) and model size)

**After the scripts have run, open the Lumi web UI and choose Marimo notebook**  
- Select the project and set the working directory to the project folder where this repo is cloned to
- Adjust the settings (Partition: `dev-g`, number of CPU cores: `7`, memory: `60 GiB`, number of GPUs: `1`, time: `2:00:00`, working directory: `/scratch/$PROJECT`, python: `lumi-multitorch`, module version: `...-20260513_121430 / default`)
    - The following settings are also required for marimo to work properly inside lumi-multitorch modules:
        - Enable the virtual environment and add the virtual environment path: `/scratch/$PROJECT/$USER/marimo_venv_lumi`
        - Enable system installed packages on venv creation
- Lastly, open and run the eval_models.py notebook