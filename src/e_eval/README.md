# Evaluation

Evaluation of the finetuned LLM is a crucial part of the finetuning process. In this part, the test dataset will be used to get answers from both the finetuned model and the same model that has not been finetuned. We will then compare these models' answers on the "ground truth answers" (NOTE! ground truth answers are also created by an LLM in the [question and answer generation part](../c_qa/README.md)).

Evaluation will be conducted with basic benchmarks used in natural language processing, including BLEU, ROUGE and more modern metrics such as BERTScore and A3CU.

This process is divided into two steps. First, answers will be created by running SLURM scripts, individually for each model (finetuned and non-finetuned variant). Then, we will head to LUMI web UI to open a Marimo notebook to run evaluations on the answers.

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

**Then, open the Lumi web UI and choose Marimo notebook**  
- Select the project and set the working directory to the project folder where this repo is cloned to  
- Adjust the settings (Partition: `dev-g`, number of CPU cores: `7`, memory: `60 GiB`, time: `2:00:00`)  
- Lastly, run the notebook