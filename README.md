# climate-llm-finetuning
Workflow for finetuning LLMs with climate related data

This repository contains codes for:  
[1. a_data](/src/a_data/): Downloading PDF and XML files of articles from copernicus.org and preprocessing the text data  
[2. b_faiss](/src/b_faiss/): Creating embeddings of the text data and storing them in a FAISS vector store  
[3. c_qa](/src/c_qa/): Creating the actual dataset (question-answer pairs) utilizing the FAISS vector store for RAG and an LLM to create the QA pairs  
[4. d_finetune](/src/d_finetune/): Fine-tuning an LLM in LUMI  
[5. e_eval](/src/e_eval/): Testing the fine-tuned LLM against the non-finetuned one

## Environments
Marimo is utilized instead of Jupyter Notebooks for certain tasks of the workflow. Marimo has several benefits over Jupyter:
- Acts like a notebook, but is a python script (can thus be run in a SLURM script in an HPC cluster)
- Git-friendly (no messy jupyter outputs stored in version control)
- The code can be run as an interactive web interface aswell (not part of this exercise)

Currently, Marimo is available in the testing version of LUMI, found [here](https://ood-testing.lumi.csc.fi/public/) under *My Interactive Sessions*.

## How to
Each subfolder has their own instructions on how to run them.