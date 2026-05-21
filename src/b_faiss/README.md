# Embeddings and FAISS

These scripts create the FAISS vector store. FAISS (Facebook AI Similarity Search) is a fast and efficient library for similarity search and clustering of vectors.

The texts of the articles will be embedded in to a multi-dimensional vector and stored in a vector database.

The dimensions of the vectors depends on the used embedding model.

By default, `gte-multilingual-base` model from Alibaba-NLP is used as the embedding model but you can choose any embedding model. If a different model is used, changes to subsequent scripts and codes are required. 

# How to

1. **First, check the JSON_FILE_NAME defined in [data_gather.py](../a_data/data_gather.py#30)**  
**Then, run the embedding creation script from project root in the LUMI login node with command (input the actual JSON_FILE_NAME)**  
`sbatch src/b_faiss/1_run_ingest.sh Alibaba-NLP/gte-multilingual-base climate-llm-finetuning/data/<JSON_FILE_NAME> climate-llm-finetuning/data/faiss_index/ 4096 500 64`  

**Arguments for the script explained:**
- embedding model
- path to json file (derived from data gather phase)
- path to where faiss_index will be saved
- chunk size (for example, with gte-multilingual-base one could set this to be 8192 as it is the max input tokens amount)
- chunk overlap (how many characters the chunks overlap with adjacent chunks)
- batch size

---
2. **Run the actual FAISS vector store creation script from project root in the LUMI login node with command**  
`sbatch src/b_faiss/2_run_merge.sh climate-llm-finetuning/data/faiss_index/`

**Arguments for the script explained:**
- path to faiss_index