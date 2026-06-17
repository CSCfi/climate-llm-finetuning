# Creating the dataset

Scripts and codes in this folder are responsible for
1. [creating a question](./3_run_create_q.sh) with an LLM based on article's abstract for each article
2. [retrieving relevant text chunks](./4_run_get_context.sh) from the vector store by querying it with each question
3. [creating an answer to the question](./5_run_create_a.sh) with an LLM based on retrieved contexts for each generated question

This is possibly the most time consuming part of this tutorial, where the answer creation takes the most time. Total time it should take for all the different parts of the workflow to be run is max 10 hours (depenging on the datasize that was configured in [data creation part](../a_data/data_gather.py#53)).

In the question and answer generation scripts, dedicated vLLM servers are set up to host the chosen LLM. A smaller 8B LLM is used by default to make the scripts run quicker.

# How to

1. **Use the same json file name that was used in [previous step](../b_faiss/README.md#13)**  
**Then, run the question creation script from project root in the LUMI login node with command (example)**  
`sbatch src/c_qa/3_run_create_q.sh Qwen/Qwen3-VL-8B-Instruct climate-llm-finetuning/data/extracted_texts_from_xml_pdf.json qa`

**Script arguments explained:**
- `Qwen/Qwen3-VL-8B-Instruct` - LLM to use for question generation
- `climate-llm-finetuning/data/extracted_texts_from_xml_pdf.json` - path to json file
- `qa` - suffix to be included in the new json file that includes the questions

---
2. **Use the same json file name that was used in [previous step](./README.md#14), including the suffix**  
**Then, run the context retrieval script from project root in the LUMI login node with command (example)**  
`sbatch src/c_qa/4_run_get_context.sh Alibaba-NLP/gte-multilingual-base climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa.json climate-llm-finetuning/data/faiss_index/ 4096`

**Script arguments explained:**
- `Alibaba-NLP/gte-multilingual-base` - embedding model that was used in [vector store creation](../b_faiss/README.md#15)
- `climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa.json` - path to json file with the suffix
- `climate-llm-finetuning/data/faiss_index/` - path to where FAISS index was saved
- `4096` - suffix to be included in the new json file that includes the questions and retrieved contexts

---
3. **Use the same json file name that was used in [previous step](./README.md#24), including both suffixes**  
**Then, run the answer generation script from project root in the LUMI login node with command (example)**  
`sbatch src/c_qa/5_run_create_a.sh Qwen/Qwen3-VL-8B-Instruct climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa_4096.json 6 1024 64`

**Script arguments explained:**
- `Qwen/Qwen3-VL-8B-Instruct` - LLM to use for answer generation
- `climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa_4096.json` - path to json file with the suffixes
- `6` - number indicating how many document context sections to retrieve from vector store
- `1024` - maximum amount of tokens when generating output (answer)
- `64` - batch size (how many prompts should be grouped up and sent to vLLM server at once)
