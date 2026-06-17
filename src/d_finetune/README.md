# Finetuning an LLM with the data

In this part, we will finetune a small LLM (8B sized model) with the dataset we created in [the previous part](../c_qa/). Accelerate framework will be used alongside PEFT and 4-bit quantization. We will use 2 nodes and all of their GPUs, so 16 GPUs in total. In this version, we will use the pytorch module provided by CSC.

In order to finetune or otherwise use Llama model(s) found in HuggingFace, one needs to:
1. Login (or register) to [HuggingFace](https://huggingface.co/)
2. [Create an access token](https://huggingface.co/settings/tokens)
3. Create a token file with your access token in LUMI terminal via ssh or opening login node from LUMI Web UI: `echo hf_access_token > ~/.cache/huggingface/token`
4. Apply for access to the model in the [model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). You will receive an email when the access has been granted.

# How to
1. **Use the same json file name that was used in [previous step](../c_qa/README.md#37)**  
**Then, run the finetuning script from project root in the LUMI login node with command (example)**  
`sbatch src/d_finetune/6_run_finetune_lumi_gpu16_accelerate.sh src/d_finetune/accelerate_config_fsdp.yaml meta-llama/Llama-3.1-8B-Instruct climate-llm-finetuning/ft_data_8b climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa_4096.json 16 14`

**Script arguments explained:**
- `src/d_finetune/accelerate_config_fsdp.yaml` - accelerate config
- `meta-llama/Llama-3.1-8B-Instruct` - LLM to be finetuned
- `climate-llm-finetuning/ft_data_8b` - path where to save the finetuned model
- `climate-llm-finetuning/data/extracted_texts_from_xml_pdf_qa_4096.json` - path to json file
- `16` - batch size
- `14` - number of workers for processing the dataset etc.