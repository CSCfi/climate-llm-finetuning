import argparse
import json
import torch
import time
import faiss
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

def get_query_embeddings(queries, batch_size=32):
    all_embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:, 0]  # CLS token
            embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.half().cpu())
    return torch.cat(all_embeddings, dim=0).numpy()


def search_faiss_batch(query_embeddings, k=3):
    distances, indices = index.search(query_embeddings, k)
    return distances, indices


def insert_contexts_batch(json_file_path, json_file_suffix, batch_size=32):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Collect all questions
    questions = []
    for item in data:
        qa = item.get("qa", {})
        question = qa.get("question", "") if qa else ""
        questions.append(question)

    # Compute all embeddings in batches
    start = time.time()
    query_embeddings = get_query_embeddings(questions, batch_size=batch_size)

    # FAISS batch search
    distances, indices = search_faiss_batch(query_embeddings, k=6)

    # Insert retrieved contexts back
    for i, item in enumerate(data):
        if questions[i]:
            item["context"] = [all_metadata[idx]["text"] for idx in indices[i]]

    end = time.time()
    print(f"Getting contexts took {end-start:.2f} seconds")

    # Save results
    with open(f"{json_file_path.split('.')[0]}_{json_file_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Get contexts from FAISS vectorstore and insert to json.')
    parser.add_argument('--model', default='Alibaba-NLP/gte-multilingual-base')
    parser.add_argument('--jsonpath', default='/scratch/project_462000824/data/extracted_texts_from_xml_pdf_qa.json')
    parser.add_argument('--faisspath', default='/scratch/project_462000824/hmerilai/philologue/scripts/RAG-60K/faiss_index_gte-multilingual-base_4096/')
    parser.add_argument('--suffix', default='4096')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model
    json_file_path = args.jsonpath
    faiss_index_path = args.faisspath
    json_file_suffix = args.suffix

    # Load the model and tokenizer for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda").eval()
    # Optional: Convert model to FP16 for faster inference
    model.half()

    # Load FAISS index and metadata
    faiss_index_file = f"{faiss_index_path}faiss_index.bin"
    metadata_file = f"{faiss_index_path}metadata.json"

    index = faiss.read_index(faiss_index_file)

    # Move FAISS index to GPU if available
    if torch.cuda.is_available():
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)

        index = faiss.index_cpu_to_all_gpus(index)

    # Load metadata
    with open(metadata_file, "r") as f:
        all_metadata = json.load(f)

    insert_contexts_batch(json_file_path, json_file_suffix)

    with open(f"{json_file_path.split('.')[0]}_{json_file_suffix}.json", 'r') as f:
        data = json.load(f)
        print(data[0])