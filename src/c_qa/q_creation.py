import argparse
import json
import re
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from openai import OpenAI

class ArticleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        abstract = item.get("abstract_text", "")
        # text_sample = item.get("text", "")
        # text_sample = text_sample[:50]
        return idx, abstract #, text_sample


def create_q_vllm(text, llm):
    messages=[
        {
            "role": "system",
            "content": """You are an expert at creating a question from given text for a question-answer pair. 
The texts are abstracts of scientific articles with emphasis on climate related topics.

Come up with a question based on the text. The question should be understandable without having read the article. 
Avoid referring to the study directly, as in 'In the study...'

Format the response as a JSON object with the 'question' key.
"""
        },
        {
            "role": "user",
            "content": f"Here is the abstract text of the article: {text}"
        }
    ]

    response = llm.chat.completions.create(
        model=model_id,
        temperature=0.6,
        response_format={"type": "json_object"},
        messages=messages,
    )

    content = response.choices[0].message.content

    question = content.strip('`')
    return question


def batch_creation(abstracts: list, llm): #text_samples:list,
    with ThreadPoolExecutor() as executor:
        contents = list(executor.map(lambda item: create_q_vllm(item, llm), abstracts))
    return contents

def batch_creation_text_sample(abstracts: list, text_samples: list, llm):
    def process(item):
        abstract, sample = item
        if not sample or not sample.strip():  # skip if empty or just whitespace
            return None
        return create_q_vllm(abstract, llm)

    with ThreadPoolExecutor() as executor:
        contents = list(executor.map(process, zip(abstracts, text_samples)))
    return contents


def batch_job(json_file_path, suffix, *args, vllm=False):
    if vllm:
        llm = args[0]
    else:
        model = args[0]
        tokenizer = args[1]

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    dataset = ArticleDataset(data)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    processed_data = data.copy()

    for batch in tqdm(loader):
        idx_batch, article_batch = batch #, text_batch
        idx_list = idx_batch.tolist()
        articles = list(article_batch)
        # text_samples = list(text_batch)

        try:
            # Batch inference (must return a list of strings)
            q_parts = batch_creation(articles, llm)
            for i, q_part in enumerate(q_parts):
                idx = idx_list[i]
                q_part = q_part.strip()
                q_part = re.sub("\n", "", q_part)

                if not q_part.startswith('{'):
                    q_part = '{' + q_part
                if not q_part.endswith('}'):
                    q_part = q_part + '}'

                try:
                    q_json = json.loads(q_part)
                    processed_data[idx]['qa'] = q_json
                except Exception as inner_e:
                    print(f"JSON parse error at idx {idx}: {inner_e}, {q_part}")
                    continue

        except Exception as e:
            print(f"Batch error: {e}")
            continue
    
    for i, item in enumerate(processed_data):
        if i % 500 == 0 and "qa" in item:
            print(item["filename"], item["qa"])

    # Create a new json file with the question-answer pairs in it
    with open(f'{json_file_path.split(".")[0]}_{suffix}.json', "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create questions from abstracts.')
    parser.add_argument('--backend', choices=['vllm'], default='vllm') #'transformers', 
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--filepath', default='/scratch/project_462000824/data/extracted_texts_from_xml_pdf.json')
    parser.add_argument('--suffix', default='qa')
    parser.add_argument('--api-url')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model
    backend = args.backend
    api_url = args.api_url
    json_file_path = args.filepath
    json_file_suffix = args.suffix

    if backend == 'vllm':
        llm = OpenAI(
            api_key="EMPTY",
            base_url=api_url,)
        batch_job(json_file_path, json_file_suffix, llm, vllm=True)

    with open(f'{json_file_path.split(".")[0]}_{json_file_suffix}.json', 'r') as f:
        data = json.load(f)
        print(data[:3])
