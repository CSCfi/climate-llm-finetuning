import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from openai import OpenAI

class QADataset(Dataset):
    def __init__(self, data, ctx_amount):
        self.data = data
        self.ctx_amount = ctx_amount

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qa = item.get("qa", {})
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        context = item.get("context", [])
        context_str = "\n\n".join(context[:self.ctx_amount]) if isinstance(context, list) else context
        return idx, question, answer, context_str


def create_a_vllm(question, context, llm):
    messages = [
        {
            "role": "system", 
            "content": """You are an expert at answering scientifically to a climate-related question. 

You are given context as an additional resource to guide your answer-creation process. 
The context is question-related text chunks of scientific articles with emphasis on climate related topics.

Do not refer to the given context directly, e.g. "In the study..." or "According to the provided context..."

The length of the answer should be a maximum of five sentences. """}, #\n Format the response as a JSON object with the 'answer' key.
        {
            "role": "user",
            "content": f"""Here is the context to base the answer on:\n\n
{context}\n\n
Here is the question: {question}"""
        }
    ]

    response = llm.chat.completions.create(
        model=model_id,
        temperature=0.6,
        messages=messages,
        max_tokens=max_tokens
    )

    content = response.choices[0].message.content

    return content


def batch_creation(questions: list, answers: list, contexts: list, llm, resume: bool):
    # if answer already exists, just reuse
    to_process = [(q, c) if resume == False else None
                  for q, c in zip(questions, contexts)]

    def worker(item):
        if item is None:  # means answer already exists
            return None
        q, c = item
        return create_a_vllm(q, c, llm)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, to_process))

    # Fill in existing answers where applicable
    if resume:
        for i, a in enumerate(answers):
            if len(a) > 0:
                results[i] = a

    return results


def batch_job(json_file_path, ctx_amount, batch_size, resume, *args, vllm=False):
    if vllm:
        llm = args[0]
   # else:
   #     model = args[0]
   #     tokenizer = args[1]

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    dataset = QADataset(data, ctx_amount=ctx_amount)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    processed_data = data.copy()

    for ix, batch in enumerate(tqdm(loader)):
        idx_batch, question_batch, answer_batch, context_batch = batch
        idx_list = idx_batch.tolist()
        questions = list(question_batch)
        answers = list(answer_batch)
        contexts = list(context_batch)

        try:
            # Batch inference (must return a list of strings)
            a_parts = batch_creation(questions, answers, contexts, llm, resume)
            for i, a_part in enumerate(a_parts):
                idx = idx_list[i]
                a_part = a_part.strip()
                # a_part = re.sub("\n", "", a_part)

                # if not a_part.startswith('{'):
                #     a_part = '{' + a_part
                # if not a_part.endswith('}'):
                #     a_part = a_part + '}'

                try:
                    # a_json_dumps = json.dumps(a_part)

                    # a_json = json.loads(a_json_dumps)
                    qa = processed_data[idx]['qa']
                    processed_data[idx]['qa'] = {**qa, "answer": a_part}
                except Exception as inner_e:
                    print(f"JSON parse error at idx {idx}: {inner_e}\n{a_part}")
                    continue
            
            # if ix % 500 == 0: # Save the files every 500 batch just in case
            #     with open(json_file_path, "w", encoding="utf-8") as f:
            #         json.dump(processed_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Batch error: {e}")
            continue
    
    for i, item in enumerate(processed_data):
        if i % 500 == 0 and "qa" in item:
            print(item["filename"], item["qa"])

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Finalize the json by creating the answer to the qa.')
    parser.add_argument('--backend', choices=['vllm'], default='vllm') #'transformers', 
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--filepath', default='/scratch/project_462000824/data/extracted_texts_from_xml_pdf_qa_4096.json')
    parser.add_argument('--api-url')
    parser.add_argument('--ctx_amount', default=4)
    parser.add_argument('--max_tokens', default=2048)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--resume', choices=['n', 'y'], default="n")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model
    backend = args.backend
    api_url = args.api_url
    json_file_path = args.filepath
    ctx_amount = int(args.ctx_amount)
    max_tokens = int(args.max_tokens)
    batch_size = int(args.batch_size)
    resume = args.resume == "y"

    print("Starting answer creation")

    print("GPU:", torch.cuda.is_available())

    if backend == 'vllm':
        llm = OpenAI(
            api_key="EMPTY",
            base_url=api_url,)
        batch_job(json_file_path, ctx_amount, batch_size, resume, llm, vllm=True)

    with open(json_file_path, 'r') as f:
        data = json.load(f)
        print(data[:5])