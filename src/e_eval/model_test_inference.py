import os
import argparse
import json

from datasets import load_from_disk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

def create_answer(question, llm, model_id, max_tokens):
    """Ask model a question and return the answer."""
    messages = [{"role": "user", "content": question}]
    return llm.chat.completions.create(
        model=model_id,
        temperature=0.0,
        messages=messages,
        max_tokens=max_tokens
    ).choices[0].message.content


def batch_creation(
        questions,
        answers,
        llm,
        model_id,
        resume: bool,
        max_tokens: int
    ):
    to_process = [q if not resume else None for q in questions]

    def worker(item):
        if item is None:  # means answer already exists
            return None
        return create_answer(item, llm, model_id, max_tokens)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, to_process))

    # If resuming, keep old answers
    if resume:
        for i, a in enumerate(answers):
            if len(a) > 0:
                results[i] = a
    return results


def run_model_on_dataset(
        test_dataset,
        model_id,
        model_name,
        llm,
        resume=False,
        batch_size=8,
        max_tokens=512,
    ):
    """Run base/finetuned model on dataset and add answers as a column."""

    questions = [item["qa"]["question"] for item in test_dataset]
    references = [item["qa"]["answer"] for item in test_dataset]

    model_answers = []
    for i in tqdm(range(0, len(questions), batch_size), desc=f"Running {model_name}"):
        q_batch = questions[i : i + batch_size]
        a_batch = batch_creation(q_batch, [], llm, model_id, resume, max_tokens)
        model_answers.extend(a_batch)

    # Build structured output
    results = []
    for q, r, a in zip(questions, references, model_answers):
        results.append({
            "question": q,
            "reference_answer": r,
            "model_answer": a,
            "model": model_name,
        })
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create answers on questions with different models for evaluation purposes')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--filepath')
    parser.add_argument('--model_suffix')
    parser.add_argument('--max_tokens', default=512)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--resume', choices=['n', 'y'], default="n")
    parser.add_argument('--api-url')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model
    model_suffix = args.model_suffix
    dataset_path = args.filepath
    max_tokens = int(args.max_tokens)
    batch_size = int(args.batch_size)
    api_url = args.api_url
    resume = args.resume == "y"

    # Load dataset once (questions + reference answers)
    test_dataset = load_from_disk(os.path.join(dataset_path, "test_dataset"))

    llm = OpenAI(
        api_key="EMPTY",
        base_url=api_url,
    )

    results = run_model_on_dataset(
        test_dataset,
        model_id,
        model_suffix,
        llm,
        resume=resume,
        batch_size=batch_size,
        max_tokens=max_tokens
    )

    # Safe file name
    output_file = os.path.join(dataset_path, f"answers_{model_suffix}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(results)} results to {output_file}")