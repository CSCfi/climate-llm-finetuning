import argparse
import os
import sys
import time
import torch

from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from functools import partial

def chunk_text(text, chunk_size, overlap_size, eos_token=""):
    """
    Split a given text into chunks with an optional overlap.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_size : int, optional
        The maximum size of each chunk. Defaults to `max_chunk_size`.
    overlap_size : int, optional
        The number of tokens that should overlap between chunks. Defaults to `overlap_tokens`.
    eos_token : str, optional
        Token to append at the end of a chunk.

    Returns
    -------
    List[str]
        A list containing the chunked text.

    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if overlap_size >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")
    if overlap_size < 0:
        raise ValueError("Overlap cannot be negative.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end] + eos_token)
        start += chunk_size - overlap_size
    return chunks


def preprocess(examples, tokenizer, max_tokens=4096, chunk_size=16384, overlap_size=200):
    """Preprocesses a batch of examples by splitting texts into chunks and tokenizing them."""
    all_chunks = []
    for text in examples["prompt"]:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap_size=overlap_size, eos_token=tokenizer.eos_token)
        all_chunks.extend(chunks)

    tokenized_output = tokenizer(
        all_chunks,
        padding=True,
        truncation=True,
        max_length=max_tokens,  # Make sure this matches the model's max token length
        add_special_tokens=True,
        return_length=False,
    )

    return {"input_ids": tokenized_output["input_ids"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="The pre-trained model from Hugging Face to use as basis: https://huggingface.co/models",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The root directory under which model checkpoints and data are stored.",
    )
    parser.add_argument(
        "--json-datapath",
        type=str,
        help="The directory where json data is stored.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="The number of CPU worker processes to use.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="If set, continue from a previously interrupted run. Otherwise, overwrite existing checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="The number of training steps.",
    )
    parser.add_argument("--peft", action="store_true", help="Use PEFT: https://huggingface.co/docs/peft/index")
    parser.add_argument(
        "--4bit",
        dest="bnb_4bit",
        action="store_true",
        help="Use 4bit quantization with bitsandbytes: https://huggingface.co/docs/bitsandbytes/main/en/index",
    )
    args, _ = parser.parse_known_args()
    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Then we determine the device on which to train the model.
    if rank == 0:
        print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        print(f"Using GPU {local_rank}, device name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"No GPU found, using CPU instead. (Rank: {local_rank})")
        device = torch.device("cpu")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} has to be a multiple of the number of GPUs={world_size}!")
        sys.exit(1)

    # We also ensure that output paths exist
    model_name = args.model.replace("/", "_")

    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, model_name)

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map
    if rank == 0:
        print("Loading model and tokenizer")

    quantization_config = None
    if args.bnb_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        quantization_config = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        # cache_dir=output_dir,
    )

    if args.peft:
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print("Using PEFT")
        model.print_trainable_parameters()

    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop - start:.2f} seconds")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        # save_strategy="no",  # good for testing
        save_strategy="steps",  # use these if you actually want to save the model
        save_steps=600,  # original is 1200
        save_total_limit=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        eval_strategy="steps",
        eval_steps=600,  # compute validation loss every 600 steps
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use validation loss to determine the best model
        greater_is_better=False,  # Smaller validation loss is better
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=args.batch_size // world_size,
        per_device_eval_batch_size=args.batch_size,
        # max_steps=args.max_steps,
        num_train_epochs=1,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],  # Enable TensorBoard logging
        logging_dir=os.path.join(output_dir, "logs"),  # Directory for TensorBoard logs
        logging_steps=500,  # Log every 500 steps
    )

    def apply_chat_template(example):
        messages = [
            {"role": "user", "content": example['qa']['question']},
            {"role": "assistant", "content": example['qa']['answer']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}

    # Load your JSON data
    # Assuming the JSON file is in the format [{"filename": ..., "text": ...}, ...]
    data_path = args.json_datapath # Path to your dataset
    # Split dataset into train and validation sets
    raw_dataset = load_dataset("json", data_files=data_path)
    raw_dataset = raw_dataset.filter(lambda example: "qa" in example and example["qa"] is not None)
    chat_template_dataset = raw_dataset.map(apply_chat_template)
    train_val_dataset = chat_template_dataset['train'].train_test_split(test_size=0.1, seed=42)
    val_test_dataset = train_val_dataset['test'].train_test_split(test_size=0.1, seed=42)
    final_dataset = DatasetDict({
        'train': train_val_dataset['train'],
        'val': val_test_dataset['train'],
        'test': val_test_dataset['test']
    })

    if rank == 0:
        # final_dataset["train"].save_to_disk(os.path.join(args.json_datapath, "train_dataset"))
        # final_dataset["test"].save_to_disk(os.path.join(args.json_datapath, "test_dataset"))
        print(f"Example from dataset:\n{final_dataset['train'][0]}")
    max_tokens = 4096
    overlap_tokens = 100

    preprocess_function = partial(
        preprocess, tokenizer=tokenizer, max_tokens=max_tokens, chunk_size=16384, overlap_size=overlap_tokens
    )

    # Update the mapping logic to preprocess the dataset
    tokenized_train_dataset = final_dataset["train"].map(
        preprocess_function,
        batched=True,
        batch_size=training_args.train_batch_size,
        remove_columns=["filename", "text", "abstract_text", "text_source", "context", "qa", "prompt"],
        num_proc=args.num_workers,
        cache_file_name=args.output_path + "/data/tr_data.arrow",
        load_from_cache_file=True,
    )

    tokenized_val_dataset = final_dataset["val"].map(
        preprocess_function,
        batched=True,
        remove_columns=["filename", "text", "abstract_text", "text_source", "context", "qa", "prompt"],
        num_proc=args.num_workers,
        cache_file_name=args.output_path + "/data/val_data.arrow",
        load_from_cache_file=True,
    )

    # Print the sizes to verify
    if rank == 0:
        print(f"Train dataset size: {len(tokenized_train_dataset)}")
        print(f"Validation dataset size: {len(tokenized_val_dataset)}")

        print(f"Example from train dataset: {tokenized_train_dataset[0]}")
    # breakpoint()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset["input_ids"],
        eval_dataset=tokenized_val_dataset["input_ids"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if rank == 0:
        print("Starting train")
    trainer.train(resume_from_checkpoint=args.resume)

    finetuned_model_path = os.path.join(output_dir, f"{model_name.split('_')[-1]}-finetuned")

    trainer.model.save_pretrained(finetuned_model_path)

    model_llama = AutoPeftModelForCausalLM.from_pretrained(
        finetuned_model_path,
        low_cpu_mem_usage=True,
        # device_map="auto",
    )

    merged_model = model_llama.merge_and_unload()

    if rank == 0:
        print()
        print("Training done, you can find all the model checkpoints in", output_dir)
        # trainer.save_model(output_dir)
        merged_model.save_pretrained(os.path.join(finetuned_model_path, "merged"))
        tokenizer.save_pretrained(os.path.join(finetuned_model_path, "merged"))
        merged_model.config.to_json_file(os.path.join(os.path.join(finetuned_model_path, "merged"), "config.json"))
