import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb
from datasets import load_dataset, Dataset, concatenate_datasets
import json
from typing import List, Dict
from tqdm import tqdm
import logging
import argparse

def load_generated_samples(json_file_path: str) -> Dataset:
    logging.info(f"Loading generated samples from {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"The file {json_file_path} was not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"The file {json_file_path} is not a valid JSON file.")
        raise
    
    if not isinstance(generated_data, list):
        logging.error("The JSON file does not contain a list of samples.")
        raise ValueError("Invalid JSON format.")
    
    for idx, sample in enumerate(generated_data):
        if not isinstance(sample, dict):
            logging.error(f"Sample at index {idx} is not a dictionary.")
            raise ValueError(f"Invalid sample format at index {idx}.")
        if 'question' not in sample or 'answer' not in sample:
            logging.error(f"Sample at index {idx} is missing 'question' or 'answer' keys.")
            raise ValueError(f"Missing keys in sample at index {idx}.")
        
    logging.info("Converting generated samples to a Dataset object...")
    questions = [sample['question'] for sample in generated_data]
    answers = [sample['answer'] for sample in generated_data]
    generated_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers
    })
    logging.info(f"Loaded {len(generated_dataset)} generated samples.")
    return generated_dataset

def combine_datasets(original_dataset: Dataset, generated_dataset: Dataset) -> Dataset:
    logging.info("Combining the original and generated datasets...")
    combined_dataset = concatenate_datasets([original_dataset, generated_dataset])
    logging.info(f"Combined dataset size: {len(combined_dataset)} samples.")
    return combined_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA models on GSM8k")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path (e.g., 'meta-llama/Llama-2-7b-hf' or 'meta-llama/Llama-3-8b-instruct').")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save fine-tuned model.")
    parser.add_argument("--generated_json_path", type=str, default="./generated_gsm8k_samples.json", help="Path to the generated JSON samples.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,
        device_map='auto',
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=8,  
        lora_alpha=32, 
        lora_dropout=0.1, 
    )
    model = get_peft_model(model, peft_config)


    dataset = load_dataset("gsm8k", "main")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    shuffled_dataset = train_dataset.shuffle(seed=42) 
    train_dataset = shuffled_dataset.select(range(2000))

    generated_dataset = load_generated_samples(args.generated_json_path)
    train_dataset = combine_datasets(train_dataset, generated_dataset)

    def preprocess_function(examples):
        prompts = [f"Question: {q}\nAnswer:" for q in examples['question']]
        answers = examples['answer']
        full_texts = [p + " " + a for p, a in zip(prompts, answers)]
        
        tokenized_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        tokenized_prompts = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        labels = tokenized_full['input_ids'].clone()
        
        for i in range(len(full_texts)):
            prompt_length = (tokenized_prompts['input_ids'][i] != tokenizer.pad_token_id).sum()
            labels[i, :prompt_length] = -100  
        
        tokenized_full['labels'] = labels
        return tokenized_full

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    tokenized_train.set_format(type='torch')
    tokenized_test.set_format(type='torch')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        learning_rate=1e-4,
        fp16=True,  
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__=="__main__":
    main()
