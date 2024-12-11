import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import json
import csv
from typing import List, Dict
from tqdm import tqdm  
from datasets import load_dataset
import argparse

sample_questions = [
    "If you have 3 apples and you buy 2 more, how many apples do you have in total?",
    "What is the product of 12 and 15?",
    "A train travels at 60 miles per hour for 2 hours. How far does it travel?",
    "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?",
    "John has twice as many candies as Mary. If Mary has 15 candies, how many does John have?",
    "A car uses 3 gallons of fuel to travel 30 miles. How many gallons are needed to travel 120 miles?",
    "If a book costs $15 and you have $60, how many books can you buy?",
    "What is the sum of the angles in a triangle?",
    "If you save $200 each month, how much will you have saved after one year?",
    "A garden has 5 rows of flowers with 12 flowers in each row. How many flowers are there in total?",
    "If it takes 5 workers 4 hours to build a wall, how long would it take 10 workers to build the same wall, assuming they work at the same rate?",
    "Sarah bought 7 packs of stickers, each containing 9 stickers. How many stickers did she buy in total?",
    "A baker made 48 cupcakes and wants to package them equally into boxes of 6. How many boxes does he need?",
    "The temperature dropped from 20°C to -5°C overnight. What was the total change in temperature?",
    "A tank holds 150 liters of water. If 45 liters are used, how much water remains in the tank?"
]

def extract_questions_from_gsm8k(num_questions: int = 100):
    dataset = load_dataset("gsm8k", "main")
    train_dataset = dataset['train']
    
    shuffled = train_dataset.shuffle(seed=42)
    questions = shuffled['question'][:num_questions]
    
    return questions

def generate_programmatic_questions(num_questions: int = 200):
    questions = []
    operations = ['add', 'subtract', 'multiply', 'divide']
    for _ in range(num_questions):
        a, b = random.randint(1, 100), random.randint(1, 100)
        operation = random.choice(operations)
        if operation == 'add':
            question = f"What is {a} plus {b}?"
        elif operation == 'subtract':
            question = f"What is {a} minus {b}?"
        elif operation == 'multiply':
            question = f"What is the product of {a} and {b}?"
        elif operation == 'divide':

            b = random.randint(1, 100)
            a = b * random.randint(1, 10)
            question = f"What is {a} divided by {b}?"
        questions.append(question)
    return questions

def load_model_and_tokenizer(model_path: str, base_model_name: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=use_4bit,
        device_map='auto',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model = PeftModel.from_pretrained(model, model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model

def generate_prompt() -> str:
    templates = [
        "Solve the following math problem:\n\nQuestion: {}\nAnswer:",
        "Here's a math question for you:\n\nQuestion: {}\nAnswer:",
        "Please provide a detailed solution to the following problem:\n\nQuestion: {}\nAnswer:",
        "Math Challenge:\n\nQuestion: {}\nAnswer:",
        "Can you help me solve this math problem?\n\nQuestion: {}\nAnswer:",
        "Attempt the following math question:\n\nQuestion: {}\nAnswer:",
        "Let's work through this math problem together:\n\nQuestion: {}\nAnswer:",
        "Provide a step-by-step solution for the following math problem:\n\nQuestion: {}\nAnswer:",
    ]
    
    if not sample_questions:
        raise ValueError("The 'sample_questions' list is empty. Please provide some questions.")
    
    question = random.choice(sample_questions)
    template = random.choice(templates)
    return template.format(question)

def generate_sample(model, tokenizer, device: torch.device, generation_config: Dict, prompt: str) -> Dict:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            **generation_config
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        question_part, answer_part = generated_text.split("Answer:")
        question = question_part.replace("Question:", "").strip()
        answer = answer_part.strip()
        
        return {"question": question, "answer": answer}
    except ValueError:
        print(f"Unexpected format in generated text:\n{generated_text}\n")
        return {"question": "", "answer": ""}

def generate_samples(model, tokenizer, device: torch.device, generation_config: Dict, num_samples: int = 1000) -> List[Dict]:
    generated_data = []
    
    with tqdm(total=num_samples, desc="Generating Samples", unit="sample") as pbar:
        while len(generated_data) < num_samples:
            prompt = generate_prompt()
            sample = generate_sample(model, tokenizer, device, generation_config, prompt)
            if sample["question"] and sample["answer"]:
                generated_data.append(sample)
                pbar.update(1)
            else:
                continue
    return generated_data

def post_process_data(generated_data: List[Dict]) -> List[Dict]:
    processed_data = []
    seen_questions = set()
    
    for sample in generated_data:
        question = sample["question"]
        answer = sample["answer"]
        
        if question in seen_questions:
            continue
        seen_questions.add(question)
        
        if len(question) < 20 or len(answer) < 20:
            continue  
        
        processed_data.append({
            "question": question,
            "answer": answer
        })
    
    return processed_data

def evaluate_sample_quality(model, tokenizer, device: torch.device, samples: List[Dict], threshold: float = 0.7) -> List[Dict]:
    verified_samples = []
    for sample in tqdm(samples, desc="Evaluating Samples", unit="sample"):
        prompt = f"Evaluate the quality of the following QA pair:\n\nQuestion: {sample['question']}\nAnswer: {sample['answer']}\n\nQuality (1-10):"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,
                temperature=0.7,
                top_p=0.9,
                do_sample=False
            )
        evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            score = float(evaluation.split()[-1])
            if score / 10.0 >= threshold:
                verified_samples.append(sample)
        except (ValueError, IndexError):
            print(f"Failed to evaluate sample: {sample}")
    return verified_samples

def save_data_as_json(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_data_as_csv(data: List[Dict], file_path: str):
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)

def main():
    parser = argparse.ArgumentParser(description="Generate math problem samples using LLaMA models.")
    parser.add_argument("--fine_tuned_model_path", type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name (e.g., 'meta-llama/Llama-2-7b-hf' or 'meta-llama/Llama-3-8b-instruct').")
    parser.add_argument("--num_samples_to_generate", type=int, default=2000, help="Number of samples to generate.")
    parser.add_argument("--output_json_file", type=str, default="./generated_gsm8k_samples.json", help="Path to save the generated JSON samples.")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="Threshold for sample quality verification.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    global generation_config
    generation_config = {
        "max_new_tokens": 300,        
        "temperature": 0.7,          
        "top_p": 0.9,               
        "top_k": 50,                
        "do_sample": True,           
        "eos_token_id": None,      
        "pad_token_id": None,         
        "repetition_penalty": 1.2,   
    }

    additional_gsm8k_questions = extract_questions_from_gsm8k(100)  
    programmatic_questions = generate_programmatic_questions(200)  
    sample_questions.extend(additional_gsm8k_questions)
    sample_questions.extend(programmatic_questions)
    print(f"Total number of sample questions: {len(sample_questions)}")

    tokenizer, model = load_model_and_tokenizer(
        model_path=args.fine_tuned_model_path,
        base_model_name=args.base_model_name,
        use_4bit=True
    )

    generation_config["eos_token_id"] = tokenizer.eos_token_id
    generation_config["pad_token_id"] = tokenizer.pad_token_id

    print(f"Generating {args.num_samples_to_generate} samples...")
    raw_generated_data = generate_samples(model, tokenizer, device, generation_config, num_samples=args.num_samples_to_generate)
    print(f"Generated {len(raw_generated_data)} raw samples.")

    clean_generated_data = post_process_data(raw_generated_data)
    print(f"After post-processing, {len(clean_generated_data)} samples remain.")

    print("Verifying the quality of generated samples...")
    high_quality_samples = evaluate_sample_quality(model, tokenizer, device, clean_generated_data, threshold=args.quality_threshold)
    print(f"After quality verification, {len(high_quality_samples)} high-quality samples remain.")

    save_data_as_json(clean_generated_data, args.output_json_file)

if __name__ == "__main__":
    main()
