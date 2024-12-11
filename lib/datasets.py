import os
from typing import Dict
import torch
import torch.nn.utils.rnn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import datasets
import random
from functools import partial
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Digits
from datasets import load_dataset, Dataset, concatenate_datasets
import json
from typing import List, Dict
from tqdm import tqdm
import logging


EOS_TOKEN = "<|endoftext_R9VQqF0Ag7|>"
PAD_TOKEN = "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ"
SEP_TOKEN = '----------------------------------------------------------------'
PRINT_PAD_TOKEN = '[PAD]'
PRINT_SEP_TOKEN = '[SEP]'
EOS_TOKEN_ID = 0
SEP_TOKEN_ID = 32021
PAD_TOKEN_ID = 25670

SAMPLE_NUM = int(os.environ.get("SAMPLE_NUM", 0))
GENERATE = int(os.environ.get("GENERATE", 0))
GENERATE_DATA_PATH = str(os.environ.get("GENERATE_DATA_PATH", None))

def load_generated_samples(json_file_path: str) -> Dataset:
    """
    Loads generated samples from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing generated samples.

    Returns:
        Dataset: A dataset containing the generated samples.
    """
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
    
    # Validate that each sample has 'question' and 'answer' keys
    for idx, sample in enumerate(generated_data):
        if not isinstance(sample, dict):
            logging.error(f"Sample at index {idx} is not a dictionary.")
            raise ValueError(f"Invalid sample format at index {idx}.")
        if 'question' not in sample or 'answer' not in sample:
            logging.error(f"Sample at index {idx} is missing 'question' or 'answer' keys.")
            raise ValueError(f"Missing keys in sample at index {idx}.")

    # Convert list of dicts to Dataset
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
    """
    Combines the original and generated datasets into one.

    Args:
        original_dataset (Dataset): The original GSM8K training dataset.
        generated_dataset (Dataset): The dataset containing generated samples.

    Returns:
        Dataset: The combined dataset.
    """
    logging.info("Combining the original and generated datasets...")
    combined_dataset = concatenate_datasets([original_dataset, generated_dataset])
    logging.info(f"Combined dataset size: {len(combined_dataset)} samples.")
    return combined_dataset


class DummyEncoding():
    def __init__(self, ids):
        self.ids = ids

class DigitWrapper(ByteLevelBPETokenizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.digit_tokenizer = Digits(individual_digits=True)
        self.__dict__.update(self.tokenizer.__dict__.items())

    def encode(self, text, digit=True):
        if digit:
            chunks = self.digit_tokenizer.pre_tokenize_str(text)
            res = self.encode_batch([i[0] for i in chunks], digit=False)
            ids = []
            for r in res:
                ids.extend(r.ids)
            enc = DummyEncoding(ids)
            return enc
        return self.tokenizer(text)


    def encode_batch(self, texts, digit=True):
        if digit:
            return [self.encode(text, digit=True) for text in texts]
        return self.tokenizer.encode_batch(texts)
    
    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        return self.tokenizer.get_vocab(with_added_tokens)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    

# str: <<45-40=5>> <<10*1.2=12>> <<12*5=60>> <<10*40=400>> <<400+60=460>> #### 460
# list: ['<<45-40=5>>', ..., '<<400+60=460>> #### 460']
def split_gsm8k_target(target):
    splits = target.split(' ')
    splits = splits[:-3] + [' '.join(splits[-3:])] 
    return [' '+i for i in splits]

def get_gsm8k_dataset(split):
    with open(f'data/gsm8k/{split}.txt', encoding="utf-8") as f:
        lines = [line.split('||') for line in f.read().splitlines() 
                if (len(line) > 0 and not line.isspace()
                                    and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        res = datasets.Dataset.from_dict(
            {
                'src': src_lines, 
                'tgt': [split_gsm8k_target(i) for i in tgt_lines],
                'task_id': [i for i in range(len(src_lines))]
            }
        )
        return res

def get_new_gsm8k_dataset(generated_samples):
    logging.info(f"Incorporating {len(generated_samples)} generated samples...")
    # Extract questions and answers from generated samples
    generated_questions = []
    generated_targets = []
    for sample in generated_samples:
        question = sample.get('question', '').strip()
        answer = sample.get('answer', '').strip()
        if question and answer:
            generated_questions.append(question)
            generated_targets.append(split_gsm8k_target(answer))

    if not generated_questions:
        logging.warning("No valid generated samples to incorporate.")
    else:
        # Assign unique task_ids for generated samples
        start_task_id = 8000
        generated_task_ids = list(range(start_task_id, start_task_id + len(generated_questions)))

        # Create a Dataset object for generated samples
        generated_dataset = Dataset.from_dict(
            {
                'src': generated_questions,
                'tgt': generated_targets,
                'task_id': generated_task_ids
            }
        )

    return generated_dataset

def split_5by5_target(text):
    rationales, target = text.split('####')
    rationales = rationales.strip().split('+')
    target = target.strip()
    # 25 * 10 = 50 + 200 = 250  => ["50 + ", "200 = ", "250"]
    cot_sequences = [r.strip()+' + ' for r in rationales[:-1]] + [rationales[-1].strip() + ' = ', target]
    return cot_sequences

def get_5by5_dataset(split):
    with open(f'data/5by5/{split}.txt', encoding="utf-8") as f:
        lines = [line.split('||') for line in f.read().splitlines() 
                if (len(line) > 0 and not line.isspace()
                                    and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        res = datasets.Dataset.from_dict({'src': [s+' = ' for s in src_lines], 
                                          'tgt': [split_5by5_target(i) for i in tgt_lines],
                                          'task_id': [i for i in range(len(src_lines))]})
        return res


def split_4by4_target(text):
    rationales, target = text.split('####')
    rationales = rationales.strip().split('+')
    target = target.strip()
    # 25 * 10 = 50 + 200 = 250  => ["50 + ", "200 = ", "250"]
    cot_sequences = [r.strip()+' + ' for r in rationales[:-1]] + [rationales[-1].strip() + ' = ', target]
    return cot_sequences

def get_4by4_dataset(split):
    with open(f'data/4by4/{split}.txt', encoding="utf-8") as f:
        lines = [line.split('||') for line in f.read().splitlines() 
                if (len(line) > 0 and not line.isspace()
                                    and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        res = datasets.Dataset.from_dict({'src': [s+' = ' for s in src_lines], 
                                          'tgt': [split_4by4_target(i) for i in tgt_lines],
                                          'task_id': [i for i in range(len(src_lines))]})
        return res


def _tokenize(items, tokenizer):
    src_encoding = tokenizer.encode_batch(items['src'])
    tgt_encoding_list= [tokenizer.encode_batch(i) for i in items['tgt']]
    return {'src_ids': [i.ids for i in src_encoding], 'tgt_ids_list': [[i.ids for i in tgt_encoding]
                                                                       for tgt_encoding in tgt_encoding_list]}

class TextDataset(Dataset):
    def __init__(self, dataset, split, tokenizer):
        self.dataset = eval(f'get_{dataset}_dataset')(split)
        if split == "train":
            self.dataset = self.dataset.select(range(SAMPLE_NUM))

            if GENERATE:
                generated_json_path = GENERATE_DATA_PATH
                generated_dataset = load_generated_samples(generated_json_path)
                generated_dataset = get_new_gsm8k_dataset(generated_dataset)
                self.dataset = combine_datasets(self.dataset, generated_dataset)

        print(self.dataset)
        self.dataset = self.dataset.map(partial(_tokenize, tokenizer=tokenizer),
                                        num_proc=4,
                                        batched=True,
                                        load_from_cache_file=False,
                                        desc="Running tokenizer on dataset",)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def length_to_mask(length_list, max_length=None):
    if max_length is None:
        max_length = max(length_list)
    batch_size = len(length_list)  # Assuming length_tensor has shape (batch_size,)
    mask = torch.arange(max_length).expand(batch_size, -1) < torch.tensor(length_list).unsqueeze(1)
    return mask

def collate_fn(batch, cot=False, seq_len=None, glance=False):
    # Sort the batch in descending order of text length
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    texts = []
    src_lens = []
    for i_in_batch, b in enumerate(batch):
        b['tgt_ids_list'][-1].append(EOS_TOKEN_ID)  # add eos after the last thought
        if cot:
            i = 0
            pre_cot = random.randint(0, len(b['tgt_ids_list'])-1)
            src_ids = b['src_ids']
            tgt_next = []
            if pre_cot == 0:
                tgt_ids = b['tgt_ids_list'][0]
            else:
                tgt_ids = []
                for i, tgt in enumerate(b['tgt_ids_list']):
                    if i < pre_cot:
                        src_ids.extend(tgt)
                    else:
                        tgt_ids.extend(tgt)
                        break
            if i < len(b['tgt_ids_list'])-1:
                tgt_next.extend(b['tgt_ids_list'][i+1])
            
        else:   
            src_ids = b['src_ids']
            tgt_ids = []
            for tgt in b['tgt_ids_list']:
                tgt_ids.extend(tgt)
        
        if seq_len is not None:
            # keep all tgt, rest for src
            tgt_ids = tgt_ids[:seq_len]
            src_ids = src_ids[-(seq_len-len(tgt_ids)):]

        if glance and len(tgt_next)>0 and i_in_batch < 2 and random.random() < 0.1:
            src_lens.append(len(src_ids))
            texts.append(torch.tensor(src_ids + tgt_ids + [SEP_TOKEN_ID] + tgt_next, dtype=torch.int64))
        else:
            src_lens.append(len(src_ids) + 1)
            texts.append(torch.tensor(src_ids + [SEP_TOKEN_ID] + tgt_ids, dtype=torch.int64))

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD_TOKEN_ID)
    attn_mask = length_to_mask([len(text) for text in texts])
    src_mask = length_to_mask(src_lens, max_length=attn_mask.shape[1])
    return texts_padded, attn_mask, src_mask


def collate_fn_test(batch, seq_len):
    # Sort the batch in descending order of text length
    src_list = []
    src_lens = []
    tgt_texts = []
    task_ids = []

    for b in batch:
        src_ids = b['src_ids']
        
        # src_ids = src_ids[:seq_len-2] # if src is too long, cut it; one for sep, one for tgt prediction
        if len(src_ids) >= seq_len:
            raise ValueError(f'seq_len={seq_len} is too short, one src has length {len(src_ids)}')

        src_lens.append(len(src_ids) + 1)
        src_list.append(torch.tensor(src_ids + [SEP_TOKEN_ID], dtype=torch.int64))
        tgt_texts.append(b['tgt'][-1])

        task_ids.append(b['task_id'])

    dummy_seq = torch.tensor([0]*seq_len, dtype=torch.int64)  # add a dummy seq with length=seq_len
    texts_padded = pad_sequence([dummy_seq]+src_list, batch_first=True, padding_value=PAD_TOKEN_ID)
    texts_padded = texts_padded[1:]  # drop the dummy sequence
    src_mask = length_to_mask(src_lens, max_length=seq_len)
    task_ids = torch.tensor(task_ids, dtype=torch.int64)

    return texts_padded, src_mask, tgt_texts, task_ids


def infinite_loader(data_loader):
    while True:
        yield from data_loader
    
def get_tokenizer(digit=False):
    tokenizer_path = os.path.join('misc/owt2_tokenizer.json')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return DigitWrapper(tokenizer) if digit else tokenizer

def get_dataloader(dataset, split, batch_size, tokenizer, seq_len, cot, glance=False):
    dataset = TextDataset(dataset, split, tokenizer)
    if split != 'test':
        sampler = DistributedSampler(dataset)  
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(collate_fn, cot=cot, seq_len=seq_len, glance=glance)
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=False)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(collate_fn_test, seq_len=seq_len)
        )
    return data_loader


def get_dataloaders(dataset, batch_size, seq_len, cot, digit, glance, only_test=False):
    if seq_len is None:
        seq_len = 1024

    tokenizer = get_tokenizer(digit)
    word2idx = {k.encode('utf-8'):v for k,v in tokenizer.get_vocab().items()}
    idx2word = {v:k for k,v in word2idx.items()}
    if only_test:
        test_loader = get_dataloader(dataset, 'test', batch_size, tokenizer, seq_len, cot)
        return (test_loader,), (word2idx, idx2word), tokenizer
    else:
        train_loader = get_dataloader(dataset, 'train', batch_size, tokenizer, seq_len, cot, glance)
        valid_loader = get_dataloader(dataset, 'valid', batch_size, tokenizer, seq_len, cot)
        return (train_loader, valid_loader), (word2idx, idx2word), tokenizer


if __name__ == '__main__':
    digit_tokenizer = get_tokenizer(digit=True)
    texts = ["This is a text", "This is 1+11=12"]
    # [1116, 321, 258, 3241]
    # [1116, 321, 221, 17, 11, 17, 17, 29, 17, 18]
    [print(r.ids) for r in digit_tokenizer.encode_batch(texts)]
    
    # [1116, 321, 258, 3241]
    # [1116, 321, 406, 11, 1970, 29, 2100]
    [print(r.ids) for r in digit_tokenizer.tokenizer.encode_batch(texts)]
    print(digit_tokenizer.tokenizer.get_vocab_size())
    print(PAD_TOKEN_ID, SEP_TOKEN_ID)