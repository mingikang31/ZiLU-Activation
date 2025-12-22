"""
Processing code from nanoGPT:
https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading WikiText-103 dataset (more reliable than OpenWebText)...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", num_proc=num_proc_load_dataset)
    
    # Filter out empty lines (WikiText has many empty entries)
    def filter_empty(example):
        return len(example['text'].strip()) > 0
    
    # Filter each split individually
    train_filtered = dataset['train'].filter(filter_empty, num_proc=num_proc)
    val_filtered = dataset['validation'].filter(filter_empty, num_proc=num_proc)
    
    # Create proper DatasetDict
    split_dataset = DatasetDict({
        'train': train_filtered,
        'val': val_filtered
    })
    
    # Define the processing function
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset - this will now work because split_dataset is a DatasetDict
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print(f"Dataset preparation complete! Files saved as train.bin and val.bin")





'''NEW CODE'''


### Fom reference.py
import torch 
from transformers.file_utils import cached_path
from transformers import BertTokenizer
import tqdm 
from torch.utils.data import DataLoader

# Download and tokenize wikitext-103 training dataset 

data_path = "./Data/wikitext-103-dataset"


# Tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False) 

if os.path.isfile(data_path): 
    dataset = torch.load(data_path)
else: 
    dataset_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.train.tokens")

    # Open Data 
    with open(dataset_file, "r", encoding="utf-8") as f: 
        dataset = f.readlines() 

    dataset = list(
        tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(
                line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]') for line in tqdm(dataset)
            )
        )
    )    

    # Convert dataset to torch tensor 
    dataset = torch.tensor([index for line in dataset for index in line], dtype = torch.long) 

    torch.save(dataset, data_path)


train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True) 


# Training Loop 
for batch in train_dataloader:
    batch = batch.transpose(0, 1).contiguous().to('cuda') # shape [Sequence_Length, Batch]

    logits, loss = model(batch, labels=batch) 



