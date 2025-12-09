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