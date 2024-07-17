import torch
import pandas as pd
from datasets import load_dataset
import os
import requests
import json
import time
from torch.utils.data import DataLoader, Dataset


def _download_file(url, local_filename, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_filename, 'wb') as file:
                file.write(response.content)
            return
        except requests.ConnectionError:
            retries += 1
            print(f"Connection error. Retrying {retries}/{max_retries}...")
            time.sleep((2 ** retries)  / 4)
    raise Exception(f"Failed to download file after {max_retries} attempts.")


# ---------------------------------------------------------------------------------------------------------
# ------------------------------------------------- AESOP -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

AESOP_URL = "https://raw.githubusercontent.com/itayniv/aesop-fables-stories/master/public/aesopFables.json"
LOCAL_AESOP_FILENAME = os.path.join('data', "aesopFables.json")

class _FablesDataset(Dataset):
    def __init__(self, fables):
        self.fables = fables['stories']

    def __len__(self):
        return len(self.fables)

    def __getitem__(self, idx):
        fable = self.fables[idx]
        return fable['title'] + "\n" + " ".join(fable['story'])



def _aesop_dataloader(fables, tokenizer, batch_size, sequence_length, padding='max_length'):
    dataset = _FablesDataset(fables)

    def tokenize(batch):
        return tokenizer(batch, padding=padding, truncation=True, max_length=sequence_length, return_tensors='pt')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)
    return dl

def load_raw_aesop_JSON():
    with open(LOCAL_AESOP_FILENAME, 'r') as file:
        data = json.load(file)
    return data

def load_aesop(tokenizer, batch_size, sequence_length):
    if not os.path.exists(LOCAL_AESOP_FILENAME):
        print(f"{LOCAL_AESOP_FILENAME} does not exist. Downloading...")
        _download_file(AESOP_URL, LOCAL_AESOP_FILENAME)

    data = load_raw_aesop_JSON()
    
    dataloader = _aesop_dataloader(data, tokenizer, batch_size, sequence_length)
    return dataloader

# ---------------------------------------------------------------------------------------------------------
# ------------------------------------------------- CLEAR -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

CLEAR_URL = "https://github.com/scrosseye/CLEAR-Corpus/blob/main/CLEAR_corpus_final.xlsx?raw=true"
LOCAL_CLEAR_FILENAME = os.path.join('data', "CLEAR_corpus_final.xlsx")

class _ClearDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sample = self.texts.iloc[idx]
        return sample['Excerpt'], sample['BT_easiness']

def _clear_dataloader(texts, tokenizer, batch_size, sequence_length, padding='max_length'):
    dataset = _ClearDataset(texts)

    def tokenize(batch):
        texts = [b[0] for b in batch]
        d = tokenizer(texts, padding=padding, truncation=True, max_length=sequence_length, return_tensors='pt')

        d['labels'] = torch.tensor([b[1] for b in batch])
        return d

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)
    return dl

# Main function to load DataLoader
def load_clear(tokenizer, batch_size, sequence_length):
    if not os.path.exists(LOCAL_CLEAR_FILENAME):
        print(f"{LOCAL_CLEAR_FILENAME} does not exist. Downloading...")
        _download_file(CLEAR_URL, LOCAL_CLEAR_FILENAME)

    data = pd.read_excel(LOCAL_CLEAR_FILENAME) 
    dataloader = _clear_dataloader(data, tokenizer, batch_size, sequence_length)
    return dataloader

# ---------------------------------------------------------------------------------------------------------
# ------------------------------------------------- PILE -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

def load_pile10k(tokenizer, batch_size, sequence_length, num_samples=5000, padding='max_length'):
    dataset = load_dataset('NeelNanda/pile-10k', split="train")
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    def tokenize(batch):
        text_input = [b['text'] for b in batch]
        return tokenizer(text_input, padding=padding, truncation=True, max_length=sequence_length, return_tensors='pt')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    return dl

