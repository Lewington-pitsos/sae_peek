from datasets import load_dataset
import os
import requests
import json
import time
from torch.utils.data import DataLoader, Dataset

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

def _download_aesop(url, local_filename, max_retries=5):
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


def _aesop_dataloader(fables, tokenizer, batch_size, sequence_length):
    dataset = _FablesDataset(fables)

    def tokenize(batch):
        return tokenizer(batch, padding='longest', truncation=True, max_length=sequence_length, return_tensors='pt')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)
    return dl

def load_aesop(tokenizer, batch_size, sequence_length):
    if not os.path.exists(LOCAL_AESOP_FILENAME):
        print(f"{LOCAL_AESOP_FILENAME} does not exist. Downloading...")
        _download_aesop(AESOP_URL, LOCAL_AESOP_FILENAME)

    with open(LOCAL_AESOP_FILENAME, 'r') as file:
        data = json.load(file)

    dataloader = _aesop_dataloader(data, tokenizer, batch_size, sequence_length)
    return dataloader




def load_pile10k(tokenizer, batch_size, sequence_length, num_samples=5000):
    dataset = load_dataset('NeelNanda/pile-10k', split="train")
    dataset = dataset.select(range(num_samples))

    def tokenize(batch):
        text_input = [b['text'] for b in batch]
        return tokenizer(text_input, padding='longest', truncation=True, max_length=sequence_length, return_tensors='pt')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    return dl

