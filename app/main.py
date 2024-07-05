from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
from tqdm import tqdm

from app.collect import collect_active_samples
from app.constants import *

def test_example():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # sae = SAE.load_from_pretrained('checkpoints/6twmlrfz/final_245760000').to(device)
    # model = HookedTransformer.from_pretrained(
    #     "tiny-stories-1L-21M"
    # ).to(device)

    sae, _, _ = SAE.from_pretrained(
        release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
        sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
        device = device
    )
    model = HookedTransformer.from_pretrained(
        "gpt2-small"
    ).to(device)

    dataset = load_dataset("imdb")
    train_subset = dataset['train'].take(4096)
    def tokenize(x):
        output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    
        return output['input_ids'], output['attention_mask']
    batch_size = 128
    dl = DataLoader(train_subset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    collect_active_samples(
        sae,
        model, 
        dl, 
        samples_per_feature=10, 
        device=device, 
        output_dir='data/gpt2'
    )


if __name__ == '__main__':
    test_example()


    