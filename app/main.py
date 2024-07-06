from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

from app.collect import create_sample_statistics
from app.constants import *

def test_example():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'



    dataset = load_dataset("imdb")
    subsets = {}
    # get label names for each label
    label_names = dataset['train'].features['label']

    for label in set(dataset['train']['label']):  # Assuming you are working with the 'train' split
        label_name = label_names.int2str(label)
        subsets[label_name] = dataset['train'].filter(lambda example: example['label'] == label)
        
        print('Count of labels', label_name, len(subsets[label_name]))

    # sae, _, _ = SAE.from_pretrained(
    #     release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    #     sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
    #     device = device
    # )
    # model = HookedTransformer.from_pretrained(
    #     "gpt2-small"
    # ).to(device)

    sae = SAE.load_from_pretrained('checkpoints/6twmlrfz/final_245760000').to(device)
    model = HookedTransformer.from_pretrained(
        "tiny-stories-1L-21M"
    ).to(device)


    batch_size = 128
    
    def tokenize(x):
        output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    
        return output['input_ids'], output['attention_mask']
    
    for label_name, subset in subsets.items():
        print(f"Collecting statistics for label {label_name}")
        subset = subset.take(4096)
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

        create_sample_statistics(
            sae,
            model, 
            dl, 
            samples_per_feature=10, 
            device=device, 
            output_dir=f'data/gpt2-imdb-{str(label_name).lower()}',
            n_fts_to_analyse=int(sae.cfg.d_sae / 10)
        )

if __name__ == '__main__':
    test_example()


    