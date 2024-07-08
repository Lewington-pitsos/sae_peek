from collections import defaultdict
import os
import json

import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

from app.collect import create_sample_statistics
from app.constants import *
from app.assess import llm_assessment
    

def build_sae(sae_model, device, sae_id=None):
    if os.path.exists(sae_model):
        return SAE.load_from_pretrained(sae_model, device=device)

    sae, _, _ =SAE.from_pretrained(
        release = sae_model, # see other options in sae_lens/pretrained_saes.yaml
        sae_id = sae_id, 
        device = device
    )

    return sae

def validate_args(*args, **kwargs):
    activation_dir = kwargs.get('activation_dir')

    if os.path.exists(activation_dir):
        # check if it is empty
        if not len(os.listdir(activation_dir)) == 0:
            raise ValueError(f'activation_dir {activation_dir} already exists and is not empty')
    
    output = kwargs.get('output')
    if os.path.exists(output):
        raise ValueError(f'output file {output} already exists')
    

def sae_assessment(
        dataset, 
        n_samples,

        sae_model, 
        device, 
    
        batch_size, 
        sequence_length, 

        activation_dir, 
        output,


        transformer_name=None, 
        samples_per_feature=5,
        sae_id=None,
        n_feats_to_analyse=None,
    ):

    validate_args(**locals())


    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    if isinstance(dataset, str):
        print('loading huggingface dataset:', dataset)
        dataset = load_dataset(dataset)
        dataset = dataset['train'].take(n_samples)

    sae = build_sae(sae_model, device, sae_id)
    if transformer_name is None:
        transformer_name = sae.cfg.model_name
        print(f'Using transformer {transformer_name}')

    model = HookedTransformer.from_pretrained(transformer_name, device=device)

    def tokenize(x):
        output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=sequence_length, padding='max_length')
    
        return output['input_ids'], output['attention_mask']
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)
    

    if n_feats_to_analyse is None:
        n_feats_to_analyse = int(sae.cfg.d_sae / 100)
        print(f'Running Analysis on the first {n_feats_to_analyse} features')

    create_sample_statistics(
        sae,
        model, 
        dl, 
        samples_per_feature=samples_per_feature, 
        device=device, 
        output=activation_dir,
        n_fts_to_analyse=n_feats_to_analyse
    )

    llm_assessment(activation_dir, output, n_feats_to_analyse, samples_per_feature)

    metrics_of_interest = [
        'feature_coherence',
        'feature_complexity',
    ]

    with open(output) as f:
        data = json.load(f)

    means = defaultdict(list)

    for d in data:
        for m in metrics_of_interest:
            means[m].append(d['human_description'][m])

    for k, v in means.items():
        print(f'{k}: {sum(v) / len(v)}, std: {np.std(v)}')

if __name__ == '__main__':
    sae_assessment(
        dataset='NeelNanda/pile-10k',
        n_samples=512,
        sae_model='gpt2-small-res-jb',
        sae_id='blocks.0.hook_resid_pre',
        device='cpu',
        batch_size=16,
        sequence_length=128,
        activation_dir='data/pile-10k',
        output='cruft/pile-10k-comp.json',
        n_feats_to_analyse=25
    )

    