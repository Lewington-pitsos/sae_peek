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

    if not os.path.exists(os.path.dirname(activation_dir)):
        raise ValueError(f"could not locate parent directory for output file {output}")

    if os.path.exists(activation_dir):
        # check if it is empty
        if not len(os.listdir(activation_dir)) == 0:
            raise ValueError(f'activation_dir {activation_dir} already exists and is not empty')
    
    output = kwargs.get('output')
    if os.path.exists(output):
        raise ValueError(f'output file {output} already exists')

    if not os.path.exists(os.path.dirname(output)):
        raise ValueError(f"could not locate parent directory for output file {output}")

    if not os.path.exists(CREDENTIALS_FILE):
        raise ValueError(f"could not locate credentials file {CREDENTIALS_FILE}")
    

def sae_assessment(
        dataset, 
        n_samples,

        sae_model, 
    
        batch_size, 
        sequence_length, 

        activation_dir, 
        output,


        device=None, 
        transformer=None, 
        samples_per_feature=15,
        sae_id=None,
        feature_indices=None,
    ):

    validate_args(**locals())


    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else: 
            device = 'cpu'
        
    if isinstance(dataset, str):
        print('loading huggingface dataset:', dataset)
        dataset = load_dataset(dataset)
        dataset = dataset['train'].take(n_samples)

    sae = build_sae(sae_model, device, sae_id)
    if transformer is None:
        transformer = sae.cfg.model_name
        print(f'Using transformer {transformer}')
    if isinstance(transformer, str):
        model = HookedTransformer.from_pretrained(transformer, device=device)
    else:
        model = transformer

    def tokenize(x):
        output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=sequence_length, padding='max_length')
    
        return output['input_ids'], output['attention_mask']
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    if feature_indices is None:
        feature_indices = list(range(int(sae.cfg.d_sae / 100)))
        print(f'Running Analysis on the first {feature_indices} features')

    create_sample_statistics(
        sae,
        model, 
        dl, 
        samples_per_feature=samples_per_feature, 
        device=device, 
        output=activation_dir,
        feature_indices=feature_indices
    )

    llm_assessment(activation_dir, output, samples_per_feature)

    metrics_of_interest = [
        'feature_coherence',
        'feature_complexity',
    ]

    with open(output) as f:
        data = json.load(f)

    means = defaultdict(list)

    for d in data:
        for m in metrics_of_interest:
            if d['human_description'] is not None:
                means[m].append(d['human_description'][m])
            else:
                print(f'the assessment was None, the LLM likely failed to make an assessment')

    for k, v in means.items():
        print(f'{k}: {sum(v) / len(v)}, std: {np.std(v)}')

if __name__ == '__main__':
    model = HookedTransformer.from_pretrained("gpt2", device='cuda')


    for sae_id in [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.2.hook_resid_pre",
        "blocks.3.hook_resid_pre",
        "blocks.4.hook_resid_pre",
        "blocks.5.hook_resid_pre",
        "blocks.6.hook_resid_pre",
        "blocks.7.hook_resid_pre",
        "blocks.8.hook_resid_pre",
        "blocks.9.hook_resid_pre",
        "blocks.10.hook_resid_pre",
        "blocks.11.hook_resid_pre",
        "blocks.11.hook_resid_post",
    ]:    
        sae_assessment(
            dataset='NeelNanda/pile-10k',
            n_samples=4096,
            sae_model='gpt2-small-res-jb',
            sae_id=sae_id,
            transformer=model,
            batch_size=256,
            sequence_length=128,
            activation_dir=f'data/pile10k-v2-{sae_id}',
            output=f'cruft/pile10k-v2-{sae_id}.json',
            feature_indices=list(range(150, 300))
        )

    