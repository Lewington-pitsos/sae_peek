from collections import defaultdict
import os
import json

import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

from app.collect import save_sample_statistics
from app.constants import *
from app.assess import llm_assessment, validate_assessment_args
    

def build_sae(sae_model, device, sae_id=None):
    if os.path.exists(sae_model):
        return SAE.load_from_pretrained(sae_model, device=device)

    sae, _, _ =SAE.from_pretrained(
        release = sae_model, # see other options in sae_lens/pretrained_saes.yaml
        sae_id = sae_id, 
        device = device
    )

    return sae

def validate_activation_args(*args, **kwargs):
    activation_dir = kwargs.get('activation_dir')

    if not os.path.exists(os.path.dirname(activation_dir)):
        raise ValueError(f"could not locate parent directory for output file {activation_dir}")

    if os.path.exists(activation_dir):
        if not len(os.listdir(activation_dir)) == 0:
            raise ValueError(f'activation_dir {activation_dir} already exists and is not empty')

    batches_in_stats_batch = kwargs.get('batches_in_stats_batch')
    if batches_in_stats_batch < 1:
        raise ValueError(f'batches_in_stats_batch must be at least 1, but got {batches_in_stats_batch}')


def generate_sae_activations(
        sae_model, 
    
        dataloader,
        activation_dir, 

        device=None, 
        transformer=None, 
        samples_per_feature=15,
        sae_id=None,
        feature_indices=None,
        batches_in_stats_batch=1,
        save_activations=True
    ):

    if not os.path.exists(activation_dir):
        print('Creating activation directory', activation_dir)
        os.makedirs(activation_dir)

    validate_activation_args(**locals())

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else: 
            device = 'cpu'
        
    sae = build_sae(sae_model, device, sae_id)
    if transformer is None:
        transformer = sae.cfg.model_name
        print(f'Using transformer {transformer}')
    if isinstance(transformer, str):
        model = HookedTransformer.from_pretrained(transformer, device=device)
    else:
        model = transformer

    if feature_indices is None:
        feature_indices = list(range(int(sae.cfg.d_sae)))
        print(f'Running Analysis on all {len(feature_indices)} features')

    save_sample_statistics(
        sae,
        model, 
        dataloader, 
        batches_in_stats_batch=batches_in_stats_batch,
        samples_per_feature=samples_per_feature, 
        device=device, 
        feature_indices=feature_indices,
        output=activation_dir,
        save_activations=save_activations
    )

def sae_assessment(
        activation_dir, 
        output,
        samples_per_feature=15,
        **activation_kwargs
    ):
    if not os.path.exists(os.path.dirname(output)):
        print('Creating parent directory for', output)
        os.mkdir(os.path.dirname(output))        

    validate_assessment_args(**locals())

    generate_sae_activations(
        activation_dir=activation_dir, 
        samples_per_feature=samples_per_feature,
        **activation_kwargs)

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
            if d['assessment'] is not None:
                means[m].append(d['assessment'][m])
            else:
                print(f'the assessment was None, the LLM likely failed to make an assessment')

    for k, v in means.items():
        print(f'{k}: {sum(v) / len(v)}, std: {np.std(v)}')

    