import h5py
import os
import torch
from tqdm import tqdm

from app.constants import *
from app.storage import ActivationDataset, data_from_tensor

                            
def new_topk_samples(start_idx, acts, current_maxes, current_max_indices, topk):
    max_acts = torch.max(acts, dim=1).values

    current_batch_topk = min(max_acts.shape[0], topk) # in case we are on the last batch and it's smaller than usual

    topk_vals, topk_indices = torch.topk(max_acts, k=current_batch_topk, dim=0)
    topk_indices += start_idx

    all_maxes = torch.cat([current_maxes, topk_vals], dim=0)
    all_indices = torch.cat([current_max_indices, topk_indices], dim=0)

    new_maxes, new_indices_idx = torch.topk(all_maxes, topk, dim=0)

    new_indices = all_indices[new_indices_idx, torch.arange(all_indices.shape[1])]

    return new_maxes, new_indices

def collect_feature_stats(start_idx, n_ft, activations, stats, topk):
    attention_mask, _, activations = data_from_tensor(activations, n_ft)

    masked_activations = activations * attention_mask

    n_elements = torch.sum(attention_mask) # (1)


    batch_mean = torch.sum(masked_activations, dim=(0, 1)) / n_elements # (n_ft)
    batch_nonzero_prop = torch.sum(masked_activations > 0, dim=(0, 1)) / n_elements # (n_ft)


    stats['mean'].add_(batch_mean)
    stats['nonzero_proportion'].add_(batch_nonzero_prop)

    stats['max_activations'], stats['max_activation_indices'] = new_topk_samples(start_idx, masked_activations, stats['max_activations'], stats['max_activation_indices'], topk)


def get_features(sae, transformer, input_ids, attention_mask):
    _, cache = transformer.run_with_cache(
        input_ids, 
        attention_mask=attention_mask, 
        prepend_bos=True, 
        stop_at_layer=sae.cfg.hook_layer + 1)

    hidden_states = cache[sae.cfg.hook_name]

    features = sae.encode(hidden_states)

    return features

def typecast(input_ids, attention_mask, device):
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.int)
    else:
        input_ids = input_ids.to(device)
    
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.int)
    else:
        attention_mask = attention_mask.to(device)
    
    return input_ids, attention_mask


def _init_stats(n_fts_to_analyse, device, feature_indices, samples_per_feature, sequence_length):
    return {
        'mean': torch.zeros(n_fts_to_analyse).to(device),
        'nonzero_proportion': torch.zeros(n_fts_to_analyse).to(device),
        'max_activations': torch.zeros(samples_per_feature, n_fts_to_analyse).to(device),
        
        'feature_indices': torch.tensor(feature_indices),

        'max_activation_indices': torch.empty(samples_per_feature, n_fts_to_analyse).to(device),
        'top_samples': torch.empty(samples_per_feature, sequence_length, n_fts_to_analyse)
    }

def save_sample_statistics(
        sae, 
        transformer, 
        dataloader, 
        device, 
        output, 
        feature_indices, 
        batches_in_stats_batch=1,
        samples_per_feature=15
    ):
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # or else transformers will complain about tqdm starting a parallel process.
    outer_batch_size = batch_size * batches_in_stats_batch

    validation_sample = next(iter(dataloader))
    assert 'input_ids' in validation_sample and 'attention_mask' in validation_sample, f'dataloader samples must have "attention_mask" and "input_ids" in order to be valid. First sample was: {validation_sample}'
    

    batch_size = validation_sample['input_ids'].shape[0]
    sequence_length = validation_sample['input_ids'].shape[1]
    n_fts_to_analyse = len(feature_indices)

    ds = ActivationDataset(output)
    stats = _init_stats(
        n_fts_to_analyse=n_fts_to_analyse, 
        device=device, 
        feature_indices=feature_indices, 
        samples_per_feature=samples_per_feature,
        sequence_length=sequence_length
    )

    with torch.no_grad():
        outer_batch = torch.tensor([]).to(device)
        batches_saved = 0
        for i, batch in enumerate(tqdm(dataloader, desc='Generating and Analysing Activations')):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            
            input_ids, attention_mask = typecast(input_ids, attention_mask, device)

            # residuals into SAE, get features
            features = get_features(sae, transformer, input_ids, attention_mask)
            if n_fts_to_analyse < features.shape[2]:
                features = features[:, :, feature_indices]
            features = torch.cat([features, attention_mask.unsqueeze(-1), input_ids.unsqueeze(-1)], dim=-1)

            # calculate activation statistics, save to disk
            

            outer_batch = torch.cat([outer_batch, features], dim=0)
            del features
            if outer_batch.shape[0] == outer_batch_size:
                collect_feature_stats(batches_saved*batch_size, n_fts_to_analyse, outer_batch, stats, samples_per_feature)
                batches_saved = (i + 1)
                ds.add(outer_batch.to('cpu'))
                outer_batch = torch.tensor([]).to(device)

        if outer_batch.shape[0] > 0:
            collect_feature_stats(batches_saved*batch_size, n_fts_to_analyse, outer_batch, stats, samples_per_feature)
            ds.add(outer_batch.to('cpu'))

    n_outer_batches = i / batches_in_stats_batch
    stats['mean'] /= n_outer_batches
    stats['nonzero_proportion'] /= n_outer_batches

    ds.finalize(stats)
    