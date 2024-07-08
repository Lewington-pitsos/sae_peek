import h5py
import os
import torch
from tqdm import tqdm

from app.constants import *
from app.storage import ActivationDataset, data_from_tensor

                            
def new_topk_samples(start_idx, acts, current_maxes, current_max_indices, k):
    max_acts = torch.max(acts, dim=1).values

    topk_vals, topk_indices = torch.topk(max_acts, k, dim=0)
    topk_indices += start_idx

    all_maxes = torch.cat([current_maxes, topk_vals], dim=0)
    all_indices = torch.cat([current_max_indices, topk_indices], dim=0)

    new_maxes, new_indices_idx = torch.topk(all_maxes, k, dim=0)

    new_indices = all_indices[new_indices_idx, torch.arange(all_indices.shape[1])]

    return new_maxes, new_indices

def collect_feature_stats(start_idx, n_ft, activations, stats, topk):
    # activations is (bs, seq_len, n_ft + 2)
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

def create_sample_statistics(sae, transformer, dataloader, device, output, samples_per_feature=5, feature_indices=None):
    if feature_indices is None:
        feature_indices = list(range(sae.cfg.d_sae))
    
    n_fts_to_analyse = len(feature_indices)

    ds = ActivationDataset(output)
    stats = {
        'mean': torch.zeros(n_fts_to_analyse).to(device),
        'feature_indices': torch.tensor(feature_indices),
        'nonzero_proportion': torch.zeros(n_fts_to_analyse).to(device),
        'max_activations': torch.zeros(samples_per_feature, n_fts_to_analyse).to(device),
        'max_activation_indices': torch.zeros(samples_per_feature, n_fts_to_analyse).to(device),
    }

    with torch.no_grad():
        for i, (input_ids, att_mask) in enumerate(tqdm(dataloader, desc='Generating and Analysing Activations')):
            batch_size = input_ids.shape[0]
            input_ids, att_mask = input_ids.to(device), att_mask.to(device)
            features = get_features(sae, transformer, input_ids, att_mask)

            # truncate to selected features, essentially randomly sample n_features features
            if n_fts_to_analyse < features.shape[2]:
                features = features[:, :, feature_indices]

            features = torch.cat([features, att_mask.unsqueeze(-1), input_ids.unsqueeze(-1)], dim=-1)

            ds.add(features.to('cpu'))

            collect_feature_stats(i*batch_size, n_fts_to_analyse, features, stats, samples_per_feature)

    ds.finalize(stats)
