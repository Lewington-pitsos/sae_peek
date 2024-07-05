import torch
from termcolor import colored
from tqdm import tqdm

from app.constants import *
from app.storage import ActivationDataset, data_from_tensor

# 4766 - get on the beers
# 14997 - salvery/racism

class Corpus():
    def __init__(self, data_dir):
        self.ds = ActivationDataset(data_dir)

        self.stats_tensor = self.ds.load_stats()
        self.n_fts = self.stats_tensor['mean'].shape[0]

    def random_features(self, k=1):
        random_indices = torch.randperm(self.n_fts)[:k]
        
        return self.features_by_index(random_indices)
        

    def features_by_metric(self, metric, stop=1, start=0):
        if metric not in METRIC_NAMES:
            raise ValueError(f"metric to sort features by must be one of {METRIC_NAMES}")
        
        if metric == 'max_activations':
            feature_data = self.stats_tensor[metric][0]
        else:
            feature_data = self.stats_tensor[metric]

        top_k_feature_indices = torch.topk(feature_data, stop).indices[start:]
        
        return self.features_by_index(top_k_feature_indices)

    def features_by_index(self, feature_indices):

        if isinstance(feature_indices, torch.Tensor):
            feature_indices = feature_indices.tolist()


        feature_data = []
        all_sample_indices = set()
        for feature_idx in feature_indices:
            feature_stats = {}

            for key, v in self.stats_tensor.items():
                if v.dim() == 1:
                    feature_stats[key] = v[feature_idx]
                elif v.dim() == 2:
                    feature_stats[key] = v[:, feature_idx]
                else:
                    raise ValueError(f"Unexpected number of dimensions for statistic {key}, {v.dim()}")

            sample_indices = self.stats_tensor['max_activation_indices'][:, feature_idx].to(torch.int).tolist()

            feature_data.append({
                'index': feature_idx,
                'stats': feature_stats,
                'max_samples': sample_indices
            })

            all_sample_indices.update(sample_indices)

        data_mapping = self.load_mapping(all_sample_indices)

        for f in feature_data:
            feature_activation_samples = []
            for sample_index in f['max_samples']:
                tokens, all_feature_activations = data_mapping[int(sample_index)]
                feature_specific_activations = all_feature_activations[:, f['index']].squeeze()
                feature_activation_samples.append((tokens, feature_specific_activations))
                
            f['max_samples'] = feature_activation_samples

        return feature_data

    def load_mapping(self, sample_indices):
        mapping = {}

        for sample_idx, sample_data in tqdm(self.ds.samples_for_indices(sample_indices), desc="Loading samples from disk", total=len(sample_indices)):

            attention_mask, tokens, activations = data_from_tensor(sample_data, self.n_fts)
            seq_len = int(torch.sum(attention_mask).item())
            tokens = tokens.squeeze()[:seq_len]
            activations = activations.squeeze()[:seq_len, :].squeeze()

            mapping[sample_idx] = (tokens, activations)

        
        return mapping



def glance_at(tokens, activations, tokenizer):
    word_tokens = tokenizer.convert_ids_to_tokens(tokens)
    word_tokens = [t.replace('Ġ', '') for t in word_tokens]
    norm_activations = normalize_activations(activations)

    if torch.isnan(norm_activations).any():
        return

    index_of_first_nonzero_activation = -1
    index_of_last_nonzero_activation = 0

    sections = []
    n_consecutive_zeros = 0
    context_len = 5
    for i, a in enumerate(norm_activations):
        if a > 0:
            index_of_last_nonzero_activation = i
            n_consecutive_zeros = 0

        if a > 0 and index_of_first_nonzero_activation == -1:
            index_of_first_nonzero_activation = i
        
        if n_consecutive_zeros > context_len and index_of_first_nonzero_activation != -1:
            first_index = max(0, index_of_first_nonzero_activation - context_len)
            last_index = min(len(word_tokens), index_of_last_nonzero_activation + context_len)

            sections.append((first_index, last_index))
            n_consecutive_zeros = 0
            index_of_first_nonzero_activation = -1
        
        n_consecutive_zeros += 1

    for first_index, last_index in sections:
        for token, activation in zip(word_tokens[first_index:last_index], norm_activations[first_index:last_index]):
            color = activation_to_color(activation)
            print(colored(token, color), end=' ')
        print()


def normalize_activations(activations):
    min_activation = torch.min(activations)
    max_activation = torch.max(activations)
    norm_activations = (activations - min_activation) / (max_activation - min_activation)
    return norm_activations

def activation_to_color(activation):
    if activation < 0.05:
        return 'grey'
    elif activation < 0.1:
        return 'green'
    elif activation < 0.2:
        return 'light_green'
    elif activation < 0.4:
        return 'red'
    elif activation < 0.6:
        return 'light_red'
    elif activation < 0.8:
        return 'yellow'
    else:
        return 'light_yellow'
