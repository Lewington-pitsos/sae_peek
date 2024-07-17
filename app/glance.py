import torch
from termcolor import colored
from tqdm import tqdm

from app.constants import *
from app.storage import ActivationDataset, data_from_tensor

class Corpus():
    def __init__(self, data_dir):
        self.ds = ActivationDataset(data_dir)

        self._stats_tensor = self.ds.load_stats()

    @property
    def n_fts(self):
        return len(self.feature_indices)

    @property
    def feature_indices(self):
        return self.stats['feature_indices'].to(torch.int).tolist()

    @property
    def stats(self):
        return self._stats_tensor

    def random_features(self, k=1):
        random_indices = torch.randperm(self.n_fts)[:k]
        
        return self.by_relative_idx(random_indices)

    def features_by_metric(self, metric, stop=1, start=0):
        if metric not in METRIC_NAMES:
            raise ValueError(f"metric to sort features by must be one of {METRIC_NAMES}")
        
        if metric == 'max_activations':
            feature_data = self._stats_tensor[metric][0]
        else:
            feature_data = self._stats_tensor[metric]

        top_k_feature_indices = torch.topk(feature_data, stop).indices[start:]
        
        return self.by_relative_idx(top_k_feature_indices)

    def all_features(self, samples_per_feature=None):
        return self.by_relative_idx(list(range(self.n_fts)), samples_per_feature=samples_per_feature)

    def by_relative_idx(self, relative_indices, samples_per_feature=None):
        if isinstance(relative_indices, torch.Tensor):
            relative_indices = relative_indices.tolist()

        feature_data = []
        all_sample_indices = set()
        for relative_idx in relative_indices:
            feature_stats = {}

            for key, v in self._stats_tensor.items():
                if v.dim() == 1:
                    feature_stats[key] = v[relative_idx]
                elif v.dim() == 2:
                    feature_stats[key] = v[:, relative_idx]
                else:
                    raise ValueError(f"Unexpected number of dimensions for statistic {key}, {v.dim()}")
            

            sample_indices = self._stats_tensor['max_activation_indices'][:, relative_idx].to(torch.int).tolist()

            if samples_per_feature is not None:
                sample_indices = sample_indices[:samples_per_feature]

            feature_idx = self.feature_indices[relative_idx]

            feature_data.append({
                'index': feature_idx,
                'relative_index': relative_idx,
                'stats': feature_stats,
                'samples': sample_indices
            })

            all_sample_indices.update(sample_indices)

        data_mapping = self.load_mapping(all_sample_indices)

        for f in feature_data:
            feature_activation_samples = []
            for sample_index in f['samples']:
                tokens, all_feature_activations = data_mapping[int(sample_index)]
                feature_specific_activations = all_feature_activations[:, :, f['relative_index']].squeeze()
                feature_activation_samples.append((tokens, feature_specific_activations))
                
            f['samples'] = feature_activation_samples

        return feature_data
    
    def load_all_activations(self):
        data = torch.tensor(self.ds.greedy_load_activations())
        _, _, activations = data_from_tensor(data, self.n_fts)

        return activations
    
    def load_all_data(self):
        data = torch.tensor(self.ds.greedy_load_activations())
        return data_from_tensor(data, self.n_fts)

    def load_mapping(self, sample_indices):
        mapping = {}

        for sample_idx, sample_data in tqdm(self.ds.samples_for_indices(sample_indices), desc="Loading samples from disk", total=len(sample_indices)):
            attention_mask, tokens, activations = data_from_tensor(sample_data, self.n_fts)
            seq_len = int(torch.sum(attention_mask).item())
            tokens = tokens.squeeze()[:seq_len]
            activations = activations[:seq_len, :]

            mapping[sample_idx] = (tokens, activations)

        
        return mapping

def active_sections_across_samples(samples, tokenizer):
    active_sections = []
    for tokens, activations in samples:
        sections = get_active_sections(tokens, activations, tokenizer)
        if sections is not None:
            active_sections.extend(sections)

    return active_sections

def get_active_sections(tokens, activations, tokenizer):
    word_tokens = tokenizer.convert_ids_to_tokens(tokens)
    word_tokens = [t.replace('Ġ', '') for t in word_tokens]
    norm_activations = normalize_activations(activations)

    if torch.max(activations) == 0:
        print('Sample did not activate for feature, skipping sample...')
        return None

    if torch.isnan(norm_activations).any():
        raise ValueError(f"NaNs found in activations {norm_activations}, this should not occur.")

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
        
        if n_consecutive_zeros >= context_len and index_of_first_nonzero_activation != -1:
            first_index = max(0, index_of_first_nonzero_activation - context_len)
            last_index = min(len(word_tokens), index_of_last_nonzero_activation + context_len)

            sections.append((first_index, last_index))
            n_consecutive_zeros = 0
            index_of_first_nonzero_activation = -1

        if a == 0:
            n_consecutive_zeros += 1
        
    if index_of_first_nonzero_activation != -1:
        first_index = max(0, index_of_first_nonzero_activation - context_len)
        last_index = min(len(word_tokens), index_of_last_nonzero_activation + context_len)

        sections.append((first_index, last_index))

    token_sections = []
    for first_index, last_index in sections:
        token_sections.append((word_tokens[first_index:last_index], norm_activations[first_index:last_index]))

    return token_sections

def glance_at(tokens, activations, tokenizer):
    sections = get_active_sections(tokens, activations, tokenizer)

    for word_tokens, norm_activations in sections:
        for token, activation in zip(word_tokens, norm_activations):
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
