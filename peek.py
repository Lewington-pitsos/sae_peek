import os
import torch

def data_from_tensor(tensor, n_ft):
    attention_mask = tensor[:, :, n_ft: n_ft + 1] # (bs, seq_len, 1)
    tokens = tensor[:, :, n_ft + 1:] # (bs, seq_len, 1)
    activations = tensor[:, :, :n_ft] # (bs, seq_len, n_ft)

    return attention_mask, tokens, activations

STATISTICS = [
    'mean',
    'nonzero_proportion',
    'max_activations',
]


class ActivationDataset():
    def __init__(self, data_dir, max_samples_in_memory=1000):
        self.activations = torch.tensor([])
        self.data_dir = data_dir
        self.max_samples_in_memory = max_samples_in_memory

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.start_idx = 0
    
    def add(self, activations):
        current_size = self.activations.shape[0]
        if current_size + activations.shape[0] > self.max_samples_in_memory:
            self.save_activations()
            self.activations = activations
        else:
            self.activations = torch.cat([self.activations, activations], dim=0)   

    def save_activations(self):
        if self.activations.shape[0] == 0:
            return

        filename = os.path.join(self.data_dir, f"sample_{self.start_idx}.pt")
        torch.save(self.activations, filename)
        self.start_idx += self.activations.shape[0]
        self.activations = torch.tensor([])

    def load_activations(self):
        all_tensor_files = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir) if "sample_" in x]
        if len(all_tensor_files) == 0:
            raise ValueError(f"No tensor files found in {self.data_dir}")
        
        all_tensor_files = sorted(all_tensor_files, key=lambda x: int(x.split('sample_')[1].split('.')[0]))

        running_offset = 0
        for tensor_file in all_tensor_files:
            tensor = torch.load(tensor_file)
            running_offset += tensor.shape[0]
            yield running_offset, tensor

    def finalize(self, stats):
        torch.save(stats, os.path.join(self.data_dir, 'stats.pt'))    
        
        self.save_activations()

    def load_stats(self):
        return torch.load(os.path.join(self.data_dir, 'stats.pt'))

class Corpus():
    def __init__(self, data_dir):
        self.ds = ActivationDataset(data_dir)

        self.stats_tensor = self.ds.load_stats()
        self.n_fts = self.stats_tensor['mean'].shape[0]

    def top_k(self, stat, k=1):
        if stat not in STATISTICS:
            raise ValueError(f"stat must be one of {STATISTICS}")
        
        if stat == 'max_activations':
            feature_data = self.stats_tensor[stat][0]
        else:
            feature_data = self.stats_tensor[stat]

        top_k_features = torch.topk(feature_data, k).indices

        feature_data = []
        all_sample_indices = set()
        for feature_idx in top_k_features:
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
        sample_indices = sorted(list(sample_indices))

        mapping = {}
        idx_idx = 0

        for offset, tensor_batch in self.ds.load_activations():
            feature_idx = int(sample_indices[idx_idx])
            while feature_idx < offset + tensor_batch.shape[0]:
                batch_idx = int(feature_idx - offset)

                tensor = tensor_batch[batch_idx, :].unsqueeze(0)
                attention_mask, tokens, activations = data_from_tensor(tensor, self.n_fts)
                seq_len = int(torch.sum(attention_mask).item())
                tokens = tokens.squeeze()[:seq_len]
                activations = activations.squeeze()[:seq_len, :].squeeze()


                mapping[feature_idx] = (tokens, activations)

                idx_idx += 1

                if idx_idx >= len(sample_indices):
                    break

                feature_idx = int(sample_indices[idx_idx])

            if idx_idx >= len(sample_indices):
                break
        
        return mapping
                            
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

    batch_max = torch.max(masked_activations, dim=1).values # (bs, n_ft)
    n_elements = torch.sum(attention_mask) # (1)


    batch_mean = torch.sum(masked_activations, dim=(0, 1)) / n_elements # (n_ft)
    batch_nonzero_prop = torch.sum(masked_activations > 0, dim=(0, 1)) / n_elements # (n_ft)


    stats['mean'].add_(batch_mean)
    stats['nonzero_proportion'].add_(batch_nonzero_prop)

    stats['max_activations'], stats['max_activation_indices'] = new_topk_samples(start_idx, masked_activations, stats['max_activations'], stats['max_activation_indices'], topk)


