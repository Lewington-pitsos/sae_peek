import torch
from tqdm import tqdm
import os
import h5py

def data_from_tensor(tensor, n_ft):
    assert tensor.shape[2] == n_ft + 2, f"Expected tensor to have {n_ft + 2} features, got {tensor.shape[2]}"

    attention_mask = tensor[:, :, n_ft: n_ft + 1] # (bs, seq_len, 1)
    tokens = tensor[:, :, n_ft + 1:] # (bs, seq_len, 1)
    activations = tensor[:, :, :n_ft] # (bs, seq_len, n_ft)

    return attention_mask, tokens, activations

class ActivationDataset():
    def __init__(self, data_dir, max_samples_in_memory=512):
        self.activations = torch.tensor([])
        self.data_dir = data_dir
        self.max_samples_in_memory = max_samples_in_memory

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self._n_tensors_saved = 0
    
    def add(self, activations):
        current_size = self.activations.shape[0]
        if current_size + activations.shape[0] > self.max_samples_in_memory:
            put_in_current_tensor = self.max_samples_in_memory - current_size
            self.activations = torch.cat([self.activations, activations[:put_in_current_tensor]], dim=0)
            self.save_activations()
            self.activations = activations[put_in_current_tensor:]
        else:
            self.activations = torch.cat([self.activations, activations], dim=0)   

    @property
    def h5_name(self):
        return os.path.join(self.data_dir, f"samples.h5")

    def save_activations(self):
        if self.activations.shape[0] == 0:
            return
        
        with h5py.File(self.h5_name, 'a') as f:
            f.create_dataset(f'sample_{self._n_tensors_saved}', data=self.activations.numpy())
        
        self._n_tensors_saved += 1
        self.activations = torch.tensor([])

    def samples_for_indices(self, indices):
        indices = sorted(indices, reverse=True)

        sample_idx = indices.pop()
        for offset, lazy_tensor_batch in self.lazy_load_activations():
            next_batch_start_idx = offset + lazy_tensor_batch.shape[0]
            while sample_idx < next_batch_start_idx:
                batch_idx = sample_idx - offset
                sample_data = torch.tensor(lazy_tensor_batch[batch_idx, :]).unsqueeze(0)

                yield sample_idx, sample_data

                if len(indices) > 0:
                    sample_idx = indices.pop()
                else:
                    return

    def lazy_load_activations(self):
        with h5py.File(self.h5_name, 'r') as h5file:
            if len(h5file.keys()) == 0:
                raise ValueError(f"No keys found in {self.h5_name}, it is empty.")


            running_offset = 0
            for key in h5file.keys():
                lazy_tensor = h5file[key]

                yield running_offset, lazy_tensor
                running_offset += lazy_tensor.shape[0]

    def finalize(self, stats):
        torch.save(stats, os.path.join(self.data_dir, 'stats.pt'))    
        
        self.save_activations()

    def load_stats(self, device='cpu'):
        return torch.load(os.path.join(self.data_dir, 'stats.pt'), map_location=device)
