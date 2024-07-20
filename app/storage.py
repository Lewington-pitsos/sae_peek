import torch
import os
import h5py
from abc import ABC

def data_from_tensor(tensor, n_ft):
    assert tensor.dim() == 3, f"Expected tensor to have 3 dimensions, got {tensor.shape}"
    assert tensor.shape[2] == n_ft + 2, f"Expected tensor to have {n_ft + 2} features, got {tensor.shape[2]}"

    attention_mask = tensor[:, :, n_ft: n_ft + 1]  # (bs, seq_len, 1)
    tokens = tensor[:, :, n_ft + 1:]  # (bs, seq_len, 1)
    activations = tensor[:, :, :n_ft]  # (bs, seq_len, n_ft)

    return attention_mask, tokens, activations

DATA_KEY = 'data'

class _StatSaver():
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def add(self, activations):
        raise NotImplementedError

    def finalize(self, stats):
        torch.save(stats, os.path.join(self.data_dir, 'stats.pt'))    

class StatDataset(_StatSaver):
    def add(self, activations):
        pass

class ActivationDataset(_StatSaver):
    def __init__(self, data_dir, max_samples_in_memory=512):
        super().__init__(data_dir)
        self.activations = torch.tensor([])
        self.max_samples_in_memory = max_samples_in_memory
        self.total_samples_saved = 0  # To keep track of the total number of samples saved

        self.h5_name = os.path.join(self.data_dir, f"{DATA_KEY}.h5")
        self._dataset_initialized = os.path.exists(self.h5_name)

    def _make_dataset(self, n_fts, seq_len):
        if not os.path.exists(self.h5_name):
            with h5py.File(self.h5_name, 'w') as f:
                f.create_dataset(DATA_KEY, shape=(0, 0, 0), maxshape=(None, n_fts, seq_len), dtype='float32')
            self._dataset_initialized = True
   
    def add(self, activations):
        if not self._dataset_initialized:
            self._make_dataset(activations.shape[1], activations.shape[2])

        current_size = self.activations.shape[0]
        if current_size + activations.shape[0] > self.max_samples_in_memory:
            put_in_current_tensor = self.max_samples_in_memory - current_size
            self.activations = torch.cat([self.activations, activations[:put_in_current_tensor]], dim=0)
            self.save_activations()
            self.activations = activations[put_in_current_tensor:]
        else:
            self.activations = torch.cat([self.activations, activations], dim=0)   

    def save_activations(self):
        if self.activations.shape[0] == 0:
            return
        
        with h5py.File(self.h5_name, 'a') as f:
            dataset = f[DATA_KEY]
            new_shape = (self.total_samples_saved + self.activations.shape[0], *self.activations.shape[1:])
            dataset.resize(new_shape)
            dataset[self.total_samples_saved:] = self.activations.numpy()
        
        self.total_samples_saved += self.activations.shape[0]
        self.activations = torch.tensor([])

    def samples_for_indices(self, indices):
        indices = sorted(indices)

        with h5py.File(self.h5_name, 'r') as f:
            dataset = f[DATA_KEY]
            for idx in indices:
                sample_data = torch.tensor(dataset[idx, :]).unsqueeze(0)
                yield idx, sample_data

    def greedy_load_activations(self):
        with h5py.File(self.h5_name, 'r') as h5file:
            dataset = h5file[DATA_KEY][:]
        return dataset
    
    def finalize(self, stats):
        super().finalize(stats)
        self.save_activations()
    
    def load_stats(self, device='cpu'):
        return torch.load(os.path.join(self.data_dir, 'stats.pt'), map_location=device)
