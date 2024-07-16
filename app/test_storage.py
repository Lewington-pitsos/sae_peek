import os
import shutil
import torch
import numpy as np
import h5py
import pytest
from app.storage import ActivationDataset, DATA_KEY

@pytest.fixture
def dataset_fx():
    data_dir = 'data/test_data'
    max_samples_in_memory = 512
    dataset = ActivationDataset(data_dir, max_samples_in_memory)
    yield dataset
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

def test_add_and_save_activations(dataset_fx):
    activations = torch.randn(300, 64, 128)
    dataset_fx.add(activations)
    
    assert dataset_fx.activations.shape[0] == 300
    
    dataset_fx.save_activations()
    
    with h5py.File(dataset_fx.h5_name, 'r') as f:
        saved_data = f[DATA_KEY][:]
    
    np.testing.assert_array_almost_equal(saved_data, activations.numpy())
    assert dataset_fx.activations.shape[0] == 0

def test_add_activations_over_memory_limit(dataset_fx):
    activations = torch.randn(600, 64, 128)
    dataset_fx.add(activations)
    
    assert dataset_fx.activations.shape[0] == 88
    
    with h5py.File(dataset_fx.h5_name, 'r') as f:
        saved_data = f[DATA_KEY][:]
    
    np.testing.assert_array_almost_equal(saved_data, activations[:512].numpy())
    np.testing.assert_array_almost_equal(dataset_fx.activations.numpy(), activations[512:].numpy())

def test_samples_for_indices(dataset_fx):
    activations = torch.randn(600, 64, 128)
    dataset_fx.add(activations)
    dataset_fx.save_activations()
    dataset_fx.add(activations)
    dataset_fx.save_activations()

    indices = [100, 200, 300, 400, 500]
    retrieved_samples = {idx: sample for idx, sample in dataset_fx.samples_for_indices(indices)}
    
    for idx in indices:
        with h5py.File(dataset_fx.h5_name, 'r') as f:
            expected_sample = f[DATA_KEY][idx]
        np.testing.assert_array_almost_equal(retrieved_samples[idx].squeeze().numpy(), expected_sample)
