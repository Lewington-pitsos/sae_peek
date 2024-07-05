import os
from sae_lens import TrainingSAE, SAE
from matplotlib import pyplot as plt
import torch
import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate
import time
from transformer_lens import HookedTransformer
import numpy as np

class Feature():
    def __init__(self, index, keep_samples=35):
        self.index = index
        self.samples = []
        self.keep_samples = keep_samples
        self.total_activation = 0
        self.mean_activation = 0
        self.max = 0
        self.frac_nonzero = 0
        self.total_samples = 0
    
    def add_sample(self, tokens, activations):
        sum_act = torch.sum(activations).item()
        self.total_samples += 1
        self.total_activation += sum_act
        self.mean_activation = self.total_activation / self.total_samples
        self.frac_nonzero += (np.sum(activations > 0).item() / activations.shape[0])

        sample_max = max(self.max, np.max(activations).item())
        self.max = sample_max
        
        if len(self.samples) < self.keep_samples:
            self.samples.append((tokens, activations, sample_max))
            self.samples = sorted(self.samples, key=lambda x: x[2], reverse=True)
        else:
            if sum_act > self.samples[-1][2]:
                self.samples.pop()
                self.samples.append((tokens, activations, sample_max))
                self.samples = sorted(self.samples, key=lambda x: x[2], reverse=True)

    def __str__(self):
        return f"Feature: {self.index}"

    def __repr__(self):
        return self.__str__()

def save_tensor(tensor, id):
    id_dir = f'data/{id}'

    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    n_existing_tensors = len(os.listdir(id_dir))

    torch.save(tensor, f'{id_dir}/tensor-{n_existing_tensors}.pt')


def save_all_features(corpus, sae_model, task_id=None):
    if task_id is None:
        task_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print('Saving features under', task_id)


    start = time.time()

    max_samples_in_memory = 500
    n_saves = 0
    all_features = torch.tensor([])

    for batch in corpus:
        batch_size = batch['input_ids'].shape[0]
        att_mask = batch['attention_mask']
        input_ids = batch['input_ids']
        features = sae_model.forward(input_ids, att_mask)

        # concatenate attention mask onto the end of the features, att mask is (batch_size, seq_len), features is (batch_size, seq_len, n_fts)

        features = torch.cat([features, att_mask.unsqueeze(-1), input_ids.unsqueeze(-1)], dim=-1).to('cpu')

        all_features = torch.cat([all_features, features], dim=0)

        if all_features.shape[0] + batch_size  > max_samples_in_memory:
            save_tensor(all_features, task_id)
            all_features = torch.tensor([])
            n_saves += 1

    save_tensor(all_features, task_id)

    end_compute = time.time()
    print('Computed features in', round(end_compute - start, 2))
    print('Saved all features under', task_id)

class SaeAnalyser():
    def __init__(self, sae, model):
        self.sae = sae
        self.model = model
    
    def forward(self, input_ids, attention_mask):

        _, cache = self.model.run_with_cache(
            input_ids, 
            attention_mask=attention_mask, 
            prepend_bos=True, 
            stop_at_layer=self.sae.cfg.hook_layer + 1)

        hidden_states = cache[self.sae.cfg.hook_name]

        features = self.sae.encode(hidden_states)

        return features

sae = SAE.load_from_pretrained('checkpoints/6twmlrfz/final_245760000')


model = HookedTransformer.from_pretrained(
    "tiny-stories-1L-21M"
).to('cpu')  # This will wrap huggingface models and has lots of nice utilities.

dataset = load_dataset("imdb")
subsets = {}
for label in set(dataset['train']['label']):  # Assuming you are working with the 'train' split
    subsets[label] = dataset['train'].filter(lambda example: example['label'] == label)

sae_analyser = SaeAnalyser(sae, model)

def tokenize(x):
    return model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)

feature_dict = {}
for key, subset in subsets.items():
    subset = subset.take(100)
    dl = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=tokenize)
    feature_dict[key] = save_all_features(dl, sae_analyser)


def update_tokens(current, current_maxes, tokens, maxes):

    highest_max_mask = maxes == maxes.max(dim=0).values
    filtered_maxes = highest_max_mask * maxes
    htm = (filtered_maxes > current_maxes) * 1

    max_tokens = tokens.T @ htm

    mask = torch.max(htm, dim=0).values
    inverted_mask = 1 - mask
    expanded_mask = inverted_mask.expand_as(current)


    masked_current = current * expanded_mask
    current = masked_current + max_tokens

    return current

def collect_feature_stats(n_ft, activations, stats):

    # activations is (bs, seq_len, n_ft + 2)
    attention_mask = activations[:, n_ft: n_ft + 1] # (bs, seq_len, 1)
    tokens = activations[:, n_ft + 1:] # (bs, seq_len, 1)
    activations = activations[:, :n_ft] # (bs, seq_len, n_ft)

    masked_activations = activations * attention_mask

    n_elements = torch.sum(attention_mask, dim=1) # (bs, 1)
    batch_max = torch.max(masked_activations, dim=1) # (bs, n_ft)

    batch_mean = torch.sum(masked_activations, dim=1) / n_elements # (bs, n_ft)
    batch_nonzero_prop = torch.sum(masked_activations > 0, dim=1) / n_elements # (bs, n_ft)
    batch_1e_neg5_prop = torch.sum(masked_activations > 1e-5, dim=1) / n_elements # (bs, n_ft)
    batch_1e_neg4_prop = torch.sum(masked_activations > 1e-4, dim=1) / n_elements # (bs, n_ft)


    stats['mean'] += batch_mean
    stats['max'] += batch_max
    stats['nonzero_prop'] += batch_nonzero_prop
    stats['1eneg5_prop'] += batch_1e_neg5_prop 
    stats['1eneg4_prop'] += batch_1e_neg4_prop 
    stats['max_tokens'] = update_tokens(stats['max_tokens'], stats['max'], tokens, batch_max)



