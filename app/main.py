from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
import tqdm

from app.peek import *
from app.constants import *

def get_features(sae, transformer, input_ids, attention_mask):
    _, cache = transformer.run_with_cache(
        input_ids, 
        attention_mask=attention_mask, 
        prepend_bos=True, 
        stop_at_layer=sae.cfg.hook_layer + 1)

    hidden_states = cache[sae.cfg.hook_name]

    features = sae.encode(hidden_states)

    return features

def peek(sae, transformer, corpus, n_features, topk, batch_size, device, output_dir):
    ds = ActivationDataset(output_dir)
    stats = {
        'mean': torch.zeros(n_features),
        'nonzero_proportion': torch.zeros(n_features),
        'max_activations': torch.zeros(topk, n_features).to(device),
        'max_activation_indices': torch.zeros(topk, n_features).to(device),
    }

    with torch.no_grad():
        for i, (input_ids, att_mask) in enumerate(tqdm.tqdm(corpus)):
            batch_size = input_ids.shape[0]
            features = get_features(sae, transformer, input_ids, att_mask)

            features = torch.cat([features, att_mask.unsqueeze(-1), input_ids.unsqueeze(-1)], dim=-1).to('cpu')

            ds.add(features)

            collect_feature_stats(i*batch_size, n_features, features, stats, topk)

    ds.finalize(stats)


def test_example():
    sae = SAE.load_from_pretrained('checkpoints/6twmlrfz/final_245760000').to('cpu')
    model = HookedTransformer.from_pretrained(
        "tiny-stories-1L-21M"
    ).to('cpu')

    dataset = load_dataset("imdb")
    train_subset = dataset['train'].take(10000)

    def tokenize(x):
        output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    
        return output['input_ids'], output['attention_mask']

    batch_size = 32
    dl = DataLoader(train_subset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    peek(
        sae,
        model, 
        dl, 
        n_features=16384, 
        topk=3, 
        batch_size=batch_size, 
        device='cpu', 
        output_dir=TEST_DATA_DIR
    )


if __name__ == '__main__':
    test_example()


    