from transformers import GPT2Tokenizer
from sae_lens import HookedSAETransformer
from app.peek import generate_sae_activations
from app.scripts import load_aesop, load_pile10k
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = HookedSAETransformer.from_pretrained("gpt2", device=device)
sae_id = "blocks.10.hook_resid_pre"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

params = [
    {
        'name': 'aesop-all',
        'load_fn': load_aesop,
        'sequence_length': 768,
        'batch_size': 32,
        'batches_in_stats_batch': 4
    },

    {
        'name': 'pile10k-all',
        'load_fn': load_pile10k,
        'sequence_length': 1024,
        'batch_size': 16,
        'batches_in_stats_batch': 4
    },
]

for p in params:
    aesop = p['load_fn'](tokenizer, int(p['batch_size'] / 8), p['sequence_length'])

    activation_dir = f'data/{p["name"]}'

    generate_sae_activations(
        dataloader=aesop,
        sae_model='gpt2-small-res-jb',
        sae_id=sae_id,
        transformer=model,
        batches_in_stats_batch=p['batches_in_stats_batch'],
        activation_dir=activation_dir,
        feature_indices=list(range(8)),
        device=device,
        save_activations=False
    )
