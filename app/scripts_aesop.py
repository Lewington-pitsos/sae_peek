from transformers import GPT2Tokenizer
from sae_lens import HookedSAETransformer
from app.peek import generate_sae_activations
from app.scripts import load_aesop, load_pile10k
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_size = 32
sequence_length = 768

model = HookedSAETransformer.from_pretrained("gpt2", device=device)
sae_id = "blocks.10.hook_resid_pre"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

params = [
    {
        'name': 'aesop-all',
        'load_fn': load_aesop,
        'sequence_length': 768
    },
    {
        'name': 'pile10k-all',
        'load_fn': load_pile10k,
        'sequence_length': 2048
    }
]

for param in params:
    aesop = params['load_fn'](tokenizer, batch_size, sequence_length)

    activation_dir = f'data/{param["name"]}'

    generate_sae_activations(
        dataloader=aesop,
        sae_model='gpt2-small-res-jb',
        sae_id=sae_id,
        transformer=model,
        batch_size=batch_size,
        batches_in_stats_batch=8,
        activation_dir=activation_dir,
        feature_indices=None,
        device=device
    )
