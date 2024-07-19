from transformers import GPT2Tokenizer
from sae_lens import HookedSAETransformer
from app.peek import generate_sae_activations
from app.scripts import load_clear
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

sequence_length = 384
batch_size = 4

model = HookedSAETransformer.from_pretrained("gpt2", device=device)
sae_id = "blocks.10.hook_resid_pre"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

clear = load_clear(tokenizer, batch_size, sequence_length)

activation_dir = 'data/clear'
generate_sae_activations(
    dataloader=clear,
    sae_model='gpt2-small-res-jb',
    sae_id=sae_id,
    transformer=model,
    batches_in_stats_batch=4,
    activation_dir=activation_dir,
    feature_indices=list(range(2048)),
    device=device
)
