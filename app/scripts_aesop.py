from transformers import GPT2Tokenizer
from sae_lens import HookedSAETransformer
from app.peek import generate_sae_activations
from app.scripts import load_aesop
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_size = 16
sequence_length = 768

model = HookedSAETransformer.from_pretrained("gpt2", device=device)
sae_id = "blocks.10.hook_resid_pre"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

aesop = load_aesop(tokenizer, batch_size, sequence_length)

activation_dir = 'data/aesop'
generate_sae_activations(
    dataloader=aesop,
    sae_model='gpt2-small-res-jb',
    sae_id=sae_id,
    transformer=model,
    batch_size=batch_size,
    batches_in_stats_batch=2,
    activation_dir=activation_dir,
    feature_indices=list(range(128)),
    device=device
)

# llm_assessment(activation_dir, output='cruft/aesop-llm.json', samples_per_feature=10)
