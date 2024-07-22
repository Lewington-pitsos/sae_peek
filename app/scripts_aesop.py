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
    # {
    #     'name': 'aesop-all',
    #     'load_fn': load_aesop,
    #     'sequence_length': 768,
    #     'batch_size': 32,
    #     'batches_in_stats_batch': 4,
            # 'save_activations': False,

    # },

    # {
    #     'name': 'pile10k-all',
    #     'load_fn': load_pile10k,
    #     'sequence_length': 1024,
    #     'batch_size': 16,
    #     'batches_in_stats_batch': 4,
            # 'save_activations': False,

    # },

    # {
    #     'name': 'aesop-32',
    #     'load_fn': load_aesop,
    #     'sequence_length': 768,
    #     'batch_size': 32,
    #     'batches_in_stats_batch': 4,
    #     "indices": [14392, 17697, 22617,  9254, 11711,  6225, 24477, 17608, 19770,  3027,
    #     19834, 18819,  7839, 11849, 15881,  8926, 11656, 23550,  7402,  8382,
    #     15718, 22181, 10453, 15906,  5324, 22459, 23132, 11416, 13618, 21132,
    #     11561,  2070],
    #     'save_activations': True,
    # },


    {
        'name': 'aesop-32-mean',
        'load_fn': load_aesop,
        'sequence_length': 768,
        'batch_size': 32,
        'batches_in_stats_batch': 4,
        "indices": [18819, 19751, 22181,  7491,  9781,   740,  4448, 15539, 24329,  2697,
        12045,  9790, 24452, 18402,  6225,   558,  7839, 14392,  4682, 17810,
         5640,  1649, 18030, 15766, 17295, 11933,   193,  7016, 14856, 10843,
        20049,  2768],
        'save_activations': True,
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
        feature_indices=p.get("indices", None),
        device=device,
        save_activations=p.get("save_activations", True),
    )
