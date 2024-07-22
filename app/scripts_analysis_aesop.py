import sys
import os
from app.glance import Corpus, glance_at
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
import torch
import json
from app.assess import llm_assessment
from app.peek import generate_sae_activations
from app.scripts import load_aesop, load_pile10k
from sae_lens import HookedSAETransformer
import torch.nn.functional as F

def stretch_activations(activations, attention_mask):
    n_samples, _, n_features = activations.shape
    max_len = attention_mask.sum(dim=1).max().item()  # Get the maximum non-padded length

    stretched_activations = torch.zeros((n_samples, max_len, n_features), device=activations.device)
    
    for i in range(n_samples):
        non_padded_len = attention_mask[i].sum().item()  # Find the actual length of each sequence
        if non_padded_len == 0:
            continue

        seq = activations[i, :non_padded_len].unsqueeze(0).permute(0, 2, 1)  # Shape: 1 x n_features x non_padded_len
        stretched_seq = F.interpolate(seq, size=(max_len,), mode='linear', align_corners=False)

        stretched_activations[i] = stretched_seq.squeeze(0).permute(1, 0)  # Shape: max_len x n_features
    
    return stretched_activations




AESOP_BATCH_SEQ_LEN = 768
AESOP_MAX_SEQ_LEN = 514

# ----------------- Define Parameters ---------------------

BASE_NAME = 'test'
TRANSFORMER_NAME = 'gpt2'
SAE_MODEL = 'gpt2-small-res-jb'
SAE_ID = "blocks.10.hook_resid_pre"
TOPK = 4
FEATURE_INDICES = list(range(8))
TOPK_DATA_DIR = f'data/{BASE_NAME}-top32'
ASSESSMENT_FILE = f'cruft/pile10k-{BASE_NAME}-mean.json'



TOPK_BATCH_SIZE = 32


PARAMS_AESOP={
        'name': f'data/aesop-{BASE_NAME}',
        'load_fn': load_aesop,
        'sequence_length': AESOP_BATCH_SEQ_LEN,
        'batch_size': 32,
        'num_samples': None,
        'indices': FEATURE_INDICES,
        'batches_in_stats_batch': 4,
        'save_activations': False,

    }
PARAMS_PILE10K ={
        'name': f'data/pile10k-{BASE_NAME}',
        'load_fn': load_pile10k,
        'num_samples': 64,
        'sequence_length': 1024,
        'batch_size': 16,
        'indices': FEATURE_INDICES,
        'batches_in_stats_batch': 4,
        'save_activations': False,
    }


# --------------------------- Optional Cleanup -----------------------------

# check if the "clenaup" argument passed as the first command line argument
if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
    print('cleaning up ...')
    for param in [PARAMS_AESOP, PARAMS_PILE10K]:
        if os.path.exists(param['name']):
            os.system(f'rm -r {param["name"]}')
    if os.path.exists(TOPK_DATA_DIR):
        os.system(f'rm -r {TOPK_DATA_DIR}')
    if os.path.exists(ASSESSMENT_FILE):
        os.system(f'rm {ASSESSMENT_FILE}')
        
    sys.exit(0)



# --------------------------- Load Model -----------------------------
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if not os.path.exists(PARAMS_AESOP['name']) or not os.path.exists(PARAMS_PILE10K['name']) or not os.path.exists(TOPK_DATA_DIR):
    model = HookedSAETransformer.from_pretrained(TRANSFORMER_NAME, device=device)
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------- Generate Statistics for all features ---------------------

for param in [PARAMS_AESOP, PARAMS_PILE10K]:
    if not os.path.exists(param['name'] + '/stats.pt'):
        data_loader = param['load_fn'](tokenizer, int(param['batch_size'] / 8), param['sequence_length'], num_samples=param['num_samples'])

        generate_sae_activations(
            dataloader=data_loader,
            sae_model=SAE_MODEL,
            sae_id=SAE_ID,
            transformer=model,
            activation_dir=param['name'],
            batches_in_stats_batch=param['batches_in_stats_batch'],
            feature_indices=param.get("indices", None),
            device=device,
            save_activations=param.get("save_activations", True),
        )

# --------------------------- Load statistics for all features for both datasets, get topk ---------------------

aesop_corpus = Corpus(PARAMS_AESOP['name'])
pile_corpus = Corpus(PARAMS_PILE10K['name'])

stat = "mean"
aesop_stat = aesop_corpus.stats[stat]
pile_stat = pile_corpus.stats[stat]
# delta = (c.stats['max_activations'][0] - pile.stats['max_activations'][0] + 1e-7) / (c.stats['max_activations'][0] + 1e-7) * (c.stats['max_activations'][0] != 0 * 1)
aesop_pile_delta = (aesop_stat - pile_stat + 1e-7) / (aesop_stat + 1e-7) * (aesop_stat != 0 * 1)

topk_values, topk_indices = torch.topk(aesop_pile_delta, TOPK)

print(f'top {TOPK} feature indices', topk_values.tolist())
print(f'top {TOPK} indices', topk_indices.tolist())

# --------------------------- Generate and load sample Activations for all topk ---------------------


if not os.path.exists(TOPK_DATA_DIR):
    generate_sae_activations(
        dataloader=load_aesop(tokenizer=tokenizer, batch_size=TOPK_BATCH_SIZE, sequence_length=AESOP_BATCH_SEQ_LEN),
        sae_model=SAE_MODEL,
        sae_id=SAE_ID,
        transformer=model,
        batches_in_stats_batch=16,
        activation_dir=TOPK_DATA_DIR,
        feature_indices=topk_indices.tolist(),
        device=device,
        save_activations=True,
    )

    del model

aesop_topk_corpus = Corpus(TOPK_DATA_DIR)
print(aesop_topk_corpus.feature_indices)
attention_mask, _, acts = aesop_topk_corpus.load_all_data()

# --------------------------- Normalize and plot activations for topk features ---------------------

acts = acts * attention_mask
acts = acts[:, :AESOP_MAX_SEQ_LEN, :]
attention_mask = attention_mask[:, :AESOP_MAX_SEQ_LEN, :]
normalized_acts = acts / (torch.amax(acts, dim=(0, 1)) + 1e-8)

plt.imshow(torch.max(normalized_acts, dim=0)[0].T, cmap='gray', aspect='auto', interpolation="nearest")
plt.savefig(f'notes/{BASE_NAME}-{TOPK}-mean.png')

s = stretch_activations(normalized_acts, attention_mask.to(torch.int))
plt.imshow(torch.max(s, dim=0)[0].T, cmap='gray', aspect='auto', interpolation="nearest")
plt.savefig(f'notes/{BASE_NAME}-{TOPK}-mean-stretched.png')

# --------------------------- Create and load LLM explanations for topk features ---------------------

if not os.path.exists(ASSESSMENT_FILE):
    llm_assessment(
        tokenizer,
        PARAMS_PILE10K['name'], 
        output=ASSESSMENT_FILE, 
        samples_per_feature=10, 
        relative_feature_indices=topk_indices.tolist()
    )
with open(ASSESSMENT_FILE) as f:
    pile10k_explanations = json.load(f)

# --------------------------- Plot topk features with explanations ---------------------

top_fts = []
explain = []
for i in topk_indices.tolist():
    for f in pile10k_explanations:
        print(f['feature'])
        if f['feature'] == i:
            if f['assessment'] is not None:
                top_fts.append(str(i) + " " + f['assessment']['feature_name'])
                explain.append(f['assessment']['feature_description'])
            else:
                top_fts.append("SKIPPED")
                explain.append("SKIPPED")

print(top_fts)
print(topk_indices)
assert len(top_fts) == topk_indices.shape[0]
toplot = torch.mean(s, dim=0).T

plt.figure(figsize=(14, 8))

plt.imshow(toplot, cmap='gray', aspect='auto', interpolation="nearest")
plt.yticks(ticks=range(len(top_fts)), labels=top_fts)

plt.xticks([])
plt.title("Features with highest mean activations in Aesop's Fables compared to NeelNanda/Pile-10k")
plt.text(0, -1, 'Fable start -->', ha='center', va='center', fontsize=10, color='black')
plt.text(toplot.shape[1] - 1, -1,  '--> Fable end', ha='center', va='center', fontsize=10, color='black')

plt.subplots_adjust(top=0.95, bottom=0.15)
plt.savefig(f'notes/{BASE_NAME}-{TOPK}-mean-stretched-features.png')

# --------------------------- Print feature activations and explanations ---------------------

top_indices = aesop_corpus.by_relative_idx(topk_indices)
for i, feature in enumerate(top_indices):
    print('---------', top_fts[i], '---------')
    print(explain[i])
    
    for tokens, token_acts in feature['samples']:
        try:
            glance_at(tokens, token_acts, tokenizer)
        except:
            continue