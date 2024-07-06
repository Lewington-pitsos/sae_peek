import torch
from transformers import GPT2Tokenizer

from app.constants import *
from app.glance import Corpus, glance_at

def analysis():
    c = Corpus('data/gpt2')

    max_features = c.features_by_metric('max_activations', start=1000, stop=1005)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for f in max_features:
        print(f'Feature {f["index"]} --------------')
        print(f'Max Activations', f['stats']['max_activations'])
        for tokens, activations in f['samples']:
            glance_at(tokens, activations, tokenizer)

def dual_analysis():
    pos = Corpus('data/gpt2-imdb-pos')
    neg = Corpus('data/gpt2-imdb-neg')

    pos_nonzero_prop = pos.stats_tensor['max_activations'][0]
    neg_nonzero_prop = neg.stats_tensor['max_activations'][0]

    delta_nonzero_prop = (pos_nonzero_prop - neg_nonzero_prop)

    most_pos_values, most_pos_features = torch.topk(delta_nonzero_prop, k=2)

    print(most_pos_values)
    print(most_pos_features)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    samples_per_feature=3
    pos_samples = pos.features_by_idx(most_pos_features, samples_per_feature)
    neg_samples = neg.features_by_idx(most_pos_features, samples_per_feature)

    for i in range(len(neg_samples)):
        pfeat = pos_samples[i]
        nfeat = neg_samples[i]
        print(f'Feature {pfeat["index"]} -----------------------------')


        print('Positive Samples -------')
        print(f'Max Activations', pfeat['stats']['max_activations'])
        for tokens, activations in pfeat['samples']:
            glance_at(tokens, activations, tokenizer)


        print('Negative Samples -------')
        print(f'Max Activations', nfeat['stats']['max_activations'])
        for tokens, activations in nfeat['samples']:
            glance_at(tokens, activations, tokenizer)


if __name__ == "__main__":
    dual_analysis()