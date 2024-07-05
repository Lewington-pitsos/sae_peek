from transformers import GPT2Tokenizer

from app.constants import *
from app.glance import Corpus, glance_at

def test_analysis():
    c = Corpus('data/gpt2')

    max_features = c.features_by_metric('max_activations', start=1000, stop=1005)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for f in max_features:
        print(f'Feature {f["index"]} --------------')
        print(f'Max Activations', f['stats']['max_activations'])
        for tokens, activations in f['max_samples']:
            glance_at(tokens, activations, tokenizer)


if __name__ == "__main__":
    test_analysis()