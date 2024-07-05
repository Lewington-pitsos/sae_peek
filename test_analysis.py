from peek import *
from constants import TEST_DATA_DIR
from transformers import GPT2Tokenizer
from termcolor import colored
import numpy as np

def normalize_activations(activations):
    min_activation = torch.min(activations)
    max_activation = torch.max(activations)
    norm_activations = (activations - min_activation) / (max_activation - min_activation)
    return norm_activations

def activation_to_color(activation):
    # Define color ranges based on normalized activation value
    if activation < 0.05:
        return 'grey'
    elif activation < 0.1:
        return 'green'
    elif activation < 0.2:
        return 'light_green'
    elif activation < 0.4:
        return 'red'
    elif activation < 0.6:
        return 'light_red'
    elif activation < 0.8:
        return 'yellow'
    else:
        return 'light_yellow'

def test_analysis():
    c = Corpus(TEST_DATA_DIR)

    max_features = c.top_k('max_activations', 4)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def display_activations(tokens, activations, tokenizer):
        word_tokens = tokenizer.convert_ids_to_tokens(tokens)
        word_tokens = [t.replace('Ġ', '') for t in word_tokens]
        norm_activations = normalize_activations(activations)

        # check if nan
        if torch.isnan(norm_activations).any():
            return
        
        for token, activation in zip(word_tokens, norm_activations):
            color = activation_to_color(activation)
            print(colored(token, color), end=' ')
        print()

    for f in max_features:
        print(f'Feature {f["index"]} --------------')
        print(f'Max Activations', f['stats']['max_activations'])
        for tokens, activations in f['max_samples']:
            display_activations(tokens, activations, tokenizer)


if __name__ == "__main__":
    test_analysis()