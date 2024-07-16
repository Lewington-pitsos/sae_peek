import torch
from transformers import GPT2Tokenizer

from app.assess import feature_representation
import pytest

# create a fixure which contains a gpt2 tokenizer
@pytest.fixture
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def test_feature_representation(tokenizer):
    story = "The quick brown fox jumps over the lazy dog, and the dog barks at the fox. This is only the beginning of the story. As the fox runs away, the dog chases it. The fox is too quick for the dog, and the dog gives up. The fox is safe."
    story_tokens = tokenizer(story, return_tensors='pt')['input_ids'].squeeze()
    
    activations = [0.0] * story_tokens.shape[0]
    activations[2] = 0.1
    activations[3] = 0.2
    activations[4] = 0.3
    activations[18] = 0.4
    activations[19] = 0.1
    activations[20] = 0.1

    activations = torch.tensor(activations)


    story2 = "for many years there lived in the wild western woods a badger. He was a very bad badger, and he did not like anyone. He was always grumpy and never smiled. He was always looking for trouble, and he always found it. He was a very bad badger."
    story_tokens2 = tokenizer(story2, return_tensors='pt')['input_ids'].squeeze()

    activations2 = [0.0] * story_tokens2.shape[0]
    activations2[18] = 0.1
    activations2[19] = 0.1
    activations2[22] = 0.1
    activations2[23] = 0.1
    activations2[25] = 0.9
    activations2[29] = 0.3

    activations2 = torch.tensor(activations2)
    
    feature = {
        'samples': [
            (story_tokens, activations),
            (story_tokens2, activations2)
        ]
    }

    rep, _ = feature_representation(feature, tokenizer)

    assert len(rep) > 0
    assert len(rep) > 100
    assert len(rep) < 2000


