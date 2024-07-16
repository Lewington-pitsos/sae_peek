from transformers import GPT2Tokenizer
from app.scripts import *
import pytest

def test_loads_pile10k():
    pytest.skip("Test takes too long")
    batch_size = 32
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dl = load_pile10k(tokenizer, 256, batch_size, num_samples=128)

    n = next(iter(dl))

    print(n)

    assert n['input_ids'].shape[0] == batch_size
    assert n['attention_mask'].shape[0] == batch_size

    # assert no other keys in n
    assert len(n.keys()) == 2