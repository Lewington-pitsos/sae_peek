from transformers import GPT2Tokenizer
from app.scripts import load_aesop, load_pile10k
import pytest

# tokenizer fixture
@pytest.fixture(scope='module')
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok

def test_loads_pile10k(tokenizer):
    pytest.skip("Test takes too long")
    batch_size = 32

    dl = load_pile10k(tokenizer, 256, batch_size, num_samples=128)

    n = next(iter(dl))

    print(n)

    assert n['input_ids'].shape[0] == batch_size
    assert n['attention_mask'].shape[0] == batch_size
    assert n['input_ids'].shape[1] <= sequence_length
    assert n['attention_mask'].shape[1] <= sequence_length
    assert len(n.keys()) == 2

def test_loads_aesop(tokenizer):
    batch_size = 32
    sequence_length = 1024
    dl = load_aesop(tokenizer, batch_size, sequence_length)

    n = next(iter(dl))

    print(n)

    assert n['input_ids'].shape[0] == batch_size
    assert n['attention_mask'].shape[0] == batch_size
    assert n['input_ids'].shape[1] <= sequence_length
    assert n['attention_mask'].shape[1] <= sequence_length
    assert len(n.keys()) == 2

