import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from sae_lens import HookedSAETransformer
from app.peek import sae_assessment

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # or else transformers will complain about tqdm starting a parallel process.

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    batch_size = 256
    sequence_length = 128
    sae_id = "blocks.10.hook_resid_pre"
    
    dataset = load_dataset('NeelNanda/pile-10k', split="train")
    dataset = dataset.select(range(270))


    def tokenize(batch):
        text_input = [b['text'] for b in batch]
        return model.tokenizer(text_input, padding='longest', truncation=True, max_length=sequence_length)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)


    sae_assessment(
        tokenizer= model.tokenizer,
        dataloader=dl,
        batch_size=batch_size,
        sae_model='gpt2-small-res-jb',
        sae_id=sae_id,
        transformer=model,
        activation_dir=f'data/pile10k-all-{sae_id}',
        output=f'cruft/pile10k-all-{sae_id}.json',
        samples_per_feature=20,
    )
