import torch
from torch.utils.data import DataLoader, Dataset
from sae_lens import HookedSAETransformer
import pandas as pd
from transformers import GPT2Tokenizer 
import transformers

from app.peek import generate_sae_activations

def load_multi_dataset(batch_size):
    df = pd.read_csv('data/us_2020_election_speeches.csv')
    biden = df[df['speaker'] == 'Joe Biden']
    trump = df[df['speaker'] == 'Donald Trump']

    print('Biden:', biden.shape[0])
    print('Trump:', trump.shape[0])

    biden_texts = biden['text'].tolist()
    trump_texts = trump['text'].tolist()

    transformers.logging.set_verbosity_error()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def tokenize_and_segment(texts, tokenizer, max_length=256, overlap=5):
        segments = []
        for text in texts:
            tokens = tokenizer(text)['input_ids']
            for i in range(0, len(tokens), max_length - overlap):
                segment = tokens[i:i + max_length]
                attention_mask = [1] * len(segment)
                if len(segment) < max_length:
                    padding = [0] * (max_length - len(segment))
                    segment += padding
                    attention_mask += padding
                segments.append({"input_ids": segment, "attention_mask": attention_mask})
        return segments

    # Tokenize and segment speeches
    biden_segments = tokenize_and_segment(biden_texts, tokenizer)
    trump_segments = tokenize_and_segment(trump_texts, tokenizer)

    def single_tensor(batch):
        return {
            'input_ids': torch.tensor([x['input_ids'] for x in batch]),
            'attention_mask': torch.tensor([x['attention_mask'] for x in batch])
        }

    biden_dataset = DataLoader(biden_segments, batch_size=batch_size, shuffle=False, collate_fn=single_tensor)
    trump_dataset = DataLoader(trump_segments, batch_size=batch_size, shuffle=False, collate_fn=single_tensor)

    return biden_dataset, trump_dataset

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    batch_size = 32
    biden, trump = load_multi_dataset(batch_size)

    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    sae_id = "blocks.10.hook_resid_pre"

    ds_mapping = {'biden': biden, 'trump': trump }
    for name, dataset in ds_mapping.items():
        generate_sae_activations(
            dataloader=dataset,
            sae_model='gpt2-small-res-jb',
            sae_id=sae_id,
            transformer=model,
            batch_size=batch_size,
            batches_in_stats_batch=4,
            activation_dir=f'data/speeches-{name}',
            feature_indices=None,
            device=device
        )
    
    print('finished generating activations')


