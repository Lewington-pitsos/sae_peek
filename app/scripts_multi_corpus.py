import torch
from torch.utils.data import DataLoader, Dataset
from sae_lens import HookedSAETransformer
import pandas as pd
from transformers import GPT2Tokenizer 
import transformers

from app.scripts import load_pile10k
from app.peek import generate_sae_activations, llm_assessment

def load_multi_dataset(tokenizer, batch_size, sequence_length):
    df = pd.read_csv('data/us_2020_election_speeches.csv')
    biden = df[df['speaker'] == 'Joe Biden']
    trump = df[df['speaker'] == 'Donald Trump']

    print('Biden:', biden.shape[0])
    print('Trump:', trump.shape[0])

    biden_texts = biden['text'].tolist()
    trump_texts = trump['text'].tolist()

    transformers.logging.set_verbosity_error()

    def tokenize_and_segment(texts, tokenizer, overlap=5):
        segments = []
        for text in texts:
            tokens = tokenizer(text)['input_ids']
            for i in range(0, len(tokens), sequence_length - overlap):
                segment = tokens[i:i + sequence_length]
                attention_mask = [1] * len(segment)
                if len(segment) < sequence_length:
                    padding = [0] * (sequence_length - len(segment))
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
    
    batch_size = 4
    sequence_length = 256
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    biden, trump = load_multi_dataset(tokenizer, batch_size, sequence_length)
    pile10k = load_pile10k(tokenizer, batch_size,  sequence_length, num_samples=None)

    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    sae_id = "blocks.10.hook_resid_pre"

    ds_mapping = {
        # 'biden': biden, 
        # 'trump': trump, 
        'pile10k': pile10k 
    }

    for name, dataset in ds_mapping.items():
        generate_sae_activations(
            dataloader=dataset,
            sae_model='gpt2-small-res-jb',
            sae_id=sae_id,
            transformer=model,
            batches_in_stats_batch=4,
            activation_dir=f'data/2048-{name}',
            feature_indices=list(range(2048)),
            device=device
        )
    
    print('finished generating activations')

    llm_assessment('data/2048-pile10k', output='cruft/pile10k-2048.json', samples_per_feature=10)
