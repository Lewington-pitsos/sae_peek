from torch.utils.data import DataLoader
from datasets import load_dataset

from sae_lens import HookedSAETransformer
from app.peek import sae_assessment


if __name__ == '__main__':
    model = HookedSAETransformer.from_pretrained("gpt2", device='cuda')
    batch_size = 256
    sequence_length = 128
    n_samples = 4096

    for sae_id in [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.2.hook_resid_pre",
        "blocks.3.hook_resid_pre",
        "blocks.4.hook_resid_pre",
        "blocks.5.hook_resid_pre",
        "blocks.6.hook_resid_pre",
        "blocks.7.hook_resid_pre",
        "blocks.8.hook_resid_pre",
        "blocks.9.hook_resid_pre",
        "blocks.10.hook_resid_pre",
        "blocks.11.hook_resid_pre",
        "blocks.11.hook_resid_post",
    ]:    
        
        dataset = load_dataset('NeelNanda/pile-10k')
        def tokenize(x):
            output = model.tokenizer([y['text'] for y in x], return_tensors='pt', truncation=True, max_length=sequence_length, padding='max_length')
        
            return output['input_ids'], output['attention_mask']

        dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)


        sae_assessment(
            dataset=dataset[:n_samples],
            batch_size=batch_size,
            sae_model='gpt2-small-res-jb',
            sae_id=sae_id,
            transformer=model,
            activation_dir=f'data/pile10k-v2-{sae_id}',
            output=f'cruft/pile10k-v2-{sae_id}.json',
            feature_indices=list(range(150, 300))
        )
