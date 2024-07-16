from datasets import load_dataset
from torch.utils.data import DataLoader

def load_pile10k(tokenizer, batch_size, sequence_length, num_samples=5000):
    dataset = load_dataset('NeelNanda/pile-10k', split="train")
    dataset = dataset.select(range(num_samples))

    def tokenize(batch):
        text_input = [b['text'] for b in batch]
        return tokenizer(text_input, padding='longest', truncation=True, max_length=sequence_length, return_tensors='pt')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tokenize)

    return dl

