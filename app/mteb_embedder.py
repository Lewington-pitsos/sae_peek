import torch
import numpy as np
from typing import Any
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE


class SAEncoder():
    def __init__(self, transformer_name, sae_model, sae_id, max_sequence_length, device) -> None:
        self.transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_sequence_length = max_sequence_length

        self.sae, _, _ = SAE.from_pretrained(
            release = sae_model, # see other options in sae_lens/pretrained_saes.yaml
            sae_id = sae_id, 
            device = device
        )
    
    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        input_ids, attention_mask = self.tokenizer(sentences, padding='longest', truncation=True, max_length=self.max_sequence_length,  return_tensors='pt')

        _, cache = self.transformer.run_with_cache(
            input_ids, 
            attention_mask=attention_mask, 
            prepend_bos=True, 
            stop_at_layer=self.sae.cfg.hook_layer + 1
        )

        hidden_states = cache[self.sae.cfg.hook_name]

        features = self.sae.encode(hidden_states)

        return features