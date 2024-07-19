from app.collect import new_topk_samples, _init_stats, collect_feature_stats
import torch

def test_keeps_raw_activations():
    start_idx = 512
    n_ft = 128
    seq_len = 64
    batch_size = 256
    samples_to_save = 10
    activations = torch.rand(batch_size, seq_len, n_ft + 2)

    stats = _init_stats(n_ft, 'cpu', list(range(n_ft)), samples_to_save, seq_len)

    collect_feature_stats(start_idx, n_ft, activations, stats, topk=samples_to_save)

    assert 'top_samples' in stats

def test_new_topk():

    start_idx = 14
    acts =  torch.tensor([
        [
            [5, 2, 7, 7],
            [2, 1, 4, 11],
        ],
        [
            [5, 4, 1, 1],
            [2, 1, 0, 12],
        ],
        [
            [0,0,0,0],
            [0,0,0,0],  
        ],
        [
            [0,0,0,0],
            [0,0,0,0],  
        ],
    ])

    current_maxes = torch.tensor([
        [87, 3, 12, 5],
        [40, 3, 11, 1]
    ])

    current_max_indices = torch.tensor([
        [1, 3, 3, 9],
        [2, 0, 11, 11]
    ])


    topk = 2

    new_maxes, new_indices = new_topk_samples(start_idx, acts, current_maxes, current_max_indices, topk)

    print(new_maxes)
    print(new_indices)

    assert torch.all(new_maxes == torch.tensor([
        [87, 4, 12, 12],
        [40, 3, 11, 11]
    ]))

    assert torch.all(new_indices == torch.tensor([
        [1, 15, 3, 15],
        [2, 0, 11, 14]
    ]))