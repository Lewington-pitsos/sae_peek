from app.collect import new_topk_samples, _init_stats, collect_feature_stats
import torch

def test_keeps_raw_activations():
    start_idx = 512
    n_ft = 16
    seq_len = 64
    batch_size = 256
    samples_to_save = 10
    samples = torch.rand(batch_size, seq_len, n_ft + 2)

    stats = _init_stats(n_ft, 'cpu', list(range(n_ft)), samples_to_save, seq_len)

    collect_feature_stats(start_idx, n_ft, samples, stats, topk=samples_to_save)

    assert 'top_samples' in stats

    max_activation_vals, max_activation_indices = torch.topk(torch.max(samples, dim=1)[0], k=samples_to_save, dim=0)

    indices_10_64_16 = max_activation_indices.unsqueeze(1).expand(-1, 64, -1)

    max_samples = torch.gather(samples, 0, indices_10_64_16)

    assert max_samples.shape == stats['top_samples'].shape
    assert torch.allclose(max_samples, stats['top_samples'], atol=1e-6), "The max_samples do not match the top_samples in stats."


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


    incoming_samples =  torch.tensor([
        [
            [5, 2, 7, 7, 101, 1],
            [2, 1, 4, 11, 102, 1],
        ],
        [
            [5, 4, 1, 1, 391, 1],
            [2, 1, 0, 12, 222, 1],
        ],
        [
            [0,0,0,0, 0, 0],
            [0,0,0,0, 0, 0],  
        ],
        [
            [0,0,0,0, 0, 0],
            [0,0,0,0, 0, 0],  
        ],
    ])
    current_samples = torch.tensor([
        [
            [
                [1, 1, 3, 3, 88, 1],
                [1, 1, 2, 2, 88, 1],
            ],
            [
                [1, 1, 1, 1, 88, 1],
                [1, 1, 1, 1, 88, 1],
            ],
            [
                [0,0,0,0, 0, 0],
                [0,0,0,0, 0, 0],  
            ],
            [
                [0,0,0,0, 0, 0],
                [0,0,0,0, 0, 0],  
            ],
        ],
        [
            [
                [1, 0, 2, 2, 88, 1],
                [1, 1, 2, 2, 88, 1],
            ],
            [
                [1, 1, 2, 2, 88, 1],
                [1, 1, 1, 1, 88, 1],
            ],
            [
                [0,0,0,0, 0, 0],
                [0,0,0,0, 0, 0],  
            ],
            [
                [0,0,0,0, 0, 0],
                [0,0,0,0, 0, 0],  
            ],
        ]
    ])


    topk = 2

    new_maxes, new_indices = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([
        [87, 4, 12, 12],
        [40, 3, 11, 11]
    ]))

    assert torch.all(new_indices == torch.tensor([
        [1, 15, 3, 15],
        [2, 0, 11, 14]
    ]))

def test_new_topk_case1():
    start_idx = 200
    acts = torch.tensor([
        [
            [3, 8, 2, 7, 6],
            [1, 4, 9, 5, 2],
        ],
        [
            [5, 3, 1, 4, 4],
            [2, 8, 7, 12, 3],
        ],
        [
            [3, 5, 2, 6, 1],
            [4, 9, 3, 7, 8],  
        ],
    ])

    current_maxes = torch.tensor([
        [5, 7, 5, 2, 1],
        [3, 7, 3, 4, 0]
    ])

    current_max_indices = torch.tensor([
        [2, 1, 6, 9, 5],
        [1, 4, 10, 3, 7]
    ])

    incoming_samples =  torch.tensor([
        [
            [5, 2, 7, 7, 0, 101, 1],
            [2, 1, 4, 11, 0, 102, 1],
        ],
        [
            [5, 4, 1, 1, 0, 391, 1],
            [2, 1, 0, 12, 0, 222, 1],
        ],
        [
            [0,0,0,0, 0, 0, 0],
            [0,0,0,0, 0, 0, 0],  
        ],
    ])
    current_samples = torch.tensor([
        [
            [
                [1, 1, 3, 1, 3, 88, 1],
                [1, 1, 2, 1, 2, 88, 1],
            ],
            [
                [1, 1, 1, 1, 1, 88, 1],
                [1, 1, 1, 1, 1, 88, 1],
            ],
            [
                [0,0,0,0, 0, 0, 0],
                [0,0,0,0, 0,0, 0],  
            ],
            [
                [0,0,0,0, 0,0, 0],
                [0,0,0,0, 0,0, 0],  
            ],
        ],
        [
            [
                [1, 0, 2, 2, 0, 88, 1],
                [1, 1, 2, 2, 0, 88, 1],
            ],
            [
                [1, 1, 2, 2, 0, 88, 1],
                [1, 1, 1, 1, 0, 88, 1],
            ],
            [
                [0,0,0,0, 0, 0, 0],
                [0,0,0,0, 0, 0, 0],  
            ],
            [
                [0,0,0,0, 0, 0, 0],
                [0,0,0,0, 0, 0, 0],  
            ],
        ]
    ])


    topk = 2

    new_maxes, new_indices = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([
        [5,  9,  9, 12,  8],
        [5,  8,  7,  7,  6]
    ]))

    assert torch.all(new_indices == torch.tensor([
        [2, 202, 200, 201, 202],
        [201, 200, 201, 200, 200]
    ]))




def test_new_topk_case2():
    start_idx = 64
    acts = torch.tensor([
        [
            [6, 7, 2],
            [9, 1, 3],
        ],
        [
            [2, 5, 8],
            [4, 7, 6],
        ],
        [
            [3, 8, 5],
            [7, 9, 4],  
        ],
    ])

    current_maxes = torch.tensor([
        [6, 2, 9],
        [3, 4, 1],
        [3, 2, 0],
        [2, 1, 0]
    ])

    current_max_indices = torch.tensor([
        [3, 2, 8],
        [4, 7, 6],
        [3, 1, 9],
        [0, 4, 1],
    ])


    incoming_samples =  torch.tensor([
        [
            [6, 7, 2, 121, 1],
            [9, 1, 3, 121, 1],
        ],
        [
            [2, 5, 8, 122, 1],
            [4, 7, 6, 100, 1],
        ],
        [
            [3, 8, 5, 80, 1],
            [7, 9, 4, 77, 1],  
        ],
    ])

    current_samples =  torch.tensor([
        [
            [
                [6, 7, 2, 121, 1],
                [9, 1, 3, 121, 1],
            ],
            [
                [2, 5, 8, 122, 1],
                [4, 7, 6, 100, 1],
            ],
            [
                [3, 8, 5, 80, 1],
                [7, 9, 4, 77, 1],  
            ],
        ],
        [
            [
                [6, 7, 2, 121, 1],
                [9, 1, 3, 121, 1],
            ],
            [
                [2, 5, 8, 122, 1],
                [4, 7, 6, 100, 1],
            ],
            [
                [3, 8, 5, 80, 1],
                [7, 9, 4, 77, 1],  
            ],
        ],
                [
            [
                [6, 7, 2, 121, 1],
                [9, 1, 3, 121, 1],
            ],
            [
                [2, 5, 8, 122, 1],
                [4, 7, 6, 100, 1],
            ],
            [
                [3, 8, 5, 80, 1],
                [7, 9, 4, 77, 1],  
            ],
        ],
                [
            [
                [6, 7, 2, 121, 1],
                [9, 1, 3, 121, 1],
            ],
            [
                [2, 5, 8, 122, 1],
                [4, 7, 6, 100, 1],
            ],
            [
                [3, 8, 5, 80, 1],
                [7, 9, 4, 77, 1],  
            ],
        ],
    ])


    topk = 4

    new_maxes, new_indices = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([[9, 9, 9],
        [7, 7, 8],
        [6, 7, 5],
        [4, 4, 3]]
    ))

    assert torch.all(new_indices == torch.tensor([[64, 66,  8],
        [66, 64, 65],
        [ 3, 65, 66],
        [65,  7, 64]]
    ))
