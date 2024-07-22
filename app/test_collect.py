from app.collect import new_topk_samples, _reshape_samples
import torch

def test_reshape_samples():
    samples = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ],
        [
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4]
        ]
    ])
    n_features = 2
    topk_indices = torch.tensor([[1, 0]])

    expected_output = torch.tensor([
        [
            [[1.3, 1.5, 1.6], [0.2, 0.3, 0.4]],
            [[1.7, 1.9, 2.0], [0.6, 0.7, 0.8]],
            [[2.1, 2.3, 2.4], [1.0, 1.1, 1.2]],
        ],
    ])

    output = _reshape_samples(samples.clone(), n_features, topk_indices)


    assert torch.equal(output, expected_output), f"Output mismatch: expected {expected_output}, got {output}"

    topk_indices = torch.tensor([[0, 1]])
    output = _reshape_samples(samples.clone(), n_features, topk_indices)
    expected_output = torch.tensor([
        [
            [[0.1, 0.3, 0.4], [1.4, 1.5, 1.6]],
            [[0.5, 0.7, 0.8], [1.8, 1.9, 2.0]],
            [[0.9, 1.1, 1.2], [2.2, 2.3, 2.4]],
        ],
    ])
    assert torch.equal(output, expected_output), f"Output mismatch: expected {expected_output}, got {output}"


def test_new_topk_no_current_maxes():
    start_idx = 1290
    acts =  torch.tensor([
        [
            [0,0,0,0],
            [0, 0, 0, 0]
        ],
        [
            [0,0,0,0],
            [0, 0, 0, 0]
        ],
    ])

    current_maxes = torch.tensor([])
    current_max_indices = torch.tensor([])
    incoming_samples =  torch.tensor([
        [
            [0,0,0,0, 101, 1],
            [0,0,0,0, 102, 1],
        ],
        [
            [0,0,0,0, 391, 1],
            [0,0,0,0, 222, 1],
        ],
    ])
    current_samples = torch.tensor([])

    topk = 2

    new_maxes, _, new_samples = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([
        [0,0,0,0],
        [0, 0, 0, 0]
    ]))

    print(new_samples)
    assert torch.all(new_samples == torch.tensor([[[[  0, 101,   1],
          [  0, 101,   1],
          [  0, 101,   1],
          [  0, 101,   1]],

         [[  0, 102,   1],
          [  0, 102,   1],
          [  0, 102,   1],
          [  0, 102,   1]]],


        [[[  0, 391,   1],
          [  0, 391,   1],
          [  0, 391,   1],
          [  0, 391,   1]],

         [[  0, 222,   1],
          [  0, 222,   1],
          [  0, 222,   1],
          [  0, 222,   1]]]]))

    


def test_new_topk_case0():

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
                [3, 88, 1],
                [2, 88, 1],
            ],
            [
                [1, 88, 1],
                [1, 88, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],  
            ],
            [
                [0, 0, 0],
                [0, 0, 0],  
            ],
        ],
        [
            [
                [2, 88, 1],
                [2, 88, 1],
            ],
            [
                [2, 88, 1],
                [1, 88, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],  
            ],
            [
                [0, 0, 0],
                [0, 0, 0],  
            ],
        ]
    ]).permute(0, 2, 1, 3)


    topk = 2

    new_maxes, new_indices, new_samples = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([
        [87, 4, 12, 12],
        [40, 3, 11, 11]
    ]))

    assert torch.all(new_indices == torch.tensor([
        [1, 15, 3, 15],
        [2, 0, 11, 14]
    ]))

    assert new_samples.shape == current_samples.shape

    expected_samples = torch.tensor([[[[  3,  88,   1],
          [  4, 391,   1],
          [  0,   0,   0],
          [  1, 391,   1]],

         [[  2,  88,   1],
          [  1, 222,   1],
          [  0,   0,   0],
          [ 12, 222,   1]]],


        [[[  2,  88,   1],
          [  2,  88,   1],
          [  0,   0,   0],
          [  7, 101,   1]],

         [[  2,  88,   1],
          [  1,  88,   1],
          [  0,   0,   0],
          [ 11, 102,   1]]]])
    assert torch.all(new_samples == expected_samples)


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
                [3, 88, 1],
                [2, 88, 1],
            ],
            [
                [1, 88, 1],
                [1, 88, 1],
            ],
            [
                [0, 0, 0],
                [0,0, 0],  
            ],
            [
                [0, 0, 0],
                [0,0, 0],  
            ],
            [
                [0, 0, 0],
                [0,0, 0],  
            ],
        ],
        [
            [
                [1, 88, 1],
                [1, 88, 1],
            ],
            [
                [12, 88, 1],
                [243, 88, 1],
            ],
            [
                [6, 88, 1],
                [3, 88, 1],
            ],
            [
                [0, 0, 0],
                [0,0, 0],  
            ],
            [
                [0, 0, 0],
                [0,0, 0],  
            ],
        ]
    ]).permute(0, 2, 1, 3)


    topk = 2

    new_maxes, new_indices, new_samples = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

    assert torch.all(new_maxes == torch.tensor([
        [5,  9,  9, 12,  8],
        [5,  8,  7,  7,  6]
    ]))

    assert torch.all(new_indices == torch.tensor([
        [2, 202, 200, 201, 202],
        [201, 200, 201, 200, 200]
    ]))
    assert new_samples.shape == current_samples.shape




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
                [2, 121, 1],
                [3, 121, 1],
            ],
            [
                [8, 122, 1],
                [6, 100, 1],
            ],
            [
                [5, 80, 1],
                [4, 77, 1],  
            ],
        ],
                [
            [
                [2, 121, 1],
                [3, 121, 1],
            ],
            [
                [8, 122, 1],
                [6, 100, 1],
            ],
            [
                [5, 80, 1],
                [4, 77, 1],  
            ],
        ],
                [
            [
                [2, 121, 1],
                [3, 121, 1],
            ],
            [
                [8, 122, 1],
                [6, 100, 1],
            ],
            [
                [5, 80, 1],
                [4, 77, 1],  
            ],
        ],
                [
            [
                [2, 121, 1],
                [3, 121, 1],
            ],
            [
                [8, 122, 1],
                [6, 100, 1],
            ],
            [
                [5, 80, 1],
                [4, 77, 1],  
            ],
        ]
    ]).permute(0, 2, 1, 3)


    topk = 4

    new_maxes, new_indices, new_samples = new_topk_samples(start_idx, current_samples, incoming_samples, acts, current_maxes, current_max_indices, topk)

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
    assert new_samples.shape == current_samples.shape

    expected_samples = torch.tensor([[[[  6, 121,   1],
          [  8,  80,   1],
          [  5,  80,   1]],

         [[  9, 121,   1],
          [  9,  77,   1],
          [  4,  77,   1]]],


        [[[  3,  80,   1],
          [  7, 121,   1],
          [  8, 122,   1]],

         [[  7,  77,   1],
          [  1, 121,   1],
          [  6, 100,   1]]],


        [[[  2, 121,   1],
          [  5, 122,   1],
          [  5,  80,   1]],

         [[  3, 121,   1],
          [  7, 100,   1],
          [  4,  77,   1]]],


        [[[  2, 122,   1],
          [  8, 122,   1],
          [  2, 121,   1]],

         [[  4, 100,   1],
          [  6, 100,   1],
          [  3, 121,   1]]]])
    

    assert torch.all(new_samples == expected_samples)

