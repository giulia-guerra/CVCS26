import torch

from src.metrics.metrics import (
    compute_srcc,
    compute_plcc
)

from src.metrics.similarity import (
    cosine_similarity,
    l2_distance
)



def test_phase1_pipeline():


    # fake feature estratte dal modello

    reference_features = torch.tensor(
        [
            [1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]
        ]
    )


    distorted_features = torch.tensor(
        [
            [0.9,0.1,0.],
            [0.,0.8,0.2],
            [0.1,0.,0.9]
        ]
    )



    # similarity

    scores = cosine_similarity(
        reference_features,
        distorted_features
    )


    assert scores.shape[0] == 3



    # distanza

    distances = l2_distance(
        reference_features,
        distorted_features
    )


    assert distances.shape[0] == 3



    # MOS fake

    mos = torch.tensor(
        [
            0.9,
            0.8,
            0.7
        ]
    )


    srcc = compute_srcc(
        scores,
        mos
    )


    plcc = compute_plcc(
        scores,
        mos
    )


    print(
        "SRCC:",
        srcc
    )


    print(
        "PLCC:",
        plcc
    )


    assert srcc >= -1
    assert srcc <= 1

    assert plcc >= -1
    assert plcc <= 1