import torch

from src.metrics.similarity import (
    cosine_similarity,
    l2_distance
)



def test_similarity():


    a = torch.tensor(
        [[1.,0.,0.]]
    )

    b = torch.tensor(
        [[1.,0.,0.]]
    )


    cos = cosine_similarity(
        a,b
    )


    dist = l2_distance(
        a,b
    )


    assert torch.isclose(
        cos,
        torch.tensor([1.])
    )


    assert torch.isclose(
        dist,
        torch.tensor([0.])
    )