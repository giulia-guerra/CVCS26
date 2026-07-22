import torch

from src.metrics.distance import cosine_distance, l2_distance


def test_distance():

    x1 = torch.tensor([[1.,0.,0.]])
    x2 = torch.tensor([[1.,0.,0.]])

    cos = cosine_distance(x1, x2)
    l2 = l2_distance(x1, x2)

    assert cos.item() == 0
    assert l2.item() == 0