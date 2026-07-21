import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.metrics.metrics import (
    srcc,
    plcc,
    cosine_distance,
    l2_distance
)


def test_metrics():

    pred = torch.tensor([1.,2.,3.,4.])
    target = torch.tensor([1.,2.,3.,4.])

    assert abs(srcc(pred,target) - 1.0) < 1e-6
    assert abs(plcc(pred,target) - 1.0) < 1e-6


def test_distance():

    x = torch.tensor([1.,0.,0.])
    y = torch.tensor([1.,0.,0.])

    assert cosine_distance(x,y) == 0
    assert l2_distance(x,y) == 0