import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.tid2013_phase2 import TID2013Phase2Dataset


def test_tid2013_phase2():

    root = "/work/cvcs2026/Cross_Entropy_Champions/datasets/tid2013"

    dataset = TID2013Phase2Dataset(root)

    sample = dataset[0]

    assert "ref_image" in sample
    assert "dist_image" in sample
    assert "name" in sample
    assert "mos" in sample