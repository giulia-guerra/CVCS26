import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.live import LIVEDataset


DATASET_ROOT = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/LIVEIQA_release2"
)


def test_live_dataset():

    dataset = LIVEDataset(DATASET_ROOT)

    print("Numero campioni:", len(dataset))

    sample = dataset[0]

    print(sample.keys())
    print(sample["image"].shape)
    print(sample["mos"])

    assert "image" in sample
    assert "mos" in sample