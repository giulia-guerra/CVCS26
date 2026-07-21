import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchvision import transforms
from datasets.tid2013 import TID2013Dataset


DATASET_ROOT = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/tid2013"
)


def test_tid2013_real_dataset():

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = TID2013Dataset(
        DATASET_ROOT,
        transform=transform
    )


    assert len(dataset) == 3000


    sample = dataset[0]


    print("Image shape:", sample["image"].shape)
    print("MOS:", sample["mos"])


    assert "image" in sample
    assert "mos" in sample

    assert sample["image"].shape[0] == 3