import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from datasets.live import LIVEDataset


DATASET_ROOT = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/LIVEIQA_release2"
)

def test_live_dataloader():

    dataset = LIVEDataset(DATASET_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )


    batch = next(iter(loader))


    print("Batch immagini:", batch["image"].shape)
    print("Batch MOS:", batch["mos"])


    assert "image" in batch
    assert "mos" in batch


    assert batch["image"].shape[0] == 4

    # canali RGB
    assert batch["image"].shape[1] == 3