import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from torch.utils.data import DataLoader
from datasets.tid2013 import TID2013Dataset



DATASET_ROOT = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/tid2013"
)



def test_tid2013_dataloader():

    dataset = TID2013Dataset(
        DATASET_ROOT
    )


    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )


    batch = next(iter(loader))


    assert "image" in batch
    assert "mos" in batch


    # formato PyTorch:
    # [batch, channels, height, width]

    assert batch["image"].shape[0] == 4

    # RGB
    assert batch["image"].shape[1] == 3


    assert len(batch["mos"]) == 4



    print(
        "Batch immagini:",
        batch["image"].shape
    )


    print(
        "Batch MOS:",
        batch["mos"]
    )