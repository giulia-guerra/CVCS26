import sys
from pathlib import Path
from numpy.random import sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# tests/test_live_phase2.py

from datasets.live import LIVEDataset
from torch.utils.data import DataLoader
from PIL import Image

LIVE_PATH = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/LIVEIQA_release2"
)

def test_live_phase2():

    dataset = LIVEDataset(
        LIVE_PATH,
        return_pil=True
    )

    # 1. numero campioni
    assert len(dataset) == 982

    # 2. primo sample
    sample = dataset[0]

    assert "ref_image" in sample
    assert "dist_image" in sample
    assert "mos" in sample
    assert "name" in sample

    # 3. PIL Images
    assert isinstance(
        sample["ref_image"],
        Image.Image
    )

    assert isinstance(
        sample["dist_image"],
        Image.Image
    )

    # 4. reference e distorted diverse
    assert (
        dataset.samples[0]["ref_path"]
        !=
        dataset.samples[0]["dist_path"]
    )

    # 5. MOS valido
    assert isinstance(
        sample["mos"],
        float
    )

    # 6. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: x
    )

    batch = next(iter(loader))

    assert len(batch) == 4

    print("\n=== TEST OK ===")
    print("Campioni:", len(dataset))
    print("Ref:", dataset.samples[0]["ref_path"])
    print("Dist:", dataset.samples[0]["dist_path"])
    print("MOS:", sample["mos"])
    print(type(sample["ref_image"]))
    print(type(sample["dist_image"]))


    for i in [0, 50, 100, 500]:
        s = dataset.samples[i]
        print(i)
        print("REF :", s["ref_path"].name)
        print("DIST:", s["dist_path"].name)
        print("MOS :", s["mos"])
        print()