import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.tid2013 import TID2013Dataset
from PIL import Image


PATH = "/work/cvcs2026/Cross_Entropy_Champions/datasets/tid2013"


def test_tid2013_real():

    dataset = TID2013Dataset(PATH)

    assert len(dataset) > 0


    sample = dataset[0]


    assert "ref_image" in sample
    assert "dist_image" in sample
    assert "mos" in sample
    assert "name" in sample


    assert isinstance(
        sample["ref_image"],
        Image.Image
    )

    assert isinstance(
        sample["dist_image"],
        Image.Image
    )


    assert (
        dataset.samples[0]["ref_path"]
        !=
        dataset.samples[0]["dist_path"]
    )


    print()
    print("=== TEST OK ===")
    print("Campioni:", len(dataset))
    print("REF:", dataset.samples[0]["ref_path"])
    print("DIST:", dataset.samples[0]["dist_path"])
    print("MOS:", sample["mos"])