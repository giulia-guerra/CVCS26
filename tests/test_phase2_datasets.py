from datasets.live_phase2 import LIVEPhase2Dataset
from datasets.tid2013_phase2 import TID2013Phase2Dataset
from datasets.pipal_phase2 import PIPALPhase2Dataset

from PIL import Image


LIVE_ROOT = "/work/cvcs2026/Cross_Entropy_Champions/datasets/LIVEIQA_release2"

TID_ROOT = "/work/cvcs2026/Cross_Entropy_Champions/datasets/tid2013"

PIPAL_ROOT = "/work/cvcs2026/Cross_Entropy_Champions/datasets/PIPAL"


def test_live_phase2():

    dataset = LIVEPhase2Dataset(
        LIVE_ROOT
    )

    sample = dataset[0]

    print(type(sample["ref_image"]))
    print(type(sample["dist_image"]))
    print(sample["name"])
    print(sample["mos"])

    assert isinstance(
        sample["ref_image"],
        Image.Image
    )

    assert isinstance(
        sample["dist_image"],
        Image.Image
    )


def test_tid_phase2():

    dataset = TID2013Phase2Dataset(
        TID_ROOT
    )

    sample = dataset[0]

    print(type(sample["ref_image"]))
    print(type(sample["dist_image"]))
    print(sample["name"])
    print(sample["mos"])


def test_pipal_phase2():

    dataset = PIPALPhase2Dataset(
        PIPAL_ROOT
    )

    sample = dataset[0]

    print(type(sample["ref_image"]))
    print(type(sample["dist_image"]))
    print(sample["name"])
    print(sample["mos"])