from datasets.tid2013 import TID2013Dataset


DATASET_ROOT = (
    "/work/cvcs2026/"
    "Cross_Entropy_Champions/"
    "datasets/tid2013"
)


def test_tid2013_format():

    dataset = TID2013Dataset(DATASET_ROOT)


    sample = dataset[0]


    assert "reference" in sample
    assert "distorted" in sample
    assert "mos" in sample
    assert "name" in sample


    print(sample["name"])
    print(sample["mos"])
    print(sample["reference"].size)
    print(sample["distorted"].size)
