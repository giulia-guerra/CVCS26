from datasets.pipal import PIPALDataset


DATASET_ROOT = (
    "/work/cvcs2026/"
    "Cross_Entropy_Champions/"
    "datasets/PIPAL"
)


def test_pipal_format():

    dataset = PIPALDataset(DATASET_ROOT)


    sample = dataset[0]


    assert "reference" in sample
    assert "distorted" in sample
    assert "mos" in sample
    assert "name" in sample


    print(sample["name"])
    print(sample["mos"])
    print(sample["reference"].size)
    print(sample["distorted"].size)
