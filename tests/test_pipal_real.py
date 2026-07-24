from datasets.pipal import PIPALDataset


def test_pipal():

    dataset = PIPALDataset(
        "/work/cvcs2026/Cross_Entropy_Champions/datasets/PIPAL"
    )


    print(
        "Campioni:",
        len(dataset)
    )


    sample = dataset[0]


    print(type(sample["ref_image"]))
    print(type(sample["dist_image"]))

    print(
        "REF:",
        sample["ref_image"].size
    )

    print(
        "DIST:",
        sample["dist_image"].size
    )

    print(
        "MOS:",
        sample["mos"]
    )


    assert "ref_image" in sample
    assert "dist_image" in sample

    assert sample["ref_image"] != sample["dist_image"]

    print("=== TEST PIPAL OK ===")