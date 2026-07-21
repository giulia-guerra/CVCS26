import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


from datasets.pipal import PIPALDataset



DATASET_ROOT = (
    "/work/cvcs2026/"
    "Cross_Entropy_Champions/"
    "datasets/PIPAL"
)



def test_pipal_val():


    dataset=PIPALDataset(
        DATASET_ROOT,
        split="val"
    )


    print(
        "Numero immagini:",
        len(dataset)
    )


    sample=dataset[0]


    print(
        sample["image"].shape
    )


    print(
        sample["mos"]
    )



    assert "image" in sample
    assert "mos" in sample