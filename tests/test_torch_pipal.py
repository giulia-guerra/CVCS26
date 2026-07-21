import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.pipal import PIPALDataset



DATASET_ROOT = (
    "/work/cvcs2026/Cross_Entropy_Champions/"
    "datasets/PIPAL"
)



def test_pipal_train():


    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor()
        ]
    )



    dataset = PIPALDataset(
        DATASET_ROOT,
        transform=transform,
        split="train"
    )



    print(
        "Numero immagini PIPAL:",
        len(dataset)
    )



    # controllo singolo elemento
    sample = dataset[0]


    assert "image" in sample

    assert "mos" in sample

    assert "path" in sample



    assert sample["image"].shape[0] == 3



    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )



    batch = next(iter(loader))



    print(
        "Batch immagini:",
        batch["image"].shape
    )


    print(
        "Batch MOS:",
        batch["mos"]
    )



    assert "image" in batch

    assert "mos" in batch



    assert batch["image"].shape[0] == 4

    assert batch["image"].shape[1] == 3



    assert len(batch["mos"]) == 4