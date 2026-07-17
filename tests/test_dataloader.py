import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tempfile

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from datasets.live import LIVEDataset


def test_fake_dataloader_live():

    with tempfile.TemporaryDirectory() as tmpdir:

        image_path = Path(tmpdir) / "fake.png"

        fake_image = np.zeros(
            (64, 64, 3),
            dtype=np.uint8
        )

        Image.fromarray(fake_image).save(image_path)

        dataset = LIVEDataset(tmpdir)

        dataset.samples = [
            (image_path, 50.0)
        ]

        loader = DataLoader(
            dataset,
            batch_size=1
        )

        batch = next(iter(loader))

        assert batch["image"].shape == (1, 64, 64, 3)
        assert batch["mos"].item() == 50.0