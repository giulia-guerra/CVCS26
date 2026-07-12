import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.live import LIVEDataset


def test_fake_live_dataset():

    with tempfile.TemporaryDirectory() as tmpdir:

        image_path = Path(tmpdir) / "fake.png"

        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)

        Image.fromarray(fake_image).save(image_path)

        dataset = LIVEDataset(tmpdir)

        dataset.samples = [
            (image_path, 80.0)
        ]

        sample = dataset[0]

        assert sample["image"].shape == (64, 64, 3)
        assert sample["mos"] == 80.0
        assert len(dataset) == 1