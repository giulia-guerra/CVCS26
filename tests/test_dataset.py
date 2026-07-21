import tempfile
from pathlib import Path

import numpy as np

from PIL import Image

from scipy.io import savemat


from datasets.live import LIVEDataset



def test_fake_live_dataset():


    with tempfile.TemporaryDirectory() as tmpdir:


        tmpdir = Path(tmpdir)



        # Fake image

        image_path = tmpdir / "fake.png"


        fake_image = np.zeros(
            (64,64,3),
            dtype=np.uint8
        )


        Image.fromarray(
            fake_image
        ).save(image_path)



        # Fake MOS file

        savemat(
            tmpdir / "dmos_realigned.mat",
            {
                "dmos": np.array(
                    [[1.0]]
                )
            }
        )



        dataset = LIVEDataset(
            tmpdir
        )



        assert len(dataset) == 1



        sample = dataset[0]


        assert "image" in sample
        assert "mos" in sample


        assert sample["image"].shape == (
            224,
            224,
            3
        )


        assert sample["mos"] == 1.0