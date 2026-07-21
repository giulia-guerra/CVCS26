import tempfile

from pathlib import Path


import numpy as np


from PIL import Image


from scipy.io import savemat


from torch.utils.data import DataLoader



from datasets.live import LIVEDataset




def test_fake_dataloader_live():


    with tempfile.TemporaryDirectory() as tmpdir:


        tmpdir = Path(tmpdir)



        # create fake images

        for i in range(4):

            image_path = (
                tmpdir /
                f"fake_{i}.png"
            )


            fake_image = np.zeros(
                (64,64,3),
                dtype=np.uint8
            )


            Image.fromarray(
                fake_image
            ).save(image_path)



        # fake MOS

        savemat(
            tmpdir / "dmos_realigned.mat",
            {
                "dmos": np.array(
                    [
                        [1.0],
                        [2.0],
                        [3.0],
                        [4.0]
                    ]
                )
            }
        )



        dataset = LIVEDataset(
            tmpdir
        )


        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False
        )


        batch = next(
            iter(loader)
        )


        assert batch["image"].shape == (
            2,
            224,
            224,
            3
        )


        assert batch["mos"].shape[0] == 2