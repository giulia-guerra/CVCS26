from scipy.io import loadmat
from pathlib import Path

from .base_dataset import BaseDataset


class LIVEDataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.samples = []

        self.load_metadata()

    def load_metadata(self):

        dmos_data = loadmat(
            self.root_dir / "dmos_realigned.mat"
        )

        dmos = dmos_data["dmos_new"].flatten()
        orgs = dmos_data["orgs"].flatten()

        index = 0

        distortions = [
            ("jp2k", 227),
            ("jpeg", 233),
            ("wn", 174),
            ("gblur", 174),
            ("fastfading", 174),
        ]

        for distortion_name, count in distortions:

            for i in range(1, count + 1):

                image_path = (
                    self.root_dir
                    / distortion_name
                    / f"img{i}.bmp"
                )

                if orgs[index] == 0:

                    self.samples.append(
                        (
                            image_path,
                            float(dmos[index])
                        )
                    )

                index += 1

    def __getitem__(self, idx):

        image_path, mos = self.samples[idx]

        image = self.load_image(image_path)

        return {
            "image": image,
            "mos": mos
        }