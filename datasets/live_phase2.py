from pathlib import Path

import scipy.io
from PIL import Image

from .base_dataset import BaseDataset


class LIVEPhase2Dataset(BaseDataset):
    
    def __init__(
        self,
        root_dir,
        transform=None
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.root_dir = Path(root_dir)

        self.samples = []

        self.dmos_file = (
            self.root_dir /
            "dmos_realigned.mat"
        )

        self._load_samples()

    def _load_samples(self):

        mat = scipy.io.loadmat(
            self.dmos_file
        )

        if "dmos_new" in mat:

            dmos = mat["dmos_new"].flatten()

        elif "dmos" in mat:

            dmos = mat["dmos"].flatten()

        else:

            raise KeyError(
                "Nessun MOS trovato"
            )

        images = sorted(
            [
                p
                for p in self.root_dir.rglob("*")
                if p.suffix.lower()
                in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )

        print(
            "Immagini LIVE trovate:",
            len(images)
        )

        for img, mos in zip(
            images,
            dmos
        ):

            self.samples.append(
                {
                    "image": img,
                    "mos": float(mos)
                }
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        img = self.load_image(
            sample["image"]
        )

        img = img.resize(
            (224, 224)
        )

        return {

            "ref_image": img,

            "dist_image": img,

            "mos": sample["mos"],

            "name": sample["image"].name

        }