from pathlib import Path

from .base_dataset import BaseDataset


class TID2013Phase2Dataset(BaseDataset):

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

        self._load_samples()

    def _load_samples(self):

        distorted_dir = (
            self.root_dir /
            "distorted_images"
        )

        images = sorted(
            [
                p
                for p in distorted_dir.rglob("*")
                if p.suffix.lower()
                in [
                    ".bmp",
                    ".png",
                    ".jpg",
                    ".jpeg"
                ]
            ]
        )

        print(
            "TID2013 immagini trovate:",
            len(images)
        )

        for img in images:

            self.samples.append(
                {
                    "image": img,
                    "mos": 0.0
                }
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        dist_img = self.load_image(
            sample["image"]
        )

        dist_img = dist_img.resize(
            (224, 224)
        )

        # placeholder
        # da sostituire col mapping reale
        ref_img = dist_img

        return {

            "ref_image": ref_img,

            "dist_image": dist_img,

            "mos": sample["mos"],

            "name": sample["image"].name

        }