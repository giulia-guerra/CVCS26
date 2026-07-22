from pathlib import Path

from .base_dataset import BaseDataset


class PIPALPhase2Dataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        split="train"
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.root_dir = Path(root_dir)

        self.split = split

        self.samples = []

        self._load_samples()

    def _load_samples(self):

        if self.split == "train":

            img_dir = (
                self.root_dir /
                "Dist_Imgs"
            )

            images = sorted(
                img_dir.rglob("*.bmp")
            )

            if len(images) == 0:

                images = sorted(
                    img_dir.rglob("*.png")
                )

        else:

            img_dir = (
                self.root_dir /
                "Val_Dist"
            )

            images = sorted(
                [
                    x
                    for x in img_dir.rglob("*")
                    if x.suffix.lower()
                    in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp"
                    ]
                ]
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

        # placeholder:
        # finché non facciamo il mapping corretto
        ref_img = dist_img

        return {

            "ref_image": ref_img,

            "dist_image": dist_img,

            "mos": sample["mos"],

            "name": sample["image"].name

        }