from pathlib import Path

from .base_dataset import BaseDataset


class PIPALDataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.samples = []

        self.load_metadata()

    def load_metadata(self):

        label_dir = self.root_dir / "Train_Label"

        for label_file in sorted(label_dir.glob("*.txt")):

            with open(label_file, "r") as f:

                for line in f:

                    image_name, mos = line.strip().split(",")

                    image_path = (
                        self.root_dir
                        / "Dist_Imgs"
                        / image_name
                    )

                    self.samples.append(
                        (
                            image_path,
                            float(mos)
                        )
                    )

    def __getitem__(self, idx):

        image_path, mos = self.samples[idx]

        image = self.load_image(image_path)

        return {
            "image": image,
            "mos": mos
        }