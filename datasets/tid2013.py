from .base_dataset import BaseDataset


class TID2013Dataset(BaseDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        mos_file = self.root_dir / "mos_with_names.txt"
        distorted_dir = self.root_dir / "distorted_images"

        self.samples = []

        with open(mos_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                mos, image_name = line.split()

                image_path = distorted_dir / image_name

                self.samples.append(
                    (image_path, float(mos))
                )

    def __getitem__(self, idx):
        image_path, mos = self.samples[idx]

        image = self.load_image(image_path)

        return {
            "image": image,
            "mos": mos
        }