from .base_dataset import BaseDataset


class LIVEDataset(BaseDataset):

    def __init__(self, root_dir):
        super().__init__(root_dir)

    def __getitem__(self, idx):
        image_path, mos = self.samples[idx]

        image = self.load_image(image_path)

        return {
            "image": image,
            "mos": mos
        }