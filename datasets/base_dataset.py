from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BaseDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, mos = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "mos": mos
        }