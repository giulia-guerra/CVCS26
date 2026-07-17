from pathlib import Path
from PIL import Image
import numpy as np


class BaseDataset:
    """
    Classe base per i dataset IQA.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Le sottoclassi devono implementare __getitem__"
        )