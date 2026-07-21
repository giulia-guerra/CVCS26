from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from torch.utils.data import Dataset



class LIVEDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.images = []
        self.mos = []

        self.load_metadata()



    def load_metadata(self):

        """
        Load LIVE metadata.
        """


        mos_file = self.root / "dmos_realigned.mat"


        data = loadmat(mos_file)



        # cerca automaticamente MOS

        possible_keys = [
            "dmos",
            "dmos_new",
            "DMOS",
            "dmos_realigned"
        ]


        mos_values = None


        for key in possible_keys:

            if key in data:

                mos_values = data[key].flatten()
                break



        if mos_values is None:

            raise KeyError(
                f"MOS non trovato. Chiavi disponibili: {data.keys()}"
            )



        # Cerca immagini anche nelle sottocartelle

        image_files = []


        extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp"
        ]



        for ext in extensions:

            image_files.extend(
                self.root.rglob(ext)
            )



        self.images = sorted(image_files)



        if len(self.images) == 0:

            raise RuntimeError(
                f"Nessuna immagine trovata dentro {self.root}"
            )



        # Mantieni solo immagini con MOS associato

        num_samples = min(
            len(self.images),
            len(mos_values)
        )


        self.images = self.images[:num_samples]


        self.mos = mos_values[:num_samples]



        print(
            f"LIVE dataset: {len(self.images)} immagini trovate"
        )



    def load_image(self, path):

        image = Image.open(path).convert("RGB")


        image = image.resize(
            (224,224)
        )


        image = np.array(
            image,
            dtype=np.uint8
        )


        return image



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        image = self.load_image(
            self.images[idx]
        )


        mos = float(
            self.mos[idx]
        )


        return {
            "image": image,
            "mos": mos
        }