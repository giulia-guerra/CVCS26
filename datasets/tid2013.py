from pathlib import Path
import numpy as np
import torch

from .base_dataset import BaseDataset



class TID2013Dataset(BaseDataset):


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
            distorted_dir.rglob("*")
        )



        images = [

            x for x in images

            if x.suffix.lower()
            in
            [
                ".bmp",
                ".png",
                ".jpg",
                ".jpeg"
            ]

        ]



        print(
            "TID2013 immagini trovate:",
            len(images)
        )



        for img in images:


            self.samples.append(
                {


                    "image": img,


                    "reference": None,


                    "distorted": img,


                    "name": img.name,


                    # temporaneamente placeholder
                    # verrà collegato ai MOS reali

                    "mos": 0.0


                }
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
            (224,224)
        )



        img = np.array(
            img
        )



        # H,W,C -> C,H,W

        img = torch.from_numpy(
            img
        ).permute(
            2,
            0,
            1
        ).float() / 255.0



        return {


            "image": img,


            "reference": img,


            "distorted": img,


            "name": sample["name"],


            "mos": sample["mos"],


            "path": str(
                sample["image"]
            )

        }