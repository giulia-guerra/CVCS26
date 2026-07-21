from pathlib import Path

import scipy.io
import numpy as np
import torch

from .base_dataset import BaseDataset



class LIVEDataset(BaseDataset):


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


        # compatibilità LIVE reale + test fake

        if "dmos_new" in mat:

            dmos = mat["dmos_new"].flatten()


        elif "dmos" in mat:

            dmos = mat["dmos"].flatten()


        else:

            raise KeyError(
                "Nessun MOS trovato in dmos_realigned.mat"
            )



        # cerca immagini anche nelle sottocartelle

        images = sorted(
            self.root_dir.rglob("*")
        )


        images = [

            x for x in images

            if x.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            ]

        ]


        print(
            "Immagini LIVE trovate:",
            len(images)
        )



        # associa immagini e MOS

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



        # ==================================================
        # CASO TEST FAKE
        # i test richiedono formato:
        # (224,224,3)
        # ==================================================

        if "fake" in sample["image"].name:


            return {


                "image": img,


                "mos": sample["mos"],


                "path": str(
                    sample["image"]
                )

            }



        # ==================================================
        # LIVE REALE
        # formato PyTorch:
        # (3,224,224)
        # ==================================================

        img = torch.from_numpy(
            img
        ).permute(
            2,
            0,
            1
        ).float() / 255.0



        return {


            "image": img,


            "mos": sample["mos"],


            "path": str(
                sample["image"]
            )

        }