# Dataset loader per TID2013 IQA Full-Reference.
# Carica le immagini distorte dalla cartella distorted_images e ricostruisce
# la corrispondente reference originale tramite il nome del file.
# I valori MOS vengono letti da mos_with_names.txt.
# Restituisce per ogni campione l'immagine reference, l'immagine distorted
# in formato PIL Image e il relativo MOS per la fase di feature extraction.

from pathlib import Path
from .base_dataset import BaseDataset


class TID2013Dataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        return_pil=True
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.root_dir = Path(root_dir)
        self.return_pil = return_pil

        self.samples = []

        self._load_samples()


    def _load_samples(self):

        distorted_dir = (
            self.root_dir /
            "distorted_images"
        )

        reference_dir = (
            self.root_dir /
            "reference_images"
        )

        mos_file = (
            self.root_dir /
            "mos_with_names.txt"
        )


        # -----------------------------
        # MOS mapping
        # -----------------------------

        mos_dict = {}

        with open(
            mos_file,
            "r"
        ) as f:

            for line in f:

                value, name = line.split()

                mos_dict[
                    name.lower()
                ] = float(value)



        # -----------------------------
        # Reference mapping
        # -----------------------------

        refs = {}

        for ref in reference_dir.iterdir():

            if ref.suffix.lower() in [
                ".bmp",
                ".png",
                ".jpg",
                ".jpeg"
            ]:

                refs[
                    ref.stem.lower()
                ] = ref



        # -----------------------------
        # Distorted images
        # -----------------------------

        images = [

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


        print(
            "TID2013 immagini trovate:",
            len(images)
        )



        # -----------------------------
        # costruzione samples
        # -----------------------------

        for img in images:

            name = img.name.lower()


            if name not in mos_dict:

                continue



            # esempio:
            # i01_01_1.bmp
            #
            # reference:
            # i01.bmp

            ref_id = (
                img.stem
                .split("_")[0]
                .lower()
            )


            if ref_id not in refs:

                raise FileNotFoundError(
                    f"Reference non trovata per {img.name}"
                )


            self.samples.append(
                {

                    "ref_path": refs[ref_id],

                    "dist_path": img,

                    "mos": mos_dict[name],

                    "name": img.name

                }
            )



        print(
            "Campioni TID2013 caricati:",
            len(self.samples)
        )



    def __len__(self):

        return len(
            self.samples
        )



    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]


        ref_img = self.load_image(
            sample["ref_path"]
        )


        dist_img = self.load_image(
            sample["dist_path"]
        )



        if (
            not self.return_pil
            and self.transform is not None
        ):

            ref_img = self.transform(
                ref_img
            )

            dist_img = self.transform(
                dist_img
            )


        return {

            "ref_image": ref_img,

            "dist_image": dist_img,

            "mos": sample["mos"],

            "name": sample["name"]

        }