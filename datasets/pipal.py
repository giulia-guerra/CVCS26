# Dataset loader per PIPAL IQA Full-Reference.
# Carica le immagini distorte dalla cartella Dist_Imgs e associa ogni immagine
# alla relativa reference contenuta in Train_Ref tramite il nome del file.
# I valori MOS vengono recuperati dai file di label presenti in Train_Label.
# Restituisce coppie reference/distorted in formato PIL Image con il relativo MOS
# per essere utilizzate dai modelli di estrazione feature.

from pathlib import Path
from .base_dataset import BaseDataset


class PIPALDataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        split="train",
        return_pil=True
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.root_dir = Path(root_dir)
        self.split = split
        self.return_pil = return_pil

        self.samples = []

        self._load_samples()


    def _load_samples(self):

        if self.split == "train":

            dist_dir = self.root_dir / "Dist_Imgs"
            ref_dir = self.root_dir / "Train_Ref"
            label_dir = self.root_dir / "Train_Label"


            images = sorted(
                dist_dir.glob("*.bmp")
            )


            for img in images:

                # esempio:
                # A0130_02_08.bmp
                # reference:
                # A0130.bmp

                ref_name = (
                    img.stem.split("_")[0]
                    + ".bmp"
                )


                ref_path = ref_dir / ref_name


                if not ref_path.exists():

                    print(
                        "Reference mancante:",
                        ref_path
                    )

                    continue


                # MOS
                mos = 0.0


                label_file = (
                    label_dir /
                    (img.stem.split("_")[0] + ".txt")
                )


                if label_file.exists():

                    with open(label_file) as f:

                        lines = f.readlines()


                    for line in lines:

                        if img.stem in line:

                            # formato PIPAL:
                            # A0001_00_00.bmp,1520.0648

                            value = line.strip().split(",")[-1]

                            mos = float(value)

                            break


                self.samples.append(
                    {
                        "ref_image": ref_path,
                        "dist_image": img,
                        "mos": mos,
                        "name": img.name
                    }
                )



        else:

            dist_dir = (
                self.root_dir /
                "Val_Dist"
            )


            images = sorted(
                [
                    x
                    for x in dist_dir.rglob("*")
                    if x.suffix.lower()
                    in [
                        ".bmp",
                        ".png",
                        ".jpg",
                        ".jpeg"
                    ]
                ]
            )


            for img in images:

                self.samples.append(
                    {
                        "ref_image": None,
                        "dist_image": img,
                        "mos": 0.0,
                        "name": img.name
                    }
                )


        print(
            "PIPAL campioni caricati:",
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
            sample["ref_image"]
        )


        dist_img = self.load_image(
            sample["dist_image"]
        )


        if not self.return_pil and self.transform:

            ref_img = self.transform(ref_img)

            dist_img = self.transform(dist_img)



        return {

            "ref_image": ref_img,

            "dist_image": dist_img,

            "mos": sample["mos"],

            "name": sample["name"]

        }