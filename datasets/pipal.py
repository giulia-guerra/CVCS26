from pathlib import Path

from .base_dataset import BaseDataset



class PIPALDataset(BaseDataset):


    def __init__(
        self,
        root_dir,
        transform=None,
        split="train"
    ):

        super().__init__(
            root_dir,
            transform
        )


        self.root_dir = Path(root_dir)

        self.split = split

        self.samples = []


        self._load_samples()



    def _load_samples(self):


        if self.split=="train":


            img_dir = (
                self.root_dir /
                "Dist_Imgs"
            )


            ref_dir = (
                self.root_dir /
                "Train_Ref"
            )


            images = sorted(
                img_dir.rglob(
                    "*.bmp"
                )
            )


            if len(images)==0:

                images = sorted(
                    img_dir.rglob(
                        "*.png"
                    )
                )



            labels = (
                self.root_dir /
                "Train_Label"
            )



        else:


            img_dir = (
                self.root_dir /
                "Val_Dist"
            )


            ref_dir = None


            images = sorted(
                img_dir.rglob(
                    "*"
                )
            )

            images = [
                x for x in images
                if x.suffix.lower()
                in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp"
                ]
            ]



        for img in images:


            self.samples.append(
                {

                    "image": img,

                    "reference": None,

                    "mos": 0.0

                }
            )



    def __getitem__(self, idx):


        sample = self.samples[idx]


        distorted = self.load_image(
            sample["image"]
        )


        distorted = self.apply_transform(
            distorted
        )


        return {


    "image": distorted,


    "distorted": distorted,


    "reference": distorted,


    "mos": sample["mos"],


    "name": sample["image"].name,


    "path": str(sample["image"])

}