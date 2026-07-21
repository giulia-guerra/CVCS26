import os
import glob

from datasets.base_dataset import BaseDataset


class PIPALDataset(BaseDataset):
    """
    PIPAL IQA Dataset.

    Structure:

    PIPAL/
    |
    |-- Dist_Imgs/
    |
    |-- Train_Label/
    |     |-- A0001.txt
    |     |-- A0002.txt
    |
    |-- val_label.txt


    Returns:

    {
        "image": distorted image tensor,
        "mos": quality score,
        "path": image path
    }

    """


    def __init__(
        self,
        root_dir,
        transform=None,
        split="train"
    ):

        super().__init__(transform)


        self.root_dir = root_dir

        self.split = split


        self.image_dir = os.path.join(
            root_dir,
            "Dist_Imgs"
        )


        self.train_label_dir = os.path.join(
            root_dir,
            "Train_Label"
        )


        self.val_label_file = os.path.join(
            root_dir,
            "val_label.txt"
        )


        if not os.path.exists(self.image_dir):

            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )


        self.samples = []


        self._load_labels()



        print(
            f"PIPAL {split}: {len(self.samples)} immagini trovate"
        )



    def _find_image(self, name):

        """
        Trova immagine case insensitive.
        """

        path = os.path.join(
            self.image_dir,
            name
        )


        if os.path.exists(path):

            return path



        lower_name = name.lower()


        for file in os.listdir(self.image_dir):

            if file.lower() == lower_name:

                return os.path.join(
                    self.image_dir,
                    file
                )


        return None




    def _load_labels(self):


        if self.split == "train":


            label_files = glob.glob(
                os.path.join(
                    self.train_label_dir,
                    "*.txt"
                )
            )


            for label_file in label_files:


                with open(
                    label_file,
                    "r"
                ) as f:


                    for line in f:


                        line = line.strip()


                        if not line:
                            continue



                        parts = line.split(",")


                        image_name = parts[0].strip()


                        mos = float(
                            parts[1].strip()
                        )



                        image_path = self._find_image(
                            image_name
                        )


                        if image_path is None:

                            continue



                        self.samples.append(
                            {
                                "image": image_path,
                                "mos": mos
                            }
                        )



        elif self.split == "val":



            if not os.path.exists(
                self.val_label_file
            ):

                raise FileNotFoundError(
                    self.val_label_file
                )



            with open(
                self.val_label_file,
                "r"
            ) as f:



                for line in f:


                    line=line.strip()


                    if not line:
                        continue



                    parts=line.split(",")



                    image_name = parts[0].strip()



                    mos=float(
                        parts[1].strip()
                    )



                    image_path=self._find_image(
                        image_name
                    )


                    if image_path is None:

                        continue



                    self.samples.append(
                        {
                            "image": image_path,
                            "mos": mos
                        }
                    )



        else:


            raise ValueError(
                "split deve essere 'train' oppure 'val'"
            )




    def __len__(self):

        return len(self.samples)




    def __getitem__(self, idx):


        sample=self.samples[idx]


        image=self.load_image(
            sample["image"]
        )


        image=self.apply_transform(
            image
        )



        return {

            "image": image,

            "mos": sample["mos"],

            "path": sample["image"]

        }